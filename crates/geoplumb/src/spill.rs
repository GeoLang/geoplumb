//! disk tier for the chunk cache: a flat binary file per chunk. entries
//! live for one engine instance, the engine owns a unique subdir and
//! removes it on drop, so files never outlive the caps they were computed
//! under

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::caps::Crs;
use crate::chunk::{Chunk, PointChunk, RasterChunk};
use crate::error::{Error, Result};
use crate::window::{Bbox, ChunkKey};
use nubis_core::{Classification, Point3, PointCloud};
use terrano_core::{BandedRaster, Raster};

const MAGIC: u32 = 0x47504C43;
const VERSION: u16 = 2;
const KIND_RASTER: u8 = 0;
const KIND_POINTS: u8 = 1;

pub(crate) struct SpillStore {
    dir: PathBuf,
}

impl SpillStore {
    /// creates a unique subdir of `base` this store owns outright
    pub(crate) fn create(base: &Path) -> Result<SpillStore> {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let dir = base.join(format!(
            "geoplumb-spill-{}-{}",
            std::process::id(),
            SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).map_err(|e| Error::Source(e.to_string()))?;
        Ok(SpillStore { dir })
    }

    pub(crate) fn path(&self, node: usize, key: ChunkKey) -> PathBuf {
        self.dir
            .join(format!("{node}_{}_{}_{}.bin", key.level, key.ix, key.iy))
    }
}

impl Drop for SpillStore {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

pub(crate) fn write_chunk(path: &Path, chunk: &Chunk) -> Result<()> {
    let io = |e: std::io::Error| Error::Source(format!("spill write: {e}"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
    f.write_all(&MAGIC.to_le_bytes()).map_err(io)?;
    f.write_all(&VERSION.to_le_bytes()).map_err(io)?;
    match chunk {
        Chunk::Raster(chunk) => {
            f.write_all(&[KIND_RASTER]).map_err(io)?;
            let (cols, rows) = (chunk.width() as u32, chunk.height() as u32);
            f.write_all(&(chunk.bands.band_count() as u16).to_le_bytes())
                .map_err(io)?;
            f.write_all(&cols.to_le_bytes()).map_err(io)?;
            f.write_all(&rows.to_le_bytes()).map_err(io)?;
            write_geo(&mut f, chunk.resolution, &chunk.bbox, chunk.crs).map_err(io)?;
            for band in chunk.bands.bands() {
                f.write_all(&band.nodata.to_le_bytes()).map_err(io)?;
                for v in band.data() {
                    f.write_all(&v.to_le_bytes()).map_err(io)?;
                }
            }
        }
        Chunk::PointCloud(chunk) => {
            f.write_all(&[KIND_POINTS]).map_err(io)?;
            f.write_all(&(chunk.points.len() as u64).to_le_bytes())
                .map_err(io)?;
            write_geo(&mut f, chunk.resolution, &chunk.bbox, chunk.crs).map_err(io)?;
            for p in chunk.points.points() {
                for v in [p.x, p.y, p.z] {
                    f.write_all(&v.to_le_bytes()).map_err(io)?;
                }
                f.write_all(&p.intensity.to_le_bytes()).map_err(io)?;
                f.write_all(&[p.classification.to_u8()]).map_err(io)?;
            }
        }
    }
    f.flush().map_err(io)
}

fn write_geo(f: &mut impl Write, resolution: f64, bbox: &Bbox, crs: Crs) -> std::io::Result<()> {
    f.write_all(&resolution.to_le_bytes())?;
    for v in [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y] {
        f.write_all(&v.to_le_bytes())?;
    }
    f.write_all(&crs.0.to_le_bytes())
}

fn rd<const N: usize>(f: &mut impl Read) -> std::io::Result<[u8; N]> {
    let mut b = [0u8; N];
    f.read_exact(&mut b)?;
    Ok(b)
}

fn rd_u32(f: &mut impl Read) -> std::io::Result<u32> {
    Ok(u32::from_le_bytes(rd(f)?))
}

fn rd_f64(f: &mut impl Read) -> std::io::Result<f64> {
    Ok(f64::from_le_bytes(rd(f)?))
}

fn rd_geo(f: &mut impl Read) -> std::io::Result<(f64, Bbox, Crs)> {
    let resolution = rd_f64(f)?;
    let bbox = Bbox {
        min_x: rd_f64(f)?,
        min_y: rd_f64(f)?,
        max_x: rd_f64(f)?,
        max_y: rd_f64(f)?,
    };
    let crs = Crs(rd_u32(f)?);
    Ok((resolution, bbox, crs))
}

pub(crate) fn read_chunk(path: &Path) -> Result<Chunk> {
    let io = |e: std::io::Error| Error::Source(format!("spill read: {e}"));
    let bad = |what: &str| Error::Source(format!("spill file corrupt: {what}"));
    let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(io)?);
    if rd_u32(&mut f).map_err(io)? != MAGIC {
        return Err(bad("magic"));
    }
    if u16::from_le_bytes(rd(&mut f).map_err(io)?) != VERSION {
        return Err(bad("version"));
    }
    let kind = rd::<1>(&mut f).map_err(io)?[0];
    match kind {
        KIND_RASTER => {
            let band_count = u16::from_le_bytes(rd(&mut f).map_err(io)?) as usize;
            let cols = rd_u32(&mut f).map_err(io)? as usize;
            let rows = rd_u32(&mut f).map_err(io)? as usize;
            let (resolution, bbox, crs) = rd_geo(&mut f).map_err(io)?;
            if band_count == 0 || cols == 0 || rows == 0 || cols * rows > (1 << 28) {
                return Err(bad("geometry"));
            }
            let mut bands = Vec::with_capacity(band_count);
            for _ in 0..band_count {
                let nodata = rd_f64(&mut f).map_err(io)?;
                let mut raw = vec![0u8; cols * rows * 8];
                f.read_exact(&mut raw).map_err(io)?;
                let data: Vec<f64> = raw
                    .chunks_exact(8)
                    .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunks")))
                    .collect();
                bands.push(
                    Raster::from_vec(cols, rows, data, resolution, nodata)
                        .map_err(Error::Terrano)?,
                );
            }
            Ok(Chunk::Raster(RasterChunk {
                bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
                bbox,
                resolution,
                crs,
            }))
        }
        KIND_POINTS => {
            let count = u64::from_le_bytes(rd(&mut f).map_err(io)?) as usize;
            let (resolution, bbox, crs) = rd_geo(&mut f).map_err(io)?;
            if count > (1 << 28) {
                return Err(bad("point count"));
            }
            let mut points = Vec::with_capacity(count);
            for _ in 0..count {
                let x = rd_f64(&mut f).map_err(io)?;
                let y = rd_f64(&mut f).map_err(io)?;
                let z = rd_f64(&mut f).map_err(io)?;
                let intensity = u16::from_le_bytes(rd(&mut f).map_err(io)?);
                let class = rd::<1>(&mut f).map_err(io)?[0];
                points.push(
                    Point3::new(x, y, z)
                        .with_intensity(intensity)
                        .with_classification(Classification::from_u8(class)),
                );
            }
            Ok(Chunk::PointCloud(PointChunk {
                points: PointCloud::from_points(points),
                bbox,
                resolution,
                crs,
            }))
        }
        _ => Err(bad("kind")),
    }
}
