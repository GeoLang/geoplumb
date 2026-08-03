//! disk tier for the chunk cache: a flat binary file per chunk. entries
//! live for one engine instance, the engine owns a unique subdir and
//! removes it on drop, so files never outlive the caps they were computed
//! under

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::caps::Crs;
use crate::chunk::RasterChunk;
use crate::error::{Error, Result};
use crate::window::{Bbox, ChunkKey};
use terrano_core::{BandedRaster, Raster};

const MAGIC: u32 = 0x47504C43;
const VERSION: u16 = 1;

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

pub(crate) fn write_chunk(path: &Path, chunk: &RasterChunk) -> Result<()> {
    let io = |e: std::io::Error| Error::Source(format!("spill write: {e}"));
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).map_err(io)?);
    let (cols, rows) = (chunk.width() as u32, chunk.height() as u32);
    f.write_all(&MAGIC.to_le_bytes()).map_err(io)?;
    f.write_all(&VERSION.to_le_bytes()).map_err(io)?;
    f.write_all(&(chunk.bands.band_count() as u16).to_le_bytes())
        .map_err(io)?;
    f.write_all(&cols.to_le_bytes()).map_err(io)?;
    f.write_all(&rows.to_le_bytes()).map_err(io)?;
    f.write_all(&chunk.resolution.to_le_bytes()).map_err(io)?;
    for v in [
        chunk.bbox.min_x,
        chunk.bbox.min_y,
        chunk.bbox.max_x,
        chunk.bbox.max_y,
    ] {
        f.write_all(&v.to_le_bytes()).map_err(io)?;
    }
    f.write_all(&chunk.crs.0.to_le_bytes()).map_err(io)?;
    for band in chunk.bands.bands() {
        f.write_all(&band.nodata.to_le_bytes()).map_err(io)?;
        for v in band.data() {
            f.write_all(&v.to_le_bytes()).map_err(io)?;
        }
    }
    f.flush().map_err(io)
}

pub(crate) fn read_chunk(path: &Path) -> Result<RasterChunk> {
    let io = |e: std::io::Error| Error::Source(format!("spill read: {e}"));
    let bad = |what: &str| Error::Source(format!("spill file corrupt: {what}"));
    let mut f = std::io::BufReader::new(std::fs::File::open(path).map_err(io)?);
    let mut u32b = [0u8; 4];
    let mut u16b = [0u8; 2];
    let mut f64b = [0u8; 8];
    let mut read_u32 = |f: &mut dyn Read| -> Result<u32> {
        f.read_exact(&mut u32b).map_err(io)?;
        Ok(u32::from_le_bytes(u32b))
    };
    if read_u32(&mut f)? != MAGIC {
        return Err(bad("magic"));
    }
    f.read_exact(&mut u16b).map_err(io)?;
    if u16::from_le_bytes(u16b) != VERSION {
        return Err(bad("version"));
    }
    f.read_exact(&mut u16b).map_err(io)?;
    let band_count = u16::from_le_bytes(u16b) as usize;
    let cols = read_u32(&mut f)? as usize;
    let rows = read_u32(&mut f)? as usize;
    let mut read_f64 = |f: &mut dyn Read| -> Result<f64> {
        f.read_exact(&mut f64b).map_err(io)?;
        Ok(f64::from_le_bytes(f64b))
    };
    let resolution = read_f64(&mut f)?;
    let bbox = Bbox {
        min_x: read_f64(&mut f)?,
        min_y: read_f64(&mut f)?,
        max_x: read_f64(&mut f)?,
        max_y: read_f64(&mut f)?,
    };
    let crs = Crs(read_u32(&mut f)?);
    if band_count == 0 || cols == 0 || rows == 0 || cols * rows > (1 << 28) {
        return Err(bad("geometry"));
    }
    let mut bands = Vec::with_capacity(band_count);
    for _ in 0..band_count {
        let nodata = read_f64(&mut f)?;
        let mut raw = vec![0u8; cols * rows * 8];
        f.read_exact(&mut raw).map_err(io)?;
        let data: Vec<f64> = raw
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().expect("8-byte chunks")))
            .collect();
        bands.push(Raster::from_vec(cols, rows, data, resolution, nodata).map_err(Error::Terrano)?);
    }
    Ok(RasterChunk {
        bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
        bbox,
        resolution,
        crs,
    })
}
