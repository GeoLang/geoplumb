//! disk tier for the chunk cache: a flat binary file per chunk, in a
//! directory named by the graph identity. the directory outlives the
//! engine, so the next engine over the same graph starts from its files
//! and an edited graph never sees them

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::SystemTime;

use crate::bands::{RasterBands, decoded_band, encoded_band};
use crate::caps::Crs;
use crate::chunk::{
    Chunk, PointChunk, RasterChunk, TensorChunk, TensorData, VectorChunk, VectorFeature,
};
use crate::error::{Error, Result};
use crate::window::{Bbox, ChunkKey, TimeInterval};
use nubis_core::{Classification, Point3, PointCloud};
use terrano_core::SampleFormat;
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{
    Coord, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, Ring,
};

const MAGIC: u32 = 0x47504C43;
const VERSION: u16 = 3;
const KIND_RASTER: u8 = 0;
const KIND_POINTS: u8 = 1;
const KIND_VECTOR: u8 = 2;
const KIND_TENSOR: u8 = 3;

const GEOM_POINT: u8 = 0;
const GEOM_LINESTRING: u8 = 1;
const GEOM_POLYGON: u8 = 2;
const GEOM_MULTIPOLYGON: u8 = 3;
const GEOM_MULTIPOINT: u8 = 4;
const GEOM_MULTILINESTRING: u8 = 5;
const GEOM_COLLECTION: u8 = 6;

/// 64-bit fnv-1a, unlike std's hashers the same in every process
#[derive(Debug)]
pub(crate) struct Fnv1a(u64);

impl Fnv1a {
    pub(crate) fn new() -> Fnv1a {
        Fnv1a(0xcbf29ce484222325)
    }

    pub(crate) fn update(&mut self, bytes: &[u8]) {
        for byte in bytes {
            self.0 ^= u64::from(*byte);
            self.0 = self.0.wrapping_mul(0x100000001b3);
        }
    }

    pub(crate) fn finish(&self) -> u64 {
        self.0
    }
}

impl Write for Fnv1a {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.update(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub(crate) fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hasher = Fnv1a::new();
    hasher.update(bytes);
    hasher.finish()
}

pub(crate) struct SpillStore {
    dir: PathBuf,
}

/// a chunk file a store directory held when it was opened
pub(crate) struct StoredChunk {
    pub(crate) node: usize,
    pub(crate) key: ChunkKey,
    pub(crate) bytes: usize,
    pub(crate) modified: SystemTime,
}

impl SpillStore {
    /// opens `dir`, creating it when missing, and returns the chunk files
    /// it holds. anything else in it, partial writes included, is deleted
    pub(crate) fn open(dir: PathBuf) -> Result<(SpillStore, Vec<StoredChunk>)> {
        let io = |e: std::io::Error| Error::Source(format!("spill open {}: {e}", dir.display()));
        std::fs::create_dir_all(&dir).map_err(io)?;
        let mut stored = Vec::new();
        for entry in std::fs::read_dir(&dir).map_err(io)? {
            let entry = entry.map_err(io)?;
            let metadata = entry.metadata().map_err(io)?;
            if !metadata.is_file() {
                continue;
            }
            let parsed = entry.file_name().to_str().and_then(parse_chunk_file_name);
            let Some((node, key)) = parsed else {
                let _ = std::fs::remove_file(entry.path());
                continue;
            };
            stored.push(StoredChunk {
                node,
                key,
                bytes: metadata.len() as usize,
                modified: metadata.modified().map_err(io)?,
            });
        }
        Ok((SpillStore { dir }, stored))
    }

    pub(crate) fn path(&self, node: usize, key: ChunkKey) -> PathBuf {
        self.dir.join(chunk_file_name(node, key))
    }
}

fn chunk_file_name(node: usize, key: ChunkKey) -> String {
    let mut name = format!("{node}_{}_{}_{}", key.level, key.ix, key.iy);
    if let Some(t) = key.time {
        name.push_str(&format!("_{}_{}", t.start_ms, t.end_ms));
    }
    name.push_str(".bin");
    name
}

/// the inverse of `chunk_file_name`, `None` for any name it would not write
fn parse_chunk_file_name(name: &str) -> Option<(usize, ChunkKey)> {
    let parts: Vec<&str> = name.strip_suffix(".bin")?.split('_').collect();
    let time = match parts.len() {
        4 => None,
        6 => Some(TimeInterval::new(
            parts[4].parse().ok()?,
            parts[5].parse().ok()?,
        )),
        _ => return None,
    };
    let node = parts[0].parse().ok()?;
    let key = ChunkKey {
        level: parts[1].parse().ok()?,
        ix: parts[2].parse().ok()?,
        iy: parts[3].parse().ok()?,
        time,
    };
    // a name with leading zeros or a plus sign parses but names another path
    (chunk_file_name(node, key) == name).then_some((node, key))
}

/// writes under a temporary name and renames into place, so a chunk name
/// never holds a partial file
pub(crate) fn write_chunk(path: &Path, chunk: &Chunk) -> Result<()> {
    // per write, so two engines over one graph never write the same temp file
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let partial = path.with_extension(format!(
        "{}-{}.tmp",
        std::process::id(),
        SEQ.fetch_add(1, Ordering::Relaxed)
    ));
    let written = write_chunk_file(&partial, chunk).and_then(|()| {
        std::fs::rename(&partial, path).map_err(|e| Error::Source(format!("spill write: {e}")))
    });
    if written.is_err() {
        let _ = std::fs::remove_file(&partial);
    }
    written
}

fn write_chunk_file(path: &Path, chunk: &Chunk) -> Result<()> {
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
                f.write_all(&[format_tag(band.format())]).map_err(io)?;
                f.write_all(&encoded_band(band)).map_err(io)?;
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
        Chunk::Vector(chunk) => {
            f.write_all(&[KIND_VECTOR]).map_err(io)?;
            f.write_all(&(chunk.features.len() as u64).to_le_bytes())
                .map_err(io)?;
            write_geo(&mut f, chunk.resolution, &chunk.bbox, chunk.crs).map_err(io)?;
            for feature in &chunk.features {
                f.write_all(&feature.id.to_le_bytes()).map_err(io)?;
                write_geometry(&mut f, &feature.geometry).map_err(io)?;
                let props = serde_json::to_vec(&feature.properties)
                    .map_err(|e| Error::Source(format!("spill write: {e}")))?;
                f.write_all(&(props.len() as u32).to_le_bytes())
                    .map_err(io)?;
                f.write_all(&props).map_err(io)?;
            }
        }
        Chunk::Tensor(chunk) => {
            f.write_all(&[KIND_TENSOR]).map_err(io)?;
            f.write_all(&chunk.channels.to_le_bytes()).map_err(io)?;
            f.write_all(&(chunk.width() as u32).to_le_bytes())
                .map_err(io)?;
            f.write_all(&(chunk.height() as u32).to_le_bytes())
                .map_err(io)?;
            write_geo(&mut f, chunk.resolution, &chunk.bbox, chunk.crs).map_err(io)?;
            for v in chunk.system() {
                f.write_all(&v.to_le_bytes()).map_err(io)?;
            }
        }
    }
    f.flush().map_err(io)
}

fn write_coords(f: &mut impl Write, coords: &[Coord]) -> std::io::Result<()> {
    f.write_all(&(coords.len() as u32).to_le_bytes())?;
    for c in coords {
        f.write_all(&c.x.to_le_bytes())?;
        f.write_all(&c.y.to_le_bytes())?;
    }
    Ok(())
}

pub(crate) fn write_polygon(f: &mut impl Write, p: &Polygon) -> std::io::Result<()> {
    f.write_all(&(1 + p.interiors().len() as u32).to_le_bytes())?;
    write_coords(f, p.exterior().coords())?;
    for hole in p.interiors() {
        write_coords(f, hole.coords())?;
    }
    Ok(())
}

pub(crate) fn write_geometry(
    f: &mut impl Write,
    geometry: &FeatureGeometry,
) -> std::io::Result<()> {
    match geometry {
        FeatureGeometry::Point(p) => {
            f.write_all(&[GEOM_POINT])?;
            f.write_all(&p.0.x.to_le_bytes())?;
            f.write_all(&p.0.y.to_le_bytes())
        }
        FeatureGeometry::MultiPoint(mp) => {
            f.write_all(&[GEOM_MULTIPOINT])?;
            f.write_all(&(mp.points().len() as u32).to_le_bytes())?;
            for p in mp.points() {
                f.write_all(&p.0.x.to_le_bytes())?;
                f.write_all(&p.0.y.to_le_bytes())?;
            }
            Ok(())
        }
        FeatureGeometry::LineString(l) => {
            f.write_all(&[GEOM_LINESTRING])?;
            write_coords(f, l.coords())
        }
        FeatureGeometry::MultiLineString(mls) => {
            f.write_all(&[GEOM_MULTILINESTRING])?;
            f.write_all(&(mls.linestrings().len() as u32).to_le_bytes())?;
            for l in mls.linestrings() {
                write_coords(f, l.coords())?;
            }
            Ok(())
        }
        FeatureGeometry::Polygon(p) => {
            f.write_all(&[GEOM_POLYGON])?;
            write_polygon(f, p)
        }
        FeatureGeometry::MultiPolygon(mp) => {
            f.write_all(&[GEOM_MULTIPOLYGON])?;
            f.write_all(&(mp.polygons().len() as u32).to_le_bytes())?;
            for p in mp.polygons() {
                write_polygon(f, p)?;
            }
            Ok(())
        }
        FeatureGeometry::GeometryCollection(members) => {
            f.write_all(&[GEOM_COLLECTION])?;
            f.write_all(&(members.len() as u32).to_le_bytes())?;
            for m in members {
                write_geometry(f, m)?;
            }
            Ok(())
        }
    }
}

fn format_tag(format: SampleFormat) -> u8 {
    let position = SampleFormat::ALL.iter().position(|known| *known == format);
    position.expect("SampleFormat::ALL lists every format") as u8
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
                let tag = rd::<1>(&mut f).map_err(io)?[0];
                let format = *SampleFormat::ALL
                    .get(usize::from(tag))
                    .ok_or(bad("sample format"))?;
                let sample_bytes = format.bytes() as usize;
                let has_nodata = rd::<1>(&mut f).map_err(io)?[0] != 0;
                let mut nodata = vec![0u8; sample_bytes];
                f.read_exact(&mut nodata).map_err(io)?;
                let mut samples = vec![0u8; cols * rows * sample_bytes];
                f.read_exact(&mut samples).map_err(io)?;
                bands.push(decoded_band(
                    format,
                    has_nodata.then_some(nodata.as_slice()),
                    &samples,
                    cols,
                    rows,
                    resolution,
                ));
            }
            Ok(Chunk::Raster(RasterChunk {
                bands: RasterBands::new(bands)?,
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
        KIND_VECTOR => {
            let count = u64::from_le_bytes(rd(&mut f).map_err(io)?) as usize;
            let (resolution, bbox, crs) = rd_geo(&mut f).map_err(io)?;
            if count > (1 << 24) {
                return Err(bad("feature count"));
            }
            let mut features = Vec::with_capacity(count);
            for _ in 0..count {
                let id = u64::from_le_bytes(rd(&mut f).map_err(io)?);
                let geometry = read_geometry(&mut f)
                    .map_err(io)?
                    .ok_or(bad("geometry tag"))?;
                let len = rd_u32(&mut f).map_err(io)? as usize;
                if len > (1 << 24) {
                    return Err(bad("properties length"));
                }
                let mut raw = vec![0u8; len];
                f.read_exact(&mut raw).map_err(io)?;
                let properties = serde_json::from_slice(&raw)
                    .map_err(|e| Error::Source(format!("spill read: {e}")))?;
                features.push(VectorFeature {
                    id,
                    geometry,
                    properties,
                });
            }
            Ok(Chunk::Vector(VectorChunk::new(
                features, bbox, resolution, crs,
            )))
        }
        KIND_TENSOR => {
            let channels = u16::from_le_bytes(rd(&mut f).map_err(io)?);
            let cols = rd_u32(&mut f).map_err(io)? as usize;
            let rows = rd_u32(&mut f).map_err(io)? as usize;
            let (resolution, bbox, crs) = rd_geo(&mut f).map_err(io)?;
            let len = (channels as usize)
                .saturating_mul(cols)
                .saturating_mul(rows);
            if channels == 0 || cols == 0 || rows == 0 || len > (1 << 28) {
                return Err(bad("geometry"));
            }
            let mut raw = vec![0u8; len * 4];
            f.read_exact(&mut raw).map_err(io)?;
            let data: Vec<f32> = raw
                .as_chunks::<4>()
                .0
                .iter()
                .map(|c| f32::from_le_bytes(*c))
                .collect();
            Ok(Chunk::Tensor(TensorChunk {
                data: TensorData::System(data),
                channels,
                bbox,
                resolution,
                crs,
            }))
        }
        _ => Err(bad("kind")),
    }
}

fn read_coords(f: &mut impl Read) -> std::io::Result<Vec<Coord>> {
    let n = rd_u32(f)? as usize;
    let mut coords = Vec::with_capacity(n.min(1 << 20));
    for _ in 0..n {
        coords.push(Coord::new(rd_f64(f)?, rd_f64(f)?));
    }
    Ok(coords)
}

fn read_polygon(f: &mut impl Read) -> std::io::Result<Polygon> {
    let rings = rd_u32(f)? as usize;
    let exterior = Ring::new(read_coords(f)?);
    let mut holes = Vec::with_capacity(rings.saturating_sub(1).min(1 << 16));
    for _ in 1..rings {
        holes.push(Ring::new(read_coords(f)?));
    }
    Ok(Polygon::new(exterior, holes))
}

fn read_geometry(f: &mut impl Read) -> std::io::Result<Option<FeatureGeometry>> {
    Ok(match rd::<1>(f)?[0] {
        GEOM_POINT => Some(FeatureGeometry::Point(Point::new(rd_f64(f)?, rd_f64(f)?))),
        GEOM_MULTIPOINT => {
            let n = rd_u32(f)? as usize;
            let mut points = Vec::with_capacity(n.min(1 << 20));
            for _ in 0..n {
                points.push(Point::new(rd_f64(f)?, rd_f64(f)?));
            }
            Some(FeatureGeometry::MultiPoint(MultiPoint::new(points)))
        }
        GEOM_LINESTRING => Some(FeatureGeometry::LineString(LineString::new(read_coords(
            f,
        )?))),
        GEOM_MULTILINESTRING => {
            let n = rd_u32(f)? as usize;
            let mut lines = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                lines.push(LineString::new(read_coords(f)?));
            }
            Some(FeatureGeometry::MultiLineString(MultiLineString::new(
                lines,
            )))
        }
        GEOM_POLYGON => Some(FeatureGeometry::Polygon(read_polygon(f)?)),
        GEOM_MULTIPOLYGON => {
            let n = rd_u32(f)? as usize;
            let mut polys = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                polys.push(read_polygon(f)?);
            }
            Some(FeatureGeometry::MultiPolygon(MultiPolygon::new(polys)))
        }
        GEOM_COLLECTION => {
            let n = rd_u32(f)? as usize;
            let mut members = Vec::with_capacity(n.min(1 << 16));
            for _ in 0..n {
                match read_geometry(f)? {
                    Some(m) => members.push(m),
                    None => return Ok(None),
                }
            }
            Some(FeatureGeometry::GeometryCollection(members))
        }
        _ => None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caps::Dtype;
    use terrano_core::TypedBand;

    #[test]
    fn fnv1a_matches_the_reference_vectors() {
        assert_eq!(fnv1a(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a(b"a"), 0xaf63dc4c8601ec8c);
    }

    #[test]
    fn chunk_file_names_round_trip() {
        let key = ChunkKey {
            level: 3,
            ix: -2,
            iy: 7,
            time: Some(TimeInterval::new(-5, 10)),
        };
        let name = chunk_file_name(4, key);
        assert_eq!(parse_chunk_file_name(&name), Some((4, key)));
        assert_eq!(parse_chunk_file_name("04_3_-2_7.bin"), None);
        assert_eq!(parse_chunk_file_name("4_3_-2_7.123-0.tmp"), None);
    }

    fn grid<T>(values: Vec<T>) -> terrano_core::SampleGrid<T> {
        terrano_core::SampleGrid::new(3, 2, 0.5, values).unwrap()
    }

    fn typed_chunk() -> RasterChunk {
        let bands = vec![
            TypedBand::U8 {
                samples: grid(vec![0u8, 1, 2, 3, 4, 255]),
                nodata: Some(0),
            },
            TypedBand::I16 {
                samples: grid(vec![-32768i16, -1, 0, 1, 32767, 7]),
                nodata: None,
            },
            TypedBand::F32(grid(vec![1.5f32, f32::NAN, -0.25, 0.0, 3.0, 1e-3])),
            TypedBand::F64(
                terrano_core::Raster::from_vec(
                    3,
                    2,
                    vec![2.5, -9999.0, f64::NAN, 0.0, 1.0, 2.0],
                    0.5,
                    -9999.0,
                )
                .unwrap(),
            ),
        ];
        RasterChunk {
            bands: RasterBands::new(bands).unwrap(),
            bbox: Bbox::new(10.0, 20.0, 11.5, 21.0),
            resolution: 0.5,
            crs: Crs(32632),
        }
    }

    #[test]
    fn typed_raster_chunks_round_trip_through_a_spill_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("0_0_0_0.bin");
        let chunk = typed_chunk();
        write_chunk(&path, &Chunk::Raster(chunk.clone())).unwrap();
        let back = read_chunk(&path).unwrap().into_raster().unwrap();

        assert_eq!(
            (back.bbox, back.resolution, back.crs),
            (chunk.bbox, 0.5, chunk.crs)
        );
        assert_eq!(back.byte_size(), chunk.byte_size());
        for (written, read) in chunk.bands.bands().iter().zip(back.bands.bands()) {
            assert_eq!(Dtype::of(read), Dtype::of(written));
            let (written, read) = (written.to_raster(), read.to_raster());
            assert_eq!(read.nodata.to_bits(), written.nodata.to_bits());
            for (got, want) in read.data().iter().zip(written.data()) {
                assert!(got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan()));
            }
        }
        let TypedBand::U8 { samples, .. } = back.bands.band(0).unwrap() else {
            panic!("the u8 band came back as another format");
        };
        assert_eq!(samples.values(), &[0, 1, 2, 3, 4, 255]);
    }

    #[test]
    fn a_version_two_spill_file_fails_the_version_check() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("0_0_0_0.bin");
        let mut bytes = MAGIC.to_le_bytes().to_vec();
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.push(KIND_RASTER);
        std::fs::write(&path, bytes).unwrap();
        let error = read_chunk(&path).expect_err("an old file");
        assert!(error.to_string().contains("version"), "{error}");
    }
}
