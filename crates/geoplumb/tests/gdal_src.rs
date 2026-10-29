#![cfg(feature = "gdal")]

use std::path::{Path, PathBuf};

use gdal::raster::Buffer;
use gdal::spatial_ref::SpatialRef;
use gdal::{Dataset, DatasetOptions, DriverManager, GdalOpenFlags};
use geoplumb::element::Source;
use geoplumb::elements::{GdalSrc, RasterSrc};
use geoplumb::{Bbox, Crs, Engine, Graph, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, GeoTiffMetadata, Raster, SampleFormat, write_geotiff_bands};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;
const BANDS: usize = 3;
const OVERVIEW_FACTORS: [i32; 2] = [2, 4];

// whole 8 px blocks, so every ladder level up to 8 cells sees a block as
// all hole or no hole and an average of averages equals the flat average
const BLOCK_HOLE_ROWS: std::ops::Range<usize> = 40..56;
const BLOCK_HOLE_COLS: std::ops::Range<usize> = 80..96;
const SECOND_BLOCK_HOLE_ROWS: std::ops::Range<usize> = 104..120;
const SECOND_BLOCK_HOLE_COLS: std::ops::Range<usize> = 200..216;
// single pixels below this row, partial blocks for the 2x overview only
const PIXEL_HOLES_FROM_ROW: usize = 300;
const PIXEL_HOLES: [(usize, usize); 3] = [(301, 451), (310, 503), (333, 477)];

/// nan holes in bands 0 and 2, none in band 1
fn banded() -> BandedRaster {
    let planes = (0..BANDS)
        .map(|b| {
            let mut data = Vec::with_capacity(W * H);
            for row in 0..H {
                for col in 0..W {
                    let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
                    let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
                    let v = 500.0 + 200.0 * ((lon * 8.0).sin() * (lat * 8.0).cos());
                    data.push(v * (b as f64 + 1.0));
                }
            }
            let (rows, cols) = match b {
                0 => (BLOCK_HOLE_ROWS, BLOCK_HOLE_COLS),
                2 => (SECOND_BLOCK_HOLE_ROWS, SECOND_BLOCK_HOLE_COLS),
                _ => return Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap(),
            };
            for row in rows {
                for col in cols.clone() {
                    data[row * W + col] = f64::NAN;
                }
            }
            for (row, col) in PIXEL_HOLES {
                data[row * W + col] = f64::NAN;
            }
            Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
        })
        .collect();
    BandedRaster::new(planes).unwrap()
}

fn geotiff_bytes() -> Vec<u8> {
    let meta = GeoTiffMetadata {
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        epsg: 4326,
    };
    let mut bytes = Vec::new();
    write_geotiff_bands(&banded(), &meta, SampleFormat::F64, &mut bytes).unwrap();
    bytes
}

/// the terrano striped file with average overviews, plus the bytes
/// terrano wrote before gdal touched it
fn fixture(dir: &Path) -> (PathBuf, Vec<u8>) {
    let bytes = geotiff_bytes();
    let path = dir.join("striped.tif");
    std::fs::write(&path, &bytes).unwrap();
    let mut dataset = Dataset::open_ex(
        &path,
        DatasetOptions {
            open_flags: GdalOpenFlags::GDAL_OF_UPDATE | GdalOpenFlags::GDAL_OF_RASTER,
            ..Default::default()
        },
    )
    .unwrap();
    // gdal's average counts a nan as a value unless the file says nan is
    // nodata, and terrano writes no nodata tag
    for b in 1..=BANDS {
        dataset
            .rasterband(b)
            .unwrap()
            .set_no_data_value(Some(f64::NAN))
            .unwrap();
    }
    dataset
        .build_overviews("AVERAGE", &OVERVIEW_FACTORS, &[])
        .unwrap();
    drop(dataset);

    let reopened = Dataset::open(&path).unwrap();
    for b in 1..=BANDS {
        let overviews = reopened.rasterband(b).unwrap().overview_count().unwrap();
        assert!(
            overviews >= OVERVIEW_FACTORS.len() as i32,
            "band {b} has {overviews} overviews"
        );
    }
    (path, bytes)
}

/// pixel bounds in base pixels, which may run past the file on any side
fn window(px0: i64, py0: i64, px1: i64, py1: i64) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + px0 as f64 * CELL,
        max_x: ORIGIN_X + px1 as f64 * CELL,
        max_y: ORIGIN_Y - py0 as f64 * CELL,
        min_y: ORIGIN_Y - py1 as f64 * CELL,
    }
}

fn request(bbox: Bbox, resolution: f64) -> WindowReq {
    WindowReq {
        bbox,
        resolution,
        time: None,
    }
}

async fn read(src: &impl Source, req: &WindowReq) -> RasterChunk {
    src.read(req).await.unwrap().into_raster().unwrap()
}

/// `RasterSrc` writes the -9999 nodata terrano reads files with where
/// `GdalSrc` writes nan
fn memory_value(band: &Raster, v: f64) -> f64 {
    if band.is_nodata(v) { f64::NAN } else { v }
}

fn assert_matches_memory(gdal: &RasterChunk, memory: &RasterChunk, relative_tolerance: f64) {
    assert_eq!(gdal.width(), memory.width());
    assert_eq!(gdal.height(), memory.height());
    assert_eq!(gdal.bands.band_count(), memory.bands.band_count());
    let width = gdal.width();
    for (b, (gb, mb)) in gdal
        .bands
        .bands()
        .iter()
        .zip(memory.bands.bands())
        .enumerate()
    {
        for (i, (&g, &m)) in gb.data().iter().zip(mb.data()).enumerate() {
            let m = memory_value(mb, m);
            let (row, col) = (i / width, i % width);
            assert_eq!(
                g.is_nan(),
                m.is_nan(),
                "band {b} row {row} col {col}: gdal {g} vs memory {m}"
            );
            if g.is_nan() {
                continue;
            }
            assert!(
                (g - m).abs() <= relative_tolerance * m.abs(),
                "band {b} row {row} col {col}: gdal {g} vs memory {m}"
            );
        }
    }
}

#[tokio::test]
async fn base_level_matches_in_memory_source_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let (path, bytes) = fixture(dir.path());
    let src = GdalSrc::open(&path).unwrap();
    let memory = RasterSrc::from_geotiff(&bytes).unwrap();
    assert_eq!(src.driver(), "GTiff");

    // 16 px past every edge, so the clamped read and the nan padding run
    let req = request(window(-16, -16, W as i64 + 16, H as i64 + 16), CELL);
    let a = read(&src, &req).await;
    let b = read(&memory, &req).await;
    assert_eq!(a.crs, Crs::WGS84);
    assert_matches_memory(&a, &b, 0.0);
}

#[tokio::test]
async fn overview_level_matches_block_average() {
    let dir = tempfile::tempdir().unwrap();
    let (path, bytes) = fixture(dir.path());
    let src = GdalSrc::open(&path).unwrap();
    let memory = RasterSrc::from_geotiff(&bytes).unwrap();

    let req = request(window(-16, -16, W as i64 + 16, H as i64 + 16), CELL * 2.0);
    let a = read(&src, &req).await;
    let b = read(&memory, &req).await;
    assert_matches_memory(&a, &b, 1e-9);
}

#[tokio::test]
async fn coarser_than_the_pyramid_decimates_the_coarsest_overview() {
    let dir = tempfile::tempdir().unwrap();
    let (path, bytes) = fixture(dir.path());
    let src = GdalSrc::open(&path).unwrap();
    let memory = RasterSrc::from_geotiff(&bytes).unwrap();

    // 8 cells is 2x past the 4x overview. the window stops at the last
    // whole block above the single pixel holes, where an average of
    // averages is weighted differently from one flat average
    let last_row = (PIXEL_HOLES_FROM_ROW / 8 * 8) as i64;
    let req = request(window(-16, 0, W as i64 + 16, last_row), CELL * 8.0);
    let a = read(&src, &req).await;
    let b = read(&memory, &req).await;
    assert_eq!(a.width(), (W + 32) / 8);
    assert_matches_memory(&a, &b, 1e-9);
}

#[tokio::test]
async fn one_band_serves_alone() {
    let dir = tempfile::tempdir().unwrap();
    let (path, bytes) = fixture(dir.path());
    let src = GdalSrc::open_band(&path, 3).unwrap();
    let memory = RasterSrc::from_geotiff(&bytes).unwrap();

    let req = request(window(0, 0, W as i64, H as i64), CELL);
    let a = read(&src, &req).await;
    let b = read(&memory, &req).await;
    assert_eq!(a.bands.band_count(), 1);
    let only_third = RasterChunk {
        bands: BandedRaster::new(vec![b.bands.band(2).unwrap().clone()]).unwrap(),
        ..b
    };
    assert_matches_memory(&a, &only_third, 0.0);

    let err = GdalSrc::open_band(&path, 4)
        .err()
        .expect("band 4 of 3 opened");
    assert!(err.to_string().contains("3 bands"), "{err}");
}

const SMALL: usize = 16;

/// a one-band gtiff written through gdal, which can carry what terrano
/// cannot, still open for the caller to write into
fn gdal_tiff(path: &Path, transform: [f64; 6], epsg: Option<u32>) -> Dataset {
    let driver = DriverManager::get_driver_by_name("GTiff").unwrap();
    let mut dataset = driver
        .create_with_band_type::<f64, _>(path, SMALL, SMALL, 1)
        .unwrap();
    dataset.set_geo_transform(&transform).unwrap();
    if let Some(code) = epsg {
        dataset
            .set_spatial_ref(&SpatialRef::from_epsg(code).unwrap())
            .unwrap();
    }
    dataset
}

const NORTH_UP: [f64; 6] = [ORIGIN_X, CELL, 0.0, ORIGIN_Y, 0.0, -CELL];

#[tokio::test]
async fn nodata_value_reads_as_nan() {
    const NODATA: f64 = -9999.0;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("nodata.tif");
    let values: Vec<f64> = (0..SMALL * SMALL)
        .map(|i| if i % 5 == 0 { NODATA } else { i as f64 })
        .collect();
    {
        let dataset = gdal_tiff(&path, NORTH_UP, Some(4326));
        let mut band = dataset.rasterband(1).unwrap();
        band.set_no_data_value(Some(NODATA)).unwrap();
        let mut buffer = Buffer::new((SMALL, SMALL), values.clone());
        band.write((0, 0), (SMALL, SMALL), &mut buffer).unwrap();
    }

    let src = GdalSrc::open(&path).unwrap();
    let req = request(window(0, 0, SMALL as i64, SMALL as i64), CELL);
    let chunk = read(&src, &req).await;
    for (i, (&got, &written)) in chunk
        .bands
        .band(0)
        .unwrap()
        .data()
        .iter()
        .zip(&values)
        .enumerate()
    {
        if written == NODATA {
            assert!(got.is_nan(), "pixel {i}: nodata read as {got}");
        } else {
            assert_eq!(got, written, "pixel {i}");
        }
    }
}

#[test]
fn rotated_geotransform_fails_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("rotated.tif");
    let rotated = [ORIGIN_X, CELL, CELL / 4.0, ORIGIN_Y, 0.0, -CELL];
    drop(gdal_tiff(&path, rotated, Some(4326)));
    let err = GdalSrc::open(&path).err().expect("rotated file opened");
    let message = err.to_string();
    assert!(message.contains("rotated.tif"), "{message}");
    assert!(message.contains("rotated"), "{message}");
}

#[test]
fn dataset_without_an_epsg_authority_fails_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("unreferenced.tif");
    drop(gdal_tiff(&path, NORTH_UP, None));
    let err = GdalSrc::open(&path).err().expect("file with no crs opened");
    assert!(err.to_string().contains("unreferenced.tif"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_chunks_match_one_read() {
    let dir = tempfile::tempdir().unwrap();
    let (path, _) = fixture(dir.path());
    let mut graph = Graph::new();
    let node = graph.add_source(Box::new(GdalSrc::open(&path).unwrap()));
    let engine = Engine::new(graph, 64 << 20).unwrap();

    // 600 by 400 at chunk_px 256 is six chunks, read on blocking tasks
    let req = request(window(0, 0, W as i64, H as i64), CELL);
    let pulled = engine.pull(node, req).await.unwrap().into_raster().unwrap();
    let whole = read(&GdalSrc::open(&path).unwrap(), &req).await;
    assert_eq!(pulled.width(), W);
    for (a, b) in pulled.bands.bands().iter().zip(whole.bands.bands()) {
        for (x, y) in a.data().iter().zip(b.data()) {
            assert!(x == y || (x.is_nan() && y.is_nan()), "{x} vs {y}");
        }
    }
}
