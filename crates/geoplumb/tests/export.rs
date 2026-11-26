use std::io::Cursor;
use std::sync::Arc;

use geoplumb::caps::ValueKind;
use geoplumb::elements::{RasterSrc, Slope};
use geoplumb::export::{ExportGrid, write_cog};
use geoplumb::resample::resample_to_grid;
use geoplumb::{Crs, Engine, Graph, NodeId, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, CogReader, Raster};

const W: usize = 1200;
const H: usize = 800;
const ORIGIN_X: f64 = 380_000.0;
const ORIGIN_Y: f64 = 5_190_000.0;
const CELL: f64 = 10.0;
const UTM_32N: Crs = Crs(32632);

const CODES: [f64; 4] = [10.0, 20.0, 30.0, 80.0];

// 1100 x 700 is no multiple of 512, so the edge tiles of every level are part padding
fn export_grid() -> ExportGrid {
    ExportGrid {
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_size: CELL,
        cols: 1100,
        rows: 700,
    }
}

fn rough_dem(band: usize) -> Raster {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let value = if row < 9 && col < 13 {
                f64::NAN
            } else {
                1000.0 * band as f64
                    + 500.0
                    + 120.0 * (col as f64 * 0.9).sin() * (row as f64 * 0.7).cos()
                    + 0.05 * col as f64
            };
            data.push(value);
        }
    }
    Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
}

fn class_map() -> Raster {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let code = if row < 5 && col < 7 {
                f64::NAN
            } else {
                CODES[(col / 37 + row / 23 + col / 5 % 2 + row % 3) % CODES.len()]
            };
            data.push(code);
        }
    }
    Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
}

fn graph_over(bands: Vec<Raster>, value_kind: ValueKind) -> (Graph, NodeId) {
    let src = RasterSrc::new(
        BandedRaster::new(bands).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        UTM_32N,
        value_kind,
    );
    let mut graph = Graph::new();
    let node = graph.add_source(Box::new(src));
    (graph, node)
}

fn engine_over(bands: Vec<Raster>, value_kind: ValueKind) -> (Arc<Engine>, NodeId) {
    let (graph, node) = graph_over(bands, value_kind);
    (Arc::new(Engine::new(graph, 256 << 20).unwrap()), node)
}

async fn export(engine: &Arc<Engine>, node: NodeId) -> Vec<u8> {
    write_cog(
        Arc::clone(engine),
        node,
        export_grid(),
        None,
        Cursor::new(Vec::new()),
    )
    .await
    .expect("the export must write")
    .into_inner()
}

async fn engine_at(
    engine: &Engine,
    node: NodeId,
    grid: ExportGrid,
    value_kind: ValueKind,
) -> RasterChunk {
    let bbox = grid.bbox();
    let pulled = engine
        .pull(
            node,
            WindowReq {
                bbox,
                resolution: grid.pixel_size,
                time: None,
            },
        )
        .await
        .unwrap();
    resample_to_grid(
        pulled.raster().unwrap(),
        &bbox,
        grid.cols,
        grid.rows,
        value_kind,
    )
}

fn written_level(bytes: &[u8], level: usize) -> BandedRaster {
    let reader = CogReader::open(bytes).unwrap();
    let shape = &reader.levels()[level];
    reader
        .read_window_bands(level, 0, 0, shape.width, shape.height)
        .unwrap()
}

fn assert_same_pixels(written: &Raster, expected: &Raster, what: &str) {
    assert_eq!(
        (written.width(), written.height()),
        (expected.width(), expected.height()),
        "{what}: shape"
    );
    let mut valid = 0;
    for (index, (got, want)) in written.data().iter().zip(expected.data()).enumerate() {
        if expected.is_nodata(*want) {
            assert!(
                got.is_nan(),
                "{what}: pixel {index} is {got}, expected nodata"
            );
            continue;
        }
        assert!(
            (got - want).abs() <= 1e-9 * want.abs().max(1.0),
            "{what}: pixel {index} is {got}, expected {want}"
        );
        valid += 1;
    }
    assert!(valid > 0, "{what}: no valid pixel to compare");
}

fn two_by_two_means(level: &Raster) -> Vec<f64> {
    let (cols, rows) = (level.width().div_ceil(2), level.height().div_ceil(2));
    let mut means = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let block: Vec<f64> = [(0usize, 0usize), (0, 1), (1, 0), (1, 1)]
                .into_iter()
                .filter_map(|(down, across)| level.get(2 * row + down, 2 * col + across))
                .filter(|value| value.is_finite())
                .collect();
            let mean = block.iter().sum::<f64>() / block.len() as f64;
            means.push(mean);
        }
    }
    means
}

#[tokio::test(flavor = "multi_thread")]
async fn the_base_level_is_the_engine_pull_resampled_onto_the_grid() {
    let (engine, node) = engine_over(vec![rough_dem(0)], ValueKind::Continuous);
    let bytes = export(&engine, node).await;

    let reader = CogReader::open(bytes.as_slice()).unwrap();
    let meta = reader.meta();
    assert_eq!(meta.epsg, 32632);
    assert_eq!((meta.origin_x, meta.origin_y), (ORIGIN_X, ORIGIN_Y));
    assert_eq!((meta.pixel_width, meta.pixel_height), (CELL, CELL));

    let expected = engine_at(&engine, node, export_grid(), ValueKind::Continuous).await;
    assert_same_pixels(
        &written_level(&bytes, 0).bands()[0],
        &expected.bands.bands()[0],
        "base level",
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn every_overview_is_pulled_from_the_engine_at_its_own_resolution() {
    let (mut graph, dem) = graph_over(vec![rough_dem(0)], ValueKind::Continuous);
    let slope = graph.add_transform(dem, Box::new(Slope));
    let engine = Arc::new(Engine::new(graph, 256 << 20).unwrap());
    let bytes = export(&engine, slope).await;

    let levels = CogReader::open(bytes.as_slice()).unwrap().levels().len();
    assert_eq!(levels, 3, "1100 px wide wants two overviews");
    for level in 0..levels {
        let expected = engine_at(
            &engine,
            slope,
            export_grid().level(level),
            ValueKind::Continuous,
        )
        .await;
        assert_same_pixels(
            &written_level(&bytes, level).bands()[0],
            &expected.bands.bands()[0],
            &format!("level {level}"),
        );
    }

    // slope is not linear in elevation, so a block mean of the fine slopes is not the coarse slope
    let base = written_level(&bytes, 0);
    let overview = written_level(&bytes, 1);
    let largest_gap = two_by_two_means(&base.bands()[0])
        .iter()
        .zip(overview.bands()[0].data())
        .filter(|(mean, pulled)| mean.is_finite() && pulled.is_finite())
        .map(|(mean, pulled)| (mean - pulled).abs())
        .fold(0.0, f64::max);
    assert!(
        largest_gap > 1.0,
        "the overview is within {largest_gap} degrees of the averaged base level"
    );
}

fn off_code_count(bytes: &[u8], level: usize) -> usize {
    written_level(bytes, level).bands()[0]
        .data()
        .iter()
        .filter(|value| !value.is_nan() && !CODES.contains(value))
        .count()
}

#[tokio::test(flavor = "multi_thread")]
async fn a_categorical_export_holds_only_codes_at_every_level() {
    let (engine, node) = engine_over(vec![class_map()], ValueKind::Categorical);
    let bytes = export(&engine, node).await;

    let levels = CogReader::open(bytes.as_slice()).unwrap().levels().len();
    assert_eq!(levels, 3);
    for level in 0..levels {
        let written = written_level(&bytes, level);
        assert_eq!(
            off_code_count(&bytes, level),
            0,
            "level {level} holds blended codes"
        );
        let expected = engine_at(
            &engine,
            node,
            export_grid().level(level),
            ValueKind::Categorical,
        )
        .await;
        assert_same_pixels(
            &written.bands()[0],
            &expected.bands.bands()[0],
            &format!("level {level}"),
        );
    }

    let blended = two_by_two_means(&written_level(&bytes, 0).bands()[0])
        .iter()
        .filter(|mean| mean.is_finite() && !CODES.contains(mean))
        .count();
    assert!(blended > 0, "averaging this fixture blends no codes");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_export_off_the_source_grid_resamples_by_the_value_kind() {
    // half a cell off the source origin and 15 m pixels over 10 m cells, so every tile is resampled
    let grid = ExportGrid {
        origin_x: ORIGIN_X + CELL / 2.0,
        origin_y: ORIGIN_Y - CELL / 2.0,
        pixel_size: 15.0,
        cols: 700,
        rows: 500,
    };
    let mut off_codes = Vec::new();
    for value_kind in [ValueKind::Categorical, ValueKind::Continuous] {
        let (engine, node) = engine_over(vec![class_map()], value_kind);
        let bytes = write_cog(engine, node, grid, None, Cursor::new(Vec::new()))
            .await
            .expect("the export must write")
            .into_inner();
        let levels = CogReader::open(bytes.as_slice()).unwrap().levels().len();
        assert!(levels > 1, "{levels} levels");
        let per_level: Vec<usize> = (0..levels)
            .map(|level| off_code_count(&bytes, level))
            .collect();
        off_codes.push(per_level);
    }
    assert!(
        off_codes[0].iter().all(|count| *count == 0),
        "categorical export blends codes: {:?}",
        off_codes[0]
    );
    assert!(
        off_codes[1][0] > 0,
        "the continuous control blends no codes, so the check shows nothing"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn a_multi_band_export_keeps_its_band_order() {
    let (engine, node) = engine_over(
        vec![rough_dem(0), rough_dem(1), rough_dem(2)],
        ValueKind::Continuous,
    );
    let bytes = export(&engine, node).await;

    let reader = CogReader::open(bytes.as_slice()).unwrap();
    assert_eq!(reader.levels()[0].samples, 3);
    for level in 0..reader.levels().len() {
        let written = written_level(&bytes, level);
        let first = written.bands()[0].data();
        for band in 1..3 {
            let mut checked = 0;
            for (base, lifted) in first.iter().zip(written.bands()[band].data()) {
                if !base.is_finite() {
                    assert!(
                        lifted.is_nan(),
                        "level {level} band {band} has a value in a hole"
                    );
                    continue;
                }
                assert!(
                    (lifted - base - 1000.0 * band as f64).abs() < 1e-6,
                    "level {level} band {band}: {lifted} over {base}"
                );
                checked += 1;
            }
            assert!(checked > 0, "level {level} band {band} holds no value");
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_grid_off_the_tile_size_keeps_the_pixels_of_its_edge_tiles() {
    let (engine, node) = engine_over(vec![class_map()], ValueKind::Categorical);
    let bytes = export(&engine, node).await;

    let reader = CogReader::open(bytes.as_slice()).unwrap();
    let shapes: Vec<(usize, usize)> = reader
        .levels()
        .iter()
        .map(|level| (level.width, level.height))
        .collect();
    assert_eq!(shapes, [(1100, 700), (550, 350), (275, 175)]);
    for level in reader.levels() {
        assert_eq!((level.tile_width, level.tile_height), (512, 512));
    }

    // the base level's corner tile holds the last 76 columns and 188 rows, the rest of it padding
    let corner = ExportGrid {
        origin_x: ORIGIN_X + 1024.0 * CELL,
        origin_y: ORIGIN_Y - 512.0 * CELL,
        pixel_size: CELL,
        cols: 76,
        rows: 188,
    };
    let written = reader.read_window_bands(0, 1024, 512, 76, 188).unwrap();
    let expected = engine_at(&engine, node, corner, ValueKind::Categorical).await;
    assert_same_pixels(
        &written.bands()[0],
        &expected.bands.bands()[0],
        "corner tile",
    );
}
