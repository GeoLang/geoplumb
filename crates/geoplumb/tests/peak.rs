use std::io::Cursor;
use std::sync::Arc;

use geoplumb::caps::{BandKinds, ValueKind};
use geoplumb::element::Source;
use geoplumb::elements::{CogSrc, Mosaic, RasterSrc, Reproject};
use geoplumb::export::{ExportGrid, write_cog};
use geoplumb::resample::resample_to_grid;
use geoplumb::{Bbox, Crs, Engine, Graph, NodeId, RasterBands, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, CogParams, CogReader, Raster};

const W: usize = 1100;
const H: usize = 600;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

const LINE_COL: usize = 41;
const PEAK: f64 = 1000.0;
const BACKGROUND: f64 = 1.0;

const MERCATOR_METERS_PER_DEGREE: f64 = 20037508.342789244 / 180.0;

fn line_raster(width: usize, height: usize, line_col: usize) -> Raster {
    let mut data = Vec::with_capacity(width * height);
    for _ in 0..height {
        for col in 0..width {
            data.push(if col == line_col { PEAK } else { BACKGROUND });
        }
    }
    Raster::from_vec(width, height, data, CELL, f64::NAN).unwrap()
}

fn line_src(origin_x: f64, origin_y: f64, value_kind: ValueKind) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![line_raster(W, H, LINE_COL)]).unwrap(),
        origin_x,
        origin_y,
        Crs::WGS84,
        value_kind,
    )
}

fn empty_src(value_kind: ValueKind) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![
            Raster::from_vec(W, H, vec![f64::NAN; W * H], CELL, f64::NAN).unwrap(),
        ])
        .unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
        value_kind,
    )
}

fn window(px0: usize, py0: usize, px1: usize, py1: usize) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + px0 as f64 * CELL,
        max_x: ORIGIN_X + px1 as f64 * CELL,
        max_y: ORIGIN_Y - py0 as f64 * CELL,
        min_y: ORIGIN_Y - py1 as f64 * CELL,
    }
}

fn engine_over(source: Box<dyn Source>) -> (Engine, NodeId) {
    let mut g = Graph::new();
    let node = g.add_source(source);
    (Engine::new(g, 64 << 20).unwrap(), node)
}

async fn pull(engine: &Engine, node: NodeId, bbox: Bbox, resolution: f64) -> Raster {
    engine
        .pull(
            node,
            WindowReq {
                bbox,
                resolution,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap()
        .bands
        .band(0)
        .unwrap()
        .to_raster()
        .into_owned()
}

fn rows_reaching_the_peak(raster: &Raster) -> usize {
    raster
        .data()
        .chunks(raster.width())
        .filter(|row| row.contains(&PEAK))
        .count()
}

fn highest(raster: &Raster) -> f64 {
    raster
        .data()
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .fold(f64::NEG_INFINITY, f64::max)
}

fn line_cog(dir: &std::path::Path) -> std::path::PathBuf {
    let path = dir.join("line.tif");
    let params = CogParams {
        tile_width: 64,
        tile_height: 64,
        // terrano averages the overviews it writes
        overview_levels: 0,
        epsg: 4326,
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        deflate: false,
        ..CogParams::default()
    };
    terrano_core::write_cog(
        &line_raster(64, 64, LINE_COL),
        &params,
        &mut std::fs::File::create(&path).unwrap(),
    )
    .unwrap();
    path
}

#[tokio::test]
async fn a_peak_line_keeps_its_value_through_a_2x_decimation_where_continuous_halves_it() {
    let dir = tempfile::tempdir().unwrap();
    let cog = line_cog(dir.path());
    let decimated = |source: Box<dyn Source>| async move {
        let (engine, node) = engine_over(source);
        pull(&engine, node, window(0, 0, 64, 64), 2.0 * CELL).await
    };
    let open_cog = |value_kind: ValueKind| -> Box<dyn Source> {
        Box::new(CogSrc::open(std::fs::File::open(&cog).unwrap(), value_kind, None).unwrap())
    };

    for (what, peak) in [
        (
            "RasterSrc",
            decimated(Box::new(line_src(ORIGIN_X, ORIGIN_Y, ValueKind::Peak))).await,
        ),
        ("CogSrc", decimated(open_cog(ValueKind::Peak)).await),
    ] {
        assert_eq!(peak.width(), 32, "{what}");
        assert_eq!(rows_reaching_the_peak(&peak), 32, "{what}");
    }

    for (what, continuous) in [
        (
            "RasterSrc",
            decimated(Box::new(line_src(
                ORIGIN_X,
                ORIGIN_Y,
                ValueKind::Continuous,
            )))
            .await,
        ),
        ("CogSrc", decimated(open_cog(ValueKind::Continuous)).await),
    ] {
        assert_eq!(
            highest(&continuous),
            (PEAK + BACKGROUND) / 2.0,
            "{what}: a 2x2 block mean over a one-cell line"
        );
    }
}

fn mercator_y(latitude: f64) -> f64 {
    6378137.0
        * (std::f64::consts::FRAC_PI_4 + latitude.to_radians() / 2.0)
            .tan()
            .ln()
}

#[tokio::test]
async fn a_peak_line_keeps_its_value_through_a_reprojection_sampling_up_2x() {
    // web mercator x is longitude times a constant
    let resolution = CELL * MERCATOR_METERS_PER_DEGREE / 2.0;
    let origin_x = ORIGIN_X * MERCATOR_METERS_PER_DEGREE;
    let origin_y = mercator_y(ORIGIN_Y - 0.1);
    let first_col = (2 * LINE_COL - 16) as f64;
    let bbox = Bbox {
        min_x: origin_x + first_col * resolution,
        max_x: origin_x + (first_col + 32.0) * resolution,
        max_y: origin_y,
        min_y: origin_y - 16.0 * resolution,
    };
    let warped = |value_kind| async move {
        let mut g = Graph::new();
        let src = g.add_source(Box::new(line_src(ORIGIN_X, ORIGIN_Y, value_kind)));
        let rp = g.add_transform(
            src,
            Box::new(Reproject::onto_grid(
                Crs::WEB_MERCATOR,
                origin_x,
                origin_y,
                resolution,
            )),
        );
        let engine = Engine::new(g, 64 << 20).unwrap();
        assert_eq!(
            engine.caps(rp).raster().value_kinds,
            BandKinds::from(value_kind)
        );
        pull(&engine, rp, bbox, resolution).await
    };

    let peak = warped(ValueKind::Peak).await;
    assert!(peak.height() > 0);
    assert_eq!(rows_reaching_the_peak(&peak), peak.height());

    let continuous = warped(ValueKind::Continuous).await;
    assert!(
        highest(&continuous) < 0.8 * PEAK,
        "bilinear sampling a quarter cell off the line reached {}",
        highest(&continuous)
    );
}

#[test]
fn a_peak_line_keeps_its_value_through_a_resample_sampling_down() {
    // at 1.75 input cells per output cell the nearest input cell to an output center skips column 8
    let (side, line_col, out_side) = (56, 8, 32);
    let bbox = window(0, 0, side, side);
    let input = RasterChunk {
        bands: RasterBands::from_rasters(vec![line_raster(side, side, line_col)]).unwrap(),
        bbox,
        resolution: CELL,
        crs: Crs::WGS84,
    };
    let resampled = |value_kind| {
        resample_to_grid(
            &input,
            &bbox,
            out_side,
            out_side,
            &BandKinds::from(value_kind),
        )
        .bands
        .band(0)
        .unwrap()
        .to_raster()
        .into_owned()
    };

    assert_eq!(
        rows_reaching_the_peak(&resampled(ValueKind::Peak)),
        out_side
    );
    assert_eq!(
        rows_reaching_the_peak(&resampled(ValueKind::Categorical)),
        0
    );
    assert!(highest(&resampled(ValueKind::Continuous)) < 0.5 * PEAK);
}

#[tokio::test]
async fn a_mosaic_keeps_a_peak_line_where_continuous_blends_it() {
    let mosaic = |value_kind| async move {
        let mut g = Graph::new();
        let empty = g.add_source(Box::new(empty_src(value_kind)));
        let shifted = g.add_source(Box::new(line_src(
            ORIGIN_X + CELL / 4.0,
            ORIGIN_Y - CELL / 4.0,
            value_kind,
        )));
        let joined = g.add_fanin(&[empty, shifted], Box::new(Mosaic::default()));
        let engine = Engine::new(g, 64 << 20).unwrap();
        assert_eq!(
            engine.caps(joined).raster().value_kinds,
            BandKinds::from(value_kind)
        );
        pull(&engine, joined, window(8, 8, 96, 72), CELL).await
    };

    let peak = mosaic(ValueKind::Peak).await;
    assert_eq!(peak.height(), 64);
    assert_eq!(rows_reaching_the_peak(&peak), peak.height());

    let continuous = mosaic(ValueKind::Continuous).await;
    assert!(
        highest(&continuous) < 0.8 * PEAK,
        "the quarter-cell shift blends nothing: {}",
        highest(&continuous)
    );
}

fn written_level(bytes: &[u8], level: usize) -> Raster {
    let reader = CogReader::open(bytes).unwrap();
    let shape = &reader.levels()[level];
    reader
        .read_window(level, 0, 0, shape.width, shape.height)
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_export_keeps_a_peak_line_on_every_overview() {
    let grid = ExportGrid {
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_size: CELL,
        cols: W,
        rows: H,
    };
    let export = |value_kind| async move {
        let (engine, node) = engine_over(Box::new(line_src(ORIGIN_X, ORIGIN_Y, value_kind)));
        write_cog(Arc::new(engine), node, grid, None, Cursor::new(Vec::new()))
            .await
            .expect("the export must write")
            .into_inner()
    };

    let bytes = export(ValueKind::Peak).await;
    let levels = CogReader::open(bytes.as_slice()).unwrap().levels().len();
    assert_eq!(levels, 3);
    for level in 0..levels {
        let written = written_level(&bytes, level);
        assert_eq!(
            rows_reaching_the_peak(&written),
            written.height(),
            "level {level}"
        );
    }

    let continuous = export(ValueKind::Continuous).await;
    assert!(highest(&written_level(&continuous, 1)) < 0.6 * PEAK);
}
