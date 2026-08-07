//! the zonal statistics and time series drivers: cross-tile merging against a
//! brute force reduction of the same pixels, per id separation, nan
//! exclusion, and one reduction per time step

use std::collections::HashMap;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use geoplumb::caps::{CapsPattern, CapsSet, Constraint, Dtype, RasterPattern, ResRange, SetField};
use geoplumb::element::Source;
use geoplumb::elements::RasterSrc;
use geoplumb::window::GridSpec;
use geoplumb::{
    Bbox, Chunk, Crs, Engine, Graph, NodeId, PixelStatistics, RasterChunk, Statistic, TimeInterval,
    VectorFeature, WindowReq, window_statistics, window_time_series, zonal_statistics,
    zonal_time_series,
};
use terrano_core::{BandedRaster, Raster};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{Coord, Polygon, Ring};

const WIDTH: usize = 600;
const HEIGHT: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;
const CHUNK_PX: usize = 256;

/// every pixel a distinct exactly representable value, so a sum over any
/// pixel set is exact in f64 and comparisons can be equalities
fn elevation(col: usize, row: usize) -> f64 {
    col as f64 + 1000.0 * row as f64
}

/// a hole of nodata inside the feature, to prove nan pixels drop out
const HOLE_COLS: Range<usize> = 100..110;
const HOLE_ROWS: Range<usize> = 300..310;

fn holed_elevation(col: usize, row: usize) -> f64 {
    if HOLE_COLS.contains(&col) && HOLE_ROWS.contains(&row) {
        f64::NAN
    } else {
        elevation(col, row)
    }
}

fn dem(sample: impl Fn(usize, usize) -> f64) -> RasterSrc {
    let mut data = Vec::with_capacity(WIDTH * HEIGHT);
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            data.push(sample(col, row));
        }
    }
    let band = Raster::from_vec(WIDTH, HEIGHT, data, CELL, f64::NAN).unwrap();
    RasterSrc::new(
        BandedRaster::new(vec![band]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

/// the pixel window a bbox spans, in source pixel indices
fn window(cols: Range<usize>, rows: Range<usize>) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + cols.start as f64 * CELL,
        max_x: ORIGIN_X + cols.end as f64 * CELL,
        max_y: ORIGIN_Y - rows.start as f64 * CELL,
        min_y: ORIGIN_Y - rows.end as f64 * CELL,
    }
}

fn request(bbox: Bbox) -> WindowReq {
    WindowReq {
        bbox,
        resolution: CELL,
        time: None,
    }
}

/// a rectangle on pixel boundaries, so the burn covers exactly `cols` by
/// `rows` and the expectation is countable by hand
fn rectangle(id: u64, cols: Range<usize>, rows: Range<usize>) -> VectorFeature {
    let bbox = window(cols, rows);
    let corners = [
        (bbox.min_x, bbox.min_y),
        (bbox.max_x, bbox.min_y),
        (bbox.max_x, bbox.max_y),
        (bbox.min_x, bbox.max_y),
        (bbox.min_x, bbox.min_y),
    ];
    let ring = Ring::new(corners.iter().map(|(x, y)| Coord::new(*x, *y)).collect());
    VectorFeature {
        id,
        geometry: FeatureGeometry::Polygon(Polygon::new(ring, vec![])),
        properties: HashMap::new(),
    }
}

/// brute force reduction of the source pixels in a rectangle, the oracle the
/// driver has to match. nothing here touches the driver's own merging
fn rectangle_statistics(
    sample: impl Fn(usize, usize) -> f64,
    cols: Range<usize>,
    rows: Range<usize>,
) -> PixelStatistics {
    let mut count = 0usize;
    let mut sum = 0.0;
    let mut minimum = f64::NAN;
    let mut maximum = f64::NAN;
    for row in rows {
        for col in cols.clone() {
            let value = sample(col, row);
            if value.is_nan() {
                continue;
            }
            count += 1;
            sum += value;
            minimum = minimum.min(value);
            maximum = maximum.max(value);
        }
    }
    PixelStatistics {
        count,
        sum,
        minimum,
        maximum,
    }
}

/// wraps a source to record the window of every read, so a test can see the
/// driver pulled tile by tile instead of one giant window
struct CountingSrc {
    inner: RasterSrc,
    reads: Arc<AtomicUsize>,
    widest: Arc<Mutex<(usize, usize)>>,
}

impl Source for CountingSrc {
    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.inner.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        Box::pin(async move {
            self.reads.fetch_add(1, Ordering::SeqCst);
            let cols = (req.bbox.width() / req.resolution).round() as usize;
            let rows = (req.bbox.height() / req.resolution).round() as usize;
            {
                let mut widest = self.widest.lock().unwrap();
                widest.0 = widest.0.max(cols);
                widest.1 = widest.1.max(rows);
            }
            self.inner.read(req).await
        })
    }
}

/// a time varying source stamping every pixel with the day its pull interval
/// starts on, so a step's reduction names the interval it was computed for
struct ClockSrc;

fn stamp(time: Option<TimeInterval>) -> f64 {
    time.map_or(-1.0, |t| t.start_ms as f64 / 86_400_000.0)
}

impl Source for ClockSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtype: SetField::one(Dtype::F64),
            bands: SetField::one(1),
            crs: SetField::one(Crs::WGS84),
            resolution: ResRange::at_least(CELL),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: ORIGIN_X,
            origin_y: ORIGIN_Y,
            base_resolution: CELL,
            chunk_px: CHUNK_PX as u32,
        }
    }

    fn time_varying(&self) -> bool {
        true
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        Box::pin(async move {
            let cols = (req.bbox.width() / req.resolution).round() as usize;
            let rows = (req.bbox.height() / req.resolution).round() as usize;
            let band = Raster::from_vec(
                cols,
                rows,
                vec![stamp(req.time); cols * rows],
                req.resolution,
                f64::NAN,
            )
            .unwrap();
            Ok(Chunk::Raster(RasterChunk {
                bands: BandedRaster::new(vec![band]).unwrap(),
                bbox: req.bbox,
                resolution: req.resolution,
                crs: Crs::WGS84,
            }))
        })
    }
}

fn engine_over(source: Box<dyn Source>) -> (Engine, NodeId) {
    let mut graph = Graph::new();
    let node = graph.add_source(source);
    // one 256 px f64 chunk is 512 KiB, a budget of two proves the driver
    // never needs the whole window resident
    (Engine::new(graph, 1 << 20).unwrap(), node)
}

fn only(rows: &[geoplumb::FeatureStatistics], feature_id: u64) -> PixelStatistics {
    rows.iter()
        .find(|row| row.feature_id == feature_id)
        .unwrap_or_else(|| panic!("no row for feature {feature_id}"))
        .statistics
}

#[tokio::test]
async fn a_feature_spanning_tiles_aggregates_to_the_whole_window_answer() {
    let reads = Arc::new(AtomicUsize::new(0));
    let widest = Arc::new(Mutex::new((0, 0)));
    let (engine, node) = engine_over(Box::new(CountingSrc {
        inner: dem(elevation),
        reads: reads.clone(),
        widest: widest.clone(),
    }));
    let cols = 10..590;
    let rows = 5..395;
    let feature = rectangle(3, cols.clone(), rows.clone());

    let stats = zonal_statistics(
        &engine,
        node,
        std::slice::from_ref(&feature),
        request(window(0..WIDTH, 0..HEIGHT)),
    )
    .await
    .unwrap();

    let expected = rectangle_statistics(elevation, cols, rows);
    assert_eq!(stats.len(), 1);
    let got = only(&stats, 3);
    assert_eq!(got.count, 580 * 390);
    assert_eq!(got, expected);
    assert_eq!(got.value(Statistic::Count), expected.count as f64);
    assert_eq!(got.value(Statistic::Sum), expected.sum);
    assert_eq!(got.value(Statistic::Minimum), elevation(10, 5));
    assert_eq!(got.value(Statistic::Maximum), elevation(589, 394));
    assert_eq!(
        got.value(Statistic::Mean),
        expected.sum / expected.count as f64
    );

    // 3 by 2 chunks over a 600 by 400 window, none of them wider than a chunk
    assert_eq!(reads.load(Ordering::SeqCst), 6);
    let widest = *widest.lock().unwrap();
    assert!(
        widest.0 <= CHUNK_PX && widest.1 <= CHUNK_PX,
        "pulled a {widest:?} px window, wider than one chunk"
    );
}

#[tokio::test]
async fn two_half_windows_merge_into_the_whole_window_reduction() {
    let (engine, node) = engine_over(Box::new(dem(elevation)));
    let feature = rectangle(3, 10..590, 5..395);
    let features = std::slice::from_ref(&feature);

    // the seam at column 300 falls inside a chunk, so each half ends on an
    // alignment the whole window pull never sees
    let left = zonal_statistics(&engine, node, features, request(window(0..300, 0..HEIGHT)))
        .await
        .unwrap();
    let right = zonal_statistics(
        &engine,
        node,
        features,
        request(window(300..WIDTH, 0..HEIGHT)),
    )
    .await
    .unwrap();
    let whole = zonal_statistics(
        &engine,
        node,
        features,
        request(window(0..WIDTH, 0..HEIGHT)),
    )
    .await
    .unwrap();

    let mut merged = only(&left, 3);
    merged.merge(&only(&right, 3));
    assert_eq!(merged, only(&whole, 3));
    assert!(only(&left, 3).count > 0 && only(&right, 3).count > 0);
}

#[tokio::test]
async fn each_feature_id_gets_its_own_row() {
    let (engine, node) = engine_over(Box::new(dem(elevation)));
    let west = rectangle(7, 20..200, 20..100);
    let east = rectangle(42, 400..560, 250..380);
    let features = vec![west, east];

    let stats = zonal_statistics(
        &engine,
        node,
        &features,
        request(window(0..WIDTH, 0..HEIGHT)),
    )
    .await
    .unwrap();

    assert_eq!(
        stats.iter().map(|row| row.feature_id).collect::<Vec<_>>(),
        vec![7, 42]
    );
    assert_eq!(
        only(&stats, 7),
        rectangle_statistics(elevation, 20..200, 20..100)
    );
    assert_eq!(
        only(&stats, 42),
        rectangle_statistics(elevation, 400..560, 250..380)
    );
    assert_ne!(only(&stats, 7).sum, only(&stats, 42).sum);
}

#[tokio::test]
async fn nan_pixels_leave_the_statistics_alone() {
    let (engine, node) = engine_over(Box::new(dem(holed_elevation)));
    let cols = 50..400;
    let rows = 250..350;
    let feature = rectangle(1, cols.clone(), rows.clone());

    let stats = zonal_statistics(
        &engine,
        node,
        std::slice::from_ref(&feature),
        request(window(0..WIDTH, 0..HEIGHT)),
    )
    .await
    .unwrap();

    let got = only(&stats, 1);
    let holed = rectangle_statistics(holed_elevation, cols.clone(), rows.clone());
    let whole = rectangle_statistics(elevation, cols, rows);
    assert_eq!(got, holed);
    // the hole sits inside the feature, so exactly its pixels are missing
    assert_eq!(got.count, whole.count - HOLE_COLS.len() * HOLE_ROWS.len());
    assert!(got.sum < whole.sum);
    assert_eq!(got.value(Statistic::Mean), got.sum / got.count as f64);
}

#[tokio::test]
async fn a_window_past_the_source_edge_counts_only_real_pixels() {
    let (engine, node) = engine_over(Box::new(dem(elevation)));
    let feature = rectangle(1, 0..WIDTH + 50, 0..HEIGHT + 50);

    let stats = zonal_statistics(
        &engine,
        node,
        std::slice::from_ref(&feature),
        request(window(0..WIDTH + 50, 0..HEIGHT + 50)),
    )
    .await
    .unwrap();

    assert_eq!(
        only(&stats, 1),
        rectangle_statistics(elevation, 0..WIDTH, 0..HEIGHT)
    );
}

#[tokio::test]
async fn a_time_series_reduces_each_step_on_its_own() {
    let (engine, node) = engine_over(Box::new(ClockSrc));
    let feature = rectangle(5, 10..590, 5..395);
    let june = TimeInterval::new(1_717_200_000_000, 1_719_792_000_000);
    let july = TimeInterval::new(1_719_792_000_000, 1_722_470_400_000);

    let series = zonal_time_series(
        &engine,
        node,
        std::slice::from_ref(&feature),
        request(window(0..WIDTH, 0..HEIGHT)),
        &[june, july],
    )
    .await
    .unwrap();

    assert_eq!(
        series.iter().map(|(step, _)| *step).collect::<Vec<_>>(),
        vec![june, july]
    );
    let means: Vec<f64> = series
        .iter()
        .map(|(_, rows)| only(rows, 5).value(Statistic::Mean))
        .collect();
    assert_eq!(means, vec![stamp(Some(june)), stamp(Some(july))]);
    assert_ne!(means[0], means[1]);
    let counts: Vec<usize> = series.iter().map(|(_, rows)| only(rows, 5).count).collect();
    assert_eq!(counts, vec![580 * 390, 580 * 390]);
}

#[tokio::test]
async fn a_whole_window_time_series_needs_no_features() {
    let (engine, node) = engine_over(Box::new(ClockSrc));
    let june = TimeInterval::new(1_717_200_000_000, 1_719_792_000_000);
    let july = TimeInterval::new(1_719_792_000_000, 1_722_470_400_000);
    let request = request(window(0..WIDTH, 0..HEIGHT));

    let series = window_time_series(&engine, node, request, &[june, july])
        .await
        .unwrap();

    assert_eq!(series.len(), 2);
    for (step, statistics) in &series {
        assert_eq!(statistics.count, WIDTH * HEIGHT);
        assert_eq!(statistics.minimum, stamp(Some(*step)));
        assert_eq!(statistics.maximum, stamp(Some(*step)));
        assert_eq!(statistics.value(Statistic::Mean), stamp(Some(*step)));
    }
    assert_ne!(series[0].1.mean(), series[1].1.mean());

    let single = window_statistics(&engine, node, request.with_time(Some(june)))
        .await
        .unwrap();
    assert_eq!(single, series[0].1);
}
