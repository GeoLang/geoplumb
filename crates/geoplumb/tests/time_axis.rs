//! per-pull time: rfc 3339 round trips, the interval riding a request
//! upstream, cache keys splitting by time only where something resolves
//! per time, and timed chunks spilling apart

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use geoplumb::caps::{
    Caps, CapsPattern, CapsSet, Constraint, Dtype, RasterCaps, RasterPattern, ResRange, SetField,
};
use geoplumb::element::{Source, Transform};
use geoplumb::elements::stac::{format_rfc3339, parse_rfc3339};
use geoplumb::elements::{Hillshade, Reproject};
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, NodeId, RasterChunk, TimeInterval, WindowReq};
use terrano_core::{BandedRaster, Raster};

const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

// one 256 px f64 chunk is 512 KiB, so this budget holds a single chunk
const ONE_CHUNK: usize = 600_000;

fn interval(from: &str, to: &str) -> TimeInterval {
    TimeInterval::new(parse_rfc3339(from).unwrap(), parse_rfc3339(to).unwrap())
}

fn june() -> TimeInterval {
    interval("2024-06-01T00:00:00Z", "2024-07-01T00:00:00Z")
}

fn july() -> TimeInterval {
    interval("2024-07-01T00:00:00Z", "2024-08-01T00:00:00Z")
}

/// the value a window pulled at `time` should carry: the source stamps
/// every pixel with the interval it was read for
fn stamp(time: Option<TimeInterval>) -> f64 {
    time.map_or(-1.0, |t| t.start_ms as f64)
}

/// a source that stamps every pixel with the pull time it was read for,
/// so a chunk names the instant it was computed for. `varying: false` is
/// the same source ignoring time, the way a plain cog source does
struct ClockSrc {
    varying: bool,
    reads: Arc<AtomicUsize>,
    /// the time of every read, so a test can see what reached the source
    seen: Arc<Mutex<Vec<Option<TimeInterval>>>>,
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
            chunk_px: 256,
        }
    }

    fn time_varying(&self) -> bool {
        self.varying
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        Box::pin(async move {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.seen.lock().unwrap().push(req.time);
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

/// window-local +1, counting its computes so a cache hit is visible
struct AddOne {
    computes: Arc<AtomicUsize>,
}

impl Transform for AddOne {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        self.computes.fetch_add(1, Ordering::SeqCst);
        let mut chunk = input.raster()?.crop_to(&out.bbox);
        for v in chunk.bands.band_mut(0).unwrap().data_mut() {
            *v += 1.0;
        }
        Ok(Chunk::Raster(chunk))
    }
}

struct Rig {
    engine: Engine,
    node: NodeId,
    reads: Arc<AtomicUsize>,
    computes: Arc<AtomicUsize>,
    seen: Arc<Mutex<Vec<Option<TimeInterval>>>>,
}

fn rig(varying: bool, budget: usize) -> Rig {
    let (reads, computes) = (Arc::new(AtomicUsize::new(0)), Arc::new(AtomicUsize::new(0)));
    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut g = Graph::new();
    let s = g.add_source(Box::new(ClockSrc {
        varying,
        reads: reads.clone(),
        seen: seen.clone(),
    }));
    let node = g.add_transform(
        s,
        Box::new(AddOne {
            computes: computes.clone(),
        }),
    );
    Rig {
        engine: Engine::new(g, budget).unwrap(),
        node,
        reads,
        computes,
        seen,
    }
}

fn window(px0: usize, py0: usize, px1: usize, py1: usize) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + px0 as f64 * CELL,
        max_x: ORIGIN_X + px1 as f64 * CELL,
        max_y: ORIGIN_Y - py0 as f64 * CELL,
        min_y: ORIGIN_Y - py1 as f64 * CELL,
    }
}

fn req(bbox: Bbox, time: Option<TimeInterval>) -> WindowReq {
    WindowReq {
        bbox,
        resolution: CELL,
        time,
    }
}

async fn pull_value(rig: &Rig, bbox: Bbox, time: Option<TimeInterval>) -> f64 {
    let chunk = rig
        .engine
        .pull(rig.node, req(bbox, time))
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let band = chunk.bands.band(0).unwrap();
    let first = band.data()[0];
    assert!(
        band.data().iter().all(|v| *v == first),
        "the stub source stamps one value per window"
    );
    first
}

#[test]
fn rfc3339_round_trips_instants_and_intervals() {
    for s in [
        "1970-01-01T00:00:00Z",
        "2024-06-01T12:30:45Z",
        "2026-08-06T23:59:59.250Z",
        "2000-02-29T00:00:00Z",
    ] {
        assert_eq!(format_rfc3339(parse_rfc3339(s).unwrap()), s);
    }
    // the epoch second of 2024-06-01, independent of this code's own math
    assert_eq!(
        parse_rfc3339("2024-06-01T00:00:00Z").unwrap(),
        1_717_200_000_000
    );
    assert_eq!(
        parse_rfc3339("2024-06-01T12:00:00+02:00").unwrap(),
        parse_rfc3339("2024-06-01T10:00:00Z").unwrap()
    );
    assert_eq!(parse_rfc3339("1969-12-31T23:59:59Z").unwrap(), -1000);

    let iv = TimeInterval::parse("2024-06-01T00:00:00Z/2024-07-01T00:00:00Z").unwrap();
    assert_eq!(iv, june());
    assert_eq!(iv.end_ms - iv.start_ms, 30 * 86_400_000);
    assert_eq!(
        iv.to_stac_datetime(),
        "2024-06-01T00:00:00Z/2024-07-01T00:00:00Z"
    );

    assert!(parse_rfc3339("last tuesday").is_err());
    assert!(parse_rfc3339("2024-13-01T00:00:00Z").is_err());
    assert!(TimeInterval::parse("2024-06-01T00:00:00Z").is_err());
    assert!(
        TimeInterval::parse("2024-07-01T00:00:00Z/2024-06-01T00:00:00Z").is_err(),
        "an interval that ends before it starts is not one"
    );
}

/// two instants off one graph: different data, and the second pull of the
/// first instant is a cache hit, so the two live side by side
#[tokio::test]
async fn two_pull_times_are_two_cache_entries() {
    let rig = rig(true, 64 << 20);
    let w = window(0, 0, 224, 224);

    let first = pull_value(&rig, w, Some(june())).await;
    let second = pull_value(&rig, w, Some(july())).await;
    assert_eq!(first, stamp(Some(june())) + 1.0);
    assert_eq!(second, stamp(Some(july())) + 1.0);
    assert_ne!(first, second);
    assert_eq!(rig.reads.load(Ordering::SeqCst), 2);
    assert_eq!(rig.computes.load(Ordering::SeqCst), 2);

    // june is still cached, july did not evict or overwrite it
    assert_eq!(pull_value(&rig, w, Some(june())).await, first);
    assert_eq!(rig.reads.load(Ordering::SeqCst), 2, "june recomputed");
    assert_eq!(rig.computes.load(Ordering::SeqCst), 2, "june recomputed");

    // and the source was asked for exactly the pull's intervals
    assert_eq!(*rig.seen.lock().unwrap(), vec![Some(june()), Some(july())]);
}

/// nothing in this graph resolves per time, so two instants share one
/// entry instead of doubling the cache
#[tokio::test]
async fn a_time_invariant_graph_computes_once_for_two_times() {
    let rig = rig(false, 64 << 20);
    let w = window(0, 0, 224, 224);

    let first = pull_value(&rig, w, Some(june())).await;
    let second = pull_value(&rig, w, Some(july())).await;
    assert_eq!(first, second);
    assert_eq!(
        rig.reads.load(Ordering::SeqCst),
        1,
        "second time recomputed"
    );
    assert_eq!(rig.computes.load(Ordering::SeqCst), 1);
    // a source that ignores time never sees one
    assert_eq!(*rig.seen.lock().unwrap(), vec![None]);
}

fn raster_caps(crs: Crs) -> Caps {
    Caps::Raster(RasterCaps {
        dtype: Dtype::F64,
        bands: 1,
        crs,
        resolution: ResRange::at_least(CELL),
        chunk_px: 256,
    })
}

/// a halo plan and a reproject plan rewrite the window, never the time
#[test]
fn halo_and_reproject_plans_carry_the_pull_time() {
    let out = req(window(0, 0, 224, 224), Some(june()));
    let hillshade = Hillshade::new(315.0, 45.0);
    let planned = hillshade.plan(&out);
    assert_eq!(planned.time, out.time);
    assert!(planned.bbox.width() > out.bbox.width(), "no halo");

    let mut reproject = Reproject::new(Crs::WEB_MERCATOR);
    reproject
        .configure(&raster_caps(Crs::WGS84), &raster_caps(Crs::WEB_MERCATOR))
        .unwrap();
    assert_eq!(reproject.plan(&out).time, out.time);
}

/// the same through a live graph: a pull at an instant reaches the source
/// as that instant, across a halo transform and a crs change
#[tokio::test]
async fn a_pull_time_reaches_the_source_through_a_reprojected_chain() {
    let (reads, seen) = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut g = Graph::new();
    let s = g.add_source(Box::new(ClockSrc {
        varying: true,
        reads: reads.clone(),
        seen: seen.clone(),
    }));
    let hs = g.add_transform(s, Box::new(Hillshade::new(315.0, 45.0)));
    let rp = g.add_transform(hs, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let tile = geoplumb::tile::XyzTile {
        z: 12,
        x: 2138,
        y: 1441,
    };
    geoplumb::tile::render_tile_at(&engine, rp, tile, Some(june()))
        .await
        .unwrap();
    let seen = seen.lock().unwrap();
    assert!(!seen.is_empty(), "the chain never reached the source");
    assert!(
        seen.iter().all(|t| *t == Some(june())),
        "the pull time was dropped upstream: {seen:?}"
    );
}

/// a timed chunk goes to disk under its own name: the reload brings back
/// the instant it was written for, not the one that evicted it
#[tokio::test]
async fn timed_chunks_spill_and_reload_apart() {
    let base = std::env::temp_dir().join(format!("geoplumb-test-{}-time", std::process::id()));
    std::fs::create_dir_all(&base).unwrap();
    let (reads, seen) = (
        Arc::new(AtomicUsize::new(0)),
        Arc::new(Mutex::new(Vec::new())),
    );
    let mut g = Graph::new();
    let node = g.add_source(Box::new(ClockSrc {
        varying: true,
        reads: reads.clone(),
        seen,
    }));
    let engine = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    let w = window(0, 0, 224, 224);

    let value = |chunk: RasterChunk| chunk.bands.band(0).unwrap().data()[0];
    let pull = async |time| {
        value(
            engine
                .pull(node, req(w, Some(time)))
                .await
                .unwrap()
                .into_raster()
                .unwrap(),
        )
    };

    let first = pull(june()).await;
    // july's chunk pushes june's off the memory budget and onto disk
    let second = pull(july()).await;
    assert_ne!(first, second);
    let settled = reads.load(Ordering::SeqCst);
    assert_eq!(pull(june()).await, first, "the spilled june chunk changed");
    assert_eq!(
        reads.load(Ordering::SeqCst),
        settled,
        "june came back from a recompute, not from disk"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&base);
}
