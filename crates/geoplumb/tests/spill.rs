//! disk cache tier: eviction demotes to disk, hits reload without
//! recomputing, budgets and invalidation delete files, drop cleans up

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use geoplumb::caps::Constraint;
use geoplumb::element::Source;
use geoplumb::elements::RasterSrc;
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Crs, Engine, Graph, NodeId, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

// one 256 px chunk is 512 KiB, so this holds a single chunk in memory
const ONE_CHUNK: usize = 600_000;

fn dem_src() -> RasterSrc {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos());
        }
    }
    RasterSrc::new(
        BandedRaster::new(vec![Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

struct CountingSrc {
    inner: RasterSrc,
    reads: Arc<AtomicUsize>,
}

impl Source for CountingSrc {
    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.inner.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<geoplumb::Chunk>> {
        Box::pin(async move {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.inner.read(req).await
        })
    }
}

fn counting_graph() -> (Graph, NodeId, Arc<AtomicUsize>) {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let src = g.add_source(Box::new(CountingSrc {
        inner: dem_src(),
        reads: reads.clone(),
    }));
    (g, src, reads)
}

fn spill_base(test: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("geoplumb-test-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn window(px0: usize, py0: usize, px1: usize, py1: usize) -> WindowReq {
    WindowReq {
        bbox: Bbox {
            min_x: ORIGIN_X + px0 as f64 * CELL,
            max_x: ORIGIN_X + px1 as f64 * CELL,
            max_y: ORIGIN_Y - py0 as f64 * CELL,
            min_y: ORIGIN_Y - py1 as f64 * CELL,
        },
        resolution: CELL,
        time: None,
    }
}

fn assert_same(a: &geoplumb::RasterChunk, b: &geoplumb::RasterChunk) {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());
    for (x, y) in a
        .bands
        .band(0)
        .unwrap()
        .data()
        .iter()
        .zip(b.bands.band(0).unwrap().data())
    {
        assert!(
            (x.is_nan() && y.is_nan()) || x == y,
            "spill roundtrip changed a value: {x} vs {y}"
        );
    }
}

#[tokio::test]
async fn evicted_chunk_is_served_from_disk() {
    let base = spill_base("serve");
    let (g, src, reads) = counting_graph();
    let engine = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();

    // this chunk spans the raster edge, so the roundtrip also covers nan
    let a = window(512, 0, 600, 224);
    let first = engine.pull(src, a).await.unwrap().into_raster().unwrap();
    let after_first = reads.load(Ordering::SeqCst);

    // a different chunk evicts the first from memory to disk
    engine.pull(src, window(0, 0, 224, 224)).await.unwrap();
    let again = engine.pull(src, a).await.unwrap().into_raster().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        after_first + 1,
        "disk hit recomputed the chunk"
    );
    assert_same(&first, &again);
    drop(engine);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn disk_budget_drops_the_oldest_file() {
    let base = spill_base("budget");
    let (g, src, reads) = counting_graph();
    // memory and disk each hold one chunk
    let engine = Engine::with_disk_cache(g, ONE_CHUNK, &base, ONE_CHUNK).unwrap();

    let (a, b, c) = (
        window(0, 0, 224, 224),
        window(256, 0, 480, 224),
        window(0, 256, 224, 400),
    );
    engine.pull(src, a).await.unwrap().into_raster().unwrap();
    engine.pull(src, b).await.unwrap().into_raster().unwrap();
    engine.pull(src, c).await.unwrap().into_raster().unwrap();
    // memory holds c, disk holds b, a's file fell over the disk budget
    let settled = reads.load(Ordering::SeqCst);
    engine.pull(src, b).await.unwrap().into_raster().unwrap();
    assert_eq!(reads.load(Ordering::SeqCst), settled, "b was on disk");
    engine.pull(src, a).await.unwrap().into_raster().unwrap();
    assert!(
        reads.load(Ordering::SeqCst) > settled,
        "a should have been dropped from the disk tier"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn invalidation_reaches_spilled_entries() {
    let base = spill_base("invalidate");
    let (g, src, reads) = counting_graph();
    let engine = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();

    let a = window(0, 0, 224, 224);
    engine.pull(src, a).await.unwrap().into_raster().unwrap();
    engine.pull(src, window(256, 0, 480, 224)).await.unwrap();
    // a is on disk now, and dirty
    let settled = reads.load(Ordering::SeqCst);
    engine.invalidate(src, window(10, 10, 50, 50).bbox);
    engine.pull(src, a).await.unwrap().into_raster().unwrap();
    assert!(
        reads.load(Ordering::SeqCst) > settled,
        "invalidated spilled chunk served stale data"
    );
    drop(engine);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn dropping_the_engine_removes_its_spill_dir() {
    let base = spill_base("cleanup");
    let (g, src, _reads) = counting_graph();
    let engine = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    engine.pull(src, window(0, 0, 224, 224)).await.unwrap();
    engine.pull(src, window(256, 0, 480, 224)).await.unwrap();
    assert!(
        std::fs::read_dir(&base).unwrap().next().is_some(),
        "no spill dir was created"
    );
    drop(engine);
    assert!(
        std::fs::read_dir(&base).unwrap().next().is_none(),
        "spill dir survived the engine"
    );
    let _ = std::fs::remove_dir_all(&base);
}
