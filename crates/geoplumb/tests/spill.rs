//! disk cache tier: eviction demotes to disk, hits reload without
//! recomputing, budgets and invalidation delete files, and the directory
//! outlives the engine for the next one over the same graph

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use geoplumb::caps::Constraint;
use geoplumb::element::{Fanin, Source, Transform};
use geoplumb::elements::{
    Aspect, BandMath, Focal, FocalOp, Hillshade, IdwGrid, Mosaic, QualityMask, RasterSrc,
    Reproject, Slope, ToRaster,
};
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Crs, Engine, Graph, NodeId, TimeInterval, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

// one 256 px chunk is 512 KiB, so this holds a single chunk in memory
const ONE_CHUNK: usize = 600_000;

fn dem_src(base_height: f64) -> RasterSrc {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(base_height + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos());
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
    time_varying: bool,
}

impl Source for CountingSrc {
    fn identity(&self) -> String {
        format!(
            "CountingSrc time_varying={} {}",
            self.time_varying,
            self.inner.identity()
        )
    }

    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.inner.grid()
    }

    fn time_varying(&self) -> bool {
        self.time_varying
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<geoplumb::Chunk>> {
        Box::pin(async move {
            self.reads.fetch_add(1, Ordering::SeqCst);
            self.inner.read(req).await
        })
    }
}

fn counting_graph() -> (Graph, NodeId, Arc<AtomicUsize>) {
    graph_over(dem_src(500.0), false)
}

fn graph_over(inner: RasterSrc, time_varying: bool) -> (Graph, NodeId, Arc<AtomicUsize>) {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let src = g.add_source(Box::new(CountingSrc {
        inner,
        reads: reads.clone(),
        time_varying,
    }));
    (g, src, reads)
}

fn store_dirs(base: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(base)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| {
            path.file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .starts_with("geoplumb-")
        })
        .collect();
    dirs.sort();
    dirs
}

fn only_store_dir(base: &Path) -> PathBuf {
    let dirs = store_dirs(base);
    assert_eq!(dirs.len(), 1, "expected one store dir: {dirs:?}");
    dirs[0].clone()
}

fn chunk_files(store: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(store)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.extension().is_some_and(|e| e == "bin"))
        .collect()
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
async fn dropping_the_engine_keeps_its_spill_dir() {
    let base = spill_base("keep");
    let (g, src, _reads) = counting_graph();
    let engine = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    engine.pull(src, window(0, 0, 224, 224)).await.unwrap();
    engine.pull(src, window(256, 0, 480, 224)).await.unwrap();
    drop(engine);
    assert_eq!(
        chunk_files(&only_store_dir(&base)).len(),
        2,
        "the spill files did not survive the engine"
    );
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn a_second_engine_over_the_same_graph_reads_the_first_ones_files() {
    let base = spill_base("reuse");
    let tile = window(0, 0, 224, 224);
    let (g, src, _reads) = counting_graph();
    let first = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    let written = first.pull(src, tile).await.unwrap().into_raster().unwrap();
    drop(first);

    let (g, src, reads) = counting_graph();
    let second = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    let reloaded = second.pull(src, tile).await.unwrap().into_raster().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        0,
        "the second engine read the source instead of the first engine's file"
    );
    assert_same(&written, &reloaded);
    only_store_dir(&base);
    drop(second);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn a_changed_parameter_gets_its_own_directory() {
    let base = spill_base("parameter");
    let tile = window(0, 0, 224, 224);
    let shaded = |azimuth| {
        let (mut g, src, reads) = counting_graph();
        let shade = g.add_transform(src, Box::new(Hillshade::new(azimuth, 45.0)));
        (g, shade, reads)
    };
    let (g, shade, _reads) = shaded(315.0);
    let first = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    first.pull(shade, tile).await.unwrap();
    drop(first);

    let (g, shade, reads) = shaded(0.0);
    let second = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    second.pull(shade, tile).await.unwrap();
    assert!(
        reads.load(Ordering::SeqCst) > 0,
        "a different azimuth was served the first engine's files"
    );
    assert_eq!(store_dirs(&base).len(), 2);
    drop(second);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn changed_source_data_gets_its_own_directory() {
    let base = spill_base("content");
    let tile = window(0, 0, 224, 224);
    let (g, src, _reads) = graph_over(dem_src(500.0), false);
    let first = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    first.pull(src, tile).await.unwrap();
    drop(first);

    let (g, src, reads) = graph_over(dem_src(501.0), false);
    let second = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    second.pull(src, tile).await.unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        1,
        "different source data was served the first engine's file"
    );
    assert_eq!(store_dirs(&base).len(), 2);
    drop(second);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn time_keyed_files_are_dropped_at_open() {
    let base = spill_base("time");
    let tile = window(0, 0, 224, 224).with_time(Some(TimeInterval::new(0, 86_400_000)));
    let (g, src, _reads) = graph_over(dem_src(500.0), true);
    let first = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    first.pull(src, tile).await.unwrap();
    drop(first);
    let store = only_store_dir(&base);
    let timed = chunk_files(&store);
    assert_eq!(timed.len(), 1);
    let parts = timed[0]
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .split('_')
        .count();
    assert_eq!(parts, 6, "the spilled chunk is not keyed by time");

    let (g, src, reads) = graph_over(dem_src(500.0), true);
    let second = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    assert!(
        chunk_files(&store).is_empty(),
        "a time-keyed file survived the open"
    );
    second.pull(src, tile).await.unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        1,
        "a time-keyed chunk came from the first engine's file"
    );
    drop(second);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn a_truncated_file_is_recomputed_and_a_leftover_temp_file_deleted() {
    let base = spill_base("truncated");
    let tile = window(0, 0, 224, 224);
    let (g, src, _reads) = counting_graph();
    let first = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    let written = first.pull(src, tile).await.unwrap().into_raster().unwrap();
    drop(first);
    let store = only_store_dir(&base);
    let files = chunk_files(&store);
    assert_eq!(files.len(), 1);
    let file = &files[0];
    let whole = std::fs::read(file).unwrap();
    std::fs::write(file, &whole[..whole.len() / 2]).unwrap();
    let leftover = store.join("0_0_0_0.1-0.tmp");
    std::fs::write(&leftover, b"partial").unwrap();

    let (g, src, reads) = counting_graph();
    let second = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    assert!(!leftover.exists(), "the open kept a partial write");
    let recomputed = second.pull(src, tile).await.unwrap().into_raster().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        1,
        "the truncated file was not recomputed"
    );
    assert_same(&written, &recomputed);
    assert_eq!(
        std::fs::metadata(file).unwrap().len(),
        whole.len() as u64,
        "the recompute did not overwrite the truncated file"
    );
    drop(second);
    let _ = std::fs::remove_dir_all(&base);
}

#[tokio::test]
async fn reopening_over_the_disk_budget_keeps_the_newest_files() {
    let base = spill_base("reopen-budget");
    let (older, newer) = (window(0, 0, 224, 224), window(256, 0, 480, 224));
    let (g, src, _reads) = counting_graph();
    let first = Engine::with_disk_cache(g, ONE_CHUNK, &base, 64 << 20).unwrap();
    first.pull(src, older).await.unwrap();
    first.pull(src, newer).await.unwrap();
    drop(first);
    // both writes can land in one mtime tick, so age the older file outright
    let store = only_store_dir(&base);
    let hour_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(3600);
    std::fs::File::options()
        .write(true)
        .open(store.join("0_0_0_0.bin"))
        .unwrap()
        .set_modified(hour_ago)
        .unwrap();

    let (g, src, reads) = counting_graph();
    let second = Engine::with_disk_cache(g, ONE_CHUNK, &base, ONE_CHUNK).unwrap();
    assert_eq!(chunk_files(&store).len(), 1, "the open kept both files");
    second.pull(src, newer).await.unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        0,
        "the newer file was dropped"
    );
    second.pull(src, older).await.unwrap();
    assert_eq!(reads.load(Ordering::SeqCst), 1, "the older file was kept");
    drop(second);
    let _ = std::fs::remove_dir_all(&base);
}

#[test]
fn element_identities_tell_parameters_apart() {
    let singles = [
        Slope.identity(),
        Aspect.identity(),
        ToRaster.identity(),
        Mosaic.identity(),
    ];
    for identity in singles {
        assert!(!identity.is_empty());
    }
    let pairs = [
        (
            Hillshade::new(315.0, 45.0).identity(),
            Hillshade::new(0.0, 45.0).identity(),
        ),
        (
            Focal::new(FocalOp::Mean, 1).identity(),
            Focal::new(FocalOp::Mean, 2).identity(),
        ),
        (
            BandMath::new("b0 + 1").unwrap().identity(),
            BandMath::new("b0 + 2").unwrap().identity(),
        ),
        (
            Reproject::new(Crs::WEB_MERCATOR).identity(),
            Reproject::new(Crs::WGS84).identity(),
        ),
        (
            IdwGrid::default().identity(),
            IdwGrid {
                power: 3.0,
                ..IdwGrid::default()
            }
            .identity(),
        ),
        (
            QualityMask::new(0, vec![4.0]).identity(),
            QualityMask::new(0, vec![5.0]).identity(),
        ),
        (dem_src(500.0).identity(), dem_src(501.0).identity()),
    ];
    for (a, b) in pairs {
        assert!(!a.is_empty());
        assert_ne!(a, b);
    }
}
