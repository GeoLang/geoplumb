use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use futures::future::BoxFuture;
use geoplumb::caps::{CapsSet, Constraint, ValueKind};
use geoplumb::element::{Source, Transform};
use geoplumb::elements::RasterSrc;
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Chunk, Crs, Engine, Error, Graph, NodeId, WindowReq};
use terrano_core::{BandedRaster, Raster};
use tokio::sync::Notify;

const CELL: f64 = 1.0;
const COLUMNS: usize = 128;
const ROWS: usize = 64;
const ORIGIN_Y: f64 = 64.0;
const REGION_COLUMNS: usize = 60;
const BUDGET: usize = 64 << 20;

fn cell_value(column: usize, row: usize) -> f64 {
    (row * 1000 + column) as f64
}

fn window(column0: usize, row0: usize, column1: usize, row1: usize) -> Bbox {
    Bbox::new(
        column0 as f64 * CELL,
        ORIGIN_Y - row1 as f64 * CELL,
        column1 as f64 * CELL,
        ORIGIN_Y - row0 as f64 * CELL,
    )
}

fn tile_origin(index: usize) -> (usize, usize) {
    (index % 4 * 16, index / 4 * 16)
}

fn tile(index: usize) -> Bbox {
    let (column0, row0) = tile_origin(index);
    window(column0, row0, column0 + 16, row0 + 16)
}

struct CountingSrc {
    inner: RasterSrc,
    reads: Arc<AtomicUsize>,
}

impl Source for CountingSrc {
    fn identity(&self) -> String {
        format!("CountingSrc {}", self.inner.identity())
    }

    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.inner.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read(req)
    }
}

struct CountingCopy {
    region: Bbox,
    computes: Arc<AtomicUsize>,
    started: Arc<Notify>,
    gate: Mutex<mpsc::Receiver<()>>,
}

impl Transform for CountingCopy {
    fn identity(&self) -> String {
        format!("CountingCopy {:?}", self.region)
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn region(&self) -> Option<Bbox> {
        Some(self.region)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        self.computes.fetch_add(1, Ordering::SeqCst);
        self.started.notify_one();
        // std mpsc recv returns once every sender is dropped
        let _ = self.gate.lock().unwrap().recv();
        Ok(Chunk::Raster(input.raster()?.crop_to(&out.bbox)))
    }
}

struct Fixture {
    engine: Arc<Engine>,
    source: NodeId,
    node: NodeId,
    reads: Arc<AtomicUsize>,
    computes: Arc<AtomicUsize>,
    started: Arc<Notify>,
}

fn fixture(gate: mpsc::Receiver<()>, budget: usize) -> Fixture {
    let mut data = Vec::with_capacity(COLUMNS * ROWS);
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            data.push(cell_value(column, row));
        }
    }
    let raster = Raster::from_vec(COLUMNS, ROWS, data, CELL, f64::NAN).unwrap();
    let reads = Arc::new(AtomicUsize::new(0));
    let computes = Arc::new(AtomicUsize::new(0));
    let started = Arc::new(Notify::new());
    let mut g = Graph::new();
    let source = g.add_source(Box::new(CountingSrc {
        inner: RasterSrc::new(
            BandedRaster::new(vec![raster]).unwrap(),
            0.0,
            ORIGIN_Y,
            Crs::WGS84,
            ValueKind::Continuous,
        ),
        reads: Arc::clone(&reads),
    }));
    let node = g.add_transform(
        source,
        Box::new(CountingCopy {
            region: window(0, 0, REGION_COLUMNS, ROWS),
            computes: Arc::clone(&computes),
            started: Arc::clone(&started),
            gate: Mutex::new(gate),
        }),
    );
    Fixture {
        engine: Arc::new(Engine::new(g, budget).unwrap()),
        source,
        node,
        reads,
        computes,
        started,
    }
}

fn open_gate() -> mpsc::Receiver<()> {
    let (release, gate) = mpsc::channel();
    drop(release);
    gate
}

async fn pull(
    engine: &Engine,
    node: NodeId,
    bbox: Bbox,
    resolution: f64,
) -> geoplumb::Result<Chunk> {
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
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn sixteen_concurrent_tile_pulls_in_a_region_compute_it_once() {
    let fixture = fixture(open_gate(), BUDGET);
    let pulls = (0..16).map(|index| {
        let engine = Arc::clone(&fixture.engine);
        let node = fixture.node;
        tokio::spawn(async move { pull(&engine, node, tile(index), CELL).await })
    });
    let tiles = futures::future::join_all(pulls).await;
    assert_eq!(fixture.computes.load(Ordering::SeqCst), 1);

    for (index, pulled) in tiles.into_iter().enumerate() {
        let raster = pulled.unwrap().unwrap().into_raster().unwrap();
        assert_eq!((raster.width(), raster.height()), (16, 16));
        let values = raster.bands.band(0).unwrap().data();
        let (column0, row0) = tile_origin(index);
        for row in 0..16 {
            for column in 0..16 {
                let value = values[row * 16 + column];
                let (column, row) = (column0 + column, row0 + row);
                if column < REGION_COLUMNS {
                    assert_eq!(value, cell_value(column, row), "column {column} row {row}");
                } else {
                    assert!(
                        value.is_nan(),
                        "column {column} row {row} past the region is {value}"
                    );
                }
            }
        }
    }
}

#[tokio::test]
async fn a_tile_outside_the_region_reads_no_source() {
    let fixture = fixture(open_gate(), BUDGET);
    let outside = pull(&fixture.engine, fixture.node, window(96, 16, 112, 32), CELL)
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    assert_eq!(fixture.reads.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.computes.load(Ordering::SeqCst), 0);
    let values = outside.bands.band(0).unwrap().data();
    assert_eq!(values.len(), 16 * 16);
    assert!(values.iter().all(|value| value.is_nan()), "{values:?}");

    pull(&fixture.engine, fixture.node, tile(0), CELL)
        .await
        .unwrap();
    assert!(
        fixture.reads.load(Ordering::SeqCst) > 0,
        "a tile in the region read no source either, so the count shows nothing"
    );
}

#[tokio::test]
async fn a_region_over_the_cap_fails_naming_the_finest_resolution_that_fits() {
    // 60 by 64 cells at 60 bytes each exceed 100000 bytes, the 30 by 32 cells of the next level do not
    let fixture = fixture(open_gate(), 100_000);
    let error = pull(&fixture.engine, fixture.node, tile(0), CELL)
        .await
        .unwrap_err();
    match error {
        Error::RegionTooLarge {
            pixels,
            cap,
            resolution,
            finest_resolution,
        } => {
            assert_eq!(pixels, 60 * 64);
            assert_eq!(cap, 100_000 / 60);
            assert_eq!(resolution, CELL);
            assert_eq!(finest_resolution, 2.0 * CELL);
        }
        other => panic!("expected RegionTooLarge, got {other}"),
    }
    assert_eq!(fixture.reads.load(Ordering::SeqCst), 0);
    assert_eq!(fixture.computes.load(Ordering::SeqCst), 0);

    pull(&fixture.engine, fixture.node, tile(0), 2.0 * CELL)
        .await
        .unwrap();
    assert_eq!(fixture.computes.load(Ordering::SeqCst), 1);
}

async fn pull_two_levels(fixture: &Fixture) {
    for resolution in [CELL, 2.0 * CELL] {
        pull(&fixture.engine, fixture.node, tile(5), resolution)
            .await
            .unwrap();
    }
}

#[tokio::test]
async fn an_invalidation_inside_the_region_drops_every_level() {
    let fixture = fixture(open_gate(), BUDGET);
    let mut events = fixture.engine.subscribe();
    pull_two_levels(&fixture).await;
    pull_two_levels(&fixture).await;
    assert_eq!(fixture.computes.load(Ordering::SeqCst), 2);

    fixture
        .engine
        .invalidate(fixture.source, window(100, 10, 101, 11));
    assert_eq!(events.try_recv().unwrap().node, fixture.source);
    assert!(
        events.try_recv().is_err(),
        "a cell outside the region dirtied the region node"
    );
    pull_two_levels(&fixture).await;
    assert_eq!(fixture.computes.load(Ordering::SeqCst), 2);

    fixture
        .engine
        .invalidate(fixture.source, window(10, 10, 11, 11));
    assert_eq!(events.try_recv().unwrap().node, fixture.source);
    let at_region = events.try_recv().expect("the region node was left clean");
    assert_eq!(at_region.node, fixture.node);
    assert_eq!(at_region.bbox, window(0, 0, REGION_COLUMNS, ROWS));
    pull_two_levels(&fixture).await;
    assert_eq!(fixture.computes.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn a_cancelled_first_pull_does_not_start_a_second_compute() {
    let (release, gate) = mpsc::channel::<()>();
    let fixture = fixture(gate, BUDGET);
    let first = tokio::spawn({
        let engine = Arc::clone(&fixture.engine);
        let node = fixture.node;
        async move { pull(&engine, node, tile(0), CELL).await }
    });
    fixture.started.notified().await;
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());

    let mut second = Box::pin(pull(&fixture.engine, fixture.node, tile(5), CELL));
    assert!(futures::poll!(&mut second).is_pending());
    drop(release);
    second.await.unwrap();
    assert_eq!(
        fixture.computes.load(Ordering::SeqCst),
        1,
        "the cancelled pull's region compute ran again"
    );
}
