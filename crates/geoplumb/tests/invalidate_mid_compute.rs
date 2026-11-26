use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};

use futures::future::BoxFuture;
use geoplumb::caps::{CapsSet, Constraint, ValueKind};
use geoplumb::element::{Source, Transform};
use geoplumb::elements::RasterSrc;
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, NodeId, WindowReq};
use terrano_core::{BandedRaster, Raster};
use tokio::sync::Notify;

const SIDE: usize = 16;
const CELL: f64 = 1.0;
const ORIGINAL: f64 = 1.0;
const EDITED: f64 = 2.0;
const BUDGET: usize = 64 << 20;

fn extent() -> Bbox {
    Bbox::new(0.0, 0.0, SIDE as f64 * CELL, SIDE as f64 * CELL)
}

fn constant_src(value: f64) -> RasterSrc {
    let raster = Raster::from_vec(SIDE, SIDE, vec![value; SIDE * SIDE], CELL, f64::NAN).unwrap();
    RasterSrc::new(
        BandedRaster::new(vec![raster]).unwrap(),
        0.0,
        SIDE as f64 * CELL,
        Crs::WGS84,
        ValueKind::Continuous,
    )
}

struct EditableSrc {
    original: RasterSrc,
    edited: RasterSrc,
    is_edited: Arc<AtomicBool>,
}

impl Source for EditableSrc {
    fn identity(&self) -> String {
        "EditableSrc".to_string()
    }

    fn constraint(&self) -> Constraint {
        self.original.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.original.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        if self.is_edited.load(Ordering::SeqCst) {
            self.edited.read(req)
        } else {
            self.original.read(req)
        }
    }
}

struct GatedCopy {
    region: Option<Bbox>,
    started: Arc<Notify>,
    gate: Mutex<mpsc::Receiver<()>>,
}

impl Transform for GatedCopy {
    fn identity(&self) -> String {
        format!("GatedCopy {:?}", self.region)
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn region(&self) -> Option<Bbox> {
        self.region
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        self.started.notify_one();
        // std mpsc recv returns once every sender is dropped
        let _ = self.gate.lock().unwrap().recv();
        Ok(Chunk::Raster(input.raster()?.crop_to(&out.bbox)))
    }
}

async fn pull(engine: &Engine, node: NodeId) -> Vec<f64> {
    engine
        .pull(
            node,
            WindowReq {
                bbox: extent(),
                resolution: CELL,
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
        .data()
        .to_vec()
}

async fn edit_source_while_a_compute_runs(region: Option<Bbox>) {
    let (release, gate) = mpsc::channel::<()>();
    let started = Arc::new(Notify::new());
    let is_edited = Arc::new(AtomicBool::new(false));
    let mut g = Graph::new();
    let source = g.add_source(Box::new(EditableSrc {
        original: constant_src(ORIGINAL),
        edited: constant_src(EDITED),
        is_edited: Arc::clone(&is_edited),
    }));
    let node = g.add_transform(
        source,
        Box::new(GatedCopy {
            region,
            started: Arc::clone(&started),
            gate: Mutex::new(gate),
        }),
    );
    let engine = Arc::new(Engine::new(g, BUDGET).unwrap());

    let first = tokio::spawn({
        let engine = Arc::clone(&engine);
        async move { pull(&engine, node).await }
    });
    started.notified().await;
    let mut waiter = Box::pin(pull(&engine, node));
    assert!(futures::poll!(&mut waiter).is_pending());

    is_edited.store(true, Ordering::SeqCst);
    engine.invalidate(source, extent());
    drop(release);
    first.await.unwrap();

    assert_eq!(
        waiter.await,
        vec![EDITED; SIDE * SIDE],
        "a pull waiting on the invalidated compute got its result"
    );
    assert_eq!(
        pull(&engine, node).await,
        vec![EDITED; SIDE * SIDE],
        "the compute that ran across the invalidation cached its result"
    );
}

#[tokio::test]
async fn a_tile_computed_across_an_invalidation_is_not_cached() {
    edit_source_while_a_compute_runs(None).await;
}

#[tokio::test]
async fn a_region_computed_across_an_invalidation_is_not_cached() {
    edit_source_while_a_compute_runs(Some(extent())).await;
}
