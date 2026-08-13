//! peak heap of a reducing composite against stack depth, measured by a
//! counting global allocator. it owns the whole test binary, so this file
//! holds exactly one test: anything running beside it would land in the
//! same peak

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::routing::get;
use geoplumb::element::Source;
use geoplumb::elements::stac::{MAX_PARALLEL_ITEM_READS, STACK_VALUE_BUDGET};
use geoplumb::elements::{Composite, StacSearch, StacSrc};
use geoplumb::{Bbox, WindowReq};
use terrano_core::{CogParams, Raster, write_cog};

struct PeakAlloc {
    live: AtomicUsize,
    peak: AtomicUsize,
    base: AtomicUsize,
}

impl PeakAlloc {
    /// peak restarts from what is live now, so a later reading is growth
    /// over the setup rather than over the whole process
    fn mark(&self) {
        let live = self.live.load(Ordering::Relaxed);
        self.base.store(live, Ordering::Relaxed);
        self.peak.store(live, Ordering::Relaxed);
    }

    fn growth(&self) -> usize {
        self.peak
            .load(Ordering::Relaxed)
            .saturating_sub(self.base.load(Ordering::Relaxed))
    }
}

unsafe impl GlobalAlloc for PeakAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            let live = self.live.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            self.peak.fetch_max(live, Ordering::Relaxed);
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        self.live.fetch_sub(layout.size(), Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}

#[global_allocator]
static ALLOC: PeakAlloc = PeakAlloc {
    live: AtomicUsize::new(0),
    peak: AtomicUsize::new(0),
    base: AtomicUsize::new(0),
};

const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

/// pixels down each side of an item and of the pull window, which cover
/// each other exactly, so every item contributes a full window
const SIDE: usize = 256;

/// one decoded item window, the unit both bounds below are written in
const WINDOW_BYTES: usize = SIDE * SIDE * size_of::<f64>();

/// the two stack depths compared: one wave of the source's parallel read
/// cap, and four of them
const SHALLOW: usize = MAX_PARALLEL_ITEM_READS;
const DEEP: usize = 4 * MAX_PARALLEL_ITEM_READS;

/// deepest stack this window still reduces in one strip, the depth the
/// control below measures under
const FITTING: usize = STACK_VALUE_BUDGET / (SIDE * SIDE);

const WINDOW: WindowReq = WindowReq {
    bbox: Bbox {
        min_x: ORIGIN_X,
        min_y: ORIGIN_Y - SIDE as f64 * CELL,
        max_x: ORIGIN_X + SIDE as f64 * CELL,
        max_y: ORIGIN_Y,
    },
    resolution: CELL,
    time: None,
};

/// a flat item: its value marks which item it is, and the flatness keeps
/// the mock's stored cogs far smaller than the windows they decode to
fn cog(value: f64) -> Vec<u8> {
    let raster = Raster::from_vec(SIDE, SIDE, vec![value; SIDE * SIDE], CELL, f64::NAN).unwrap();
    let params = CogParams {
        tile_width: 128,
        tile_height: 128,
        overview_levels: 1,
        epsg: 4326,
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        deflate: true,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    write_cog(&raster, &params, &mut buf).unwrap();
    buf.into_inner()
}

struct Mock {
    cogs: std::collections::HashMap<String, Vec<u8>>,
    features: Vec<serde_json::Value>,
}

async fn serve_search(
    State(mock): State<Arc<Mock>>,
) -> ([(&'static str, &'static str); 1], String) {
    let body = serde_json::json!({ "type": "FeatureCollection", "features": mock.features });
    ([("content-type", "application/geo+json")], body.to_string())
}

async fn serve_cog(
    State(mock): State<Arc<Mock>>,
    Path(name): Path<String>,
    headers: HeaderMap,
) -> (StatusCode, Vec<u8>) {
    let Some(bytes) = mock.cogs.get(&name) else {
        return (StatusCode::NOT_FOUND, Vec::new());
    };
    let Some(range) = headers.get(header::RANGE) else {
        return (StatusCode::OK, bytes.clone());
    };
    let spec = range.to_str().unwrap().trim_start_matches("bytes=");
    let (s, e) = spec.split_once('-').unwrap();
    let (s, e): (usize, usize) = (s.parse().unwrap(), e.parse().unwrap());
    (
        StatusCode::PARTIAL_CONTENT,
        bytes[s..=e.min(bytes.len() - 1)].to_vec(),
    )
}

/// `items` co-located items one day apart, item i flat at value i
async fn open_stack(items: usize, composite: Composite) -> StacSrc {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    let mut cogs = std::collections::HashMap::new();
    let mut features = Vec::new();
    for i in 0..items {
        let name = format!("deep_{i}.tif");
        cogs.insert(name.clone(), cog(i as f64));
        features.push(serde_json::json!({
            "id": name,
            "bbox": [WINDOW.bbox.min_x, WINDOW.bbox.min_y, WINDOW.bbox.max_x, WINDOW.bbox.max_y],
            "properties": {
                "datetime": format!("2024-{:02}-{:02}T00:00:00Z", i / 28 + 1, i % 28 + 1),
                "proj:epsg": 4326,
            },
            "assets": { "data": { "href": format!("{base}/cog/{name}") } }
        }));
    }
    let app = axum::Router::new()
        .route("/search", get(serve_search))
        .route("/cog/{name}", get(serve_cog))
        .with_state(Arc::new(Mock { cogs, features }));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let mut search = StacSearch::new(
        &base,
        "deep",
        "data",
        [
            WINDOW.bbox.min_x,
            WINDOW.bbox.min_y,
            WINDOW.bbox.max_x,
            WINDOW.bbox.max_y,
        ],
    );
    search.composite = composite;
    // one page, so a search cannot spread the stack over `next` links
    search.limit = (DEEP * 2) as u32;
    tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap()
}

/// peak heap growth over one pull, and the value it put in every pixel
async fn pull(items: usize, composite: Composite) -> (usize, f64) {
    let src = open_stack(items, composite).await;
    assert_eq!(src.item_count(), items);
    ALLOC.mark();
    let chunk = src.read(&WINDOW).await.unwrap().into_raster().unwrap();
    let growth = ALLOC.growth();
    let band = chunk.bands.band(0).unwrap();
    assert_eq!(band.width() * band.height(), SIDE * SIDE);
    let got = band.data()[0];
    assert!(
        band.data().iter().all(|v| *v == got),
        "flat items must reduce flat"
    );
    (growth, got)
}

/// the memory half of the dense-collection limit: neither peak may follow
/// the item count, the folding reducer because it drops each wave and the
/// stack reducer because it reads the window in strips. the control is the
/// stack reducer under the budget, where a deeper stack really is another
/// resident window per item: without it both bounds would pass on a
/// harness too noisy to measure anything
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_composite_peaks_flat_in_the_stack_depth() {
    const {
        assert!(
            FITTING < DEEP,
            "the deep stack no longer passes what a reduction holds at once"
        )
    };
    let extra_items = DEEP - SHALLOW;

    let (shallow_fold, shallow_mean) = pull(SHALLOW, Composite::Mean).await;
    let (deep_fold, deep_mean) = pull(DEEP, Composite::Mean).await;
    assert_eq!(shallow_mean, (SHALLOW - 1) as f64 / 2.0);
    assert_eq!(deep_mean, (DEEP - 1) as f64 / 2.0);

    let (shallow_stack, shallow_median) = pull(SHALLOW, Composite::Median).await;
    let (deep_stack, deep_median) = pull(DEEP, Composite::Median).await;
    // item i is flat at i, so an even stack's median is (n - 1) / 2 and a
    // strip that lost or repeated an item would miss it
    assert_eq!(shallow_median, (SHALLOW - 1) as f64 / 2.0);
    assert_eq!(deep_median, (DEEP - 1) as f64 / 2.0);

    let (small_stack, _) = pull(FITTING / 4, Composite::Median).await;
    let (fitting_stack, _) = pull(FITTING, Composite::Median).await;

    let fold_step = deep_fold.saturating_sub(shallow_fold);
    let stack_step = deep_stack.saturating_sub(shallow_stack);
    let control_step = fitting_stack.saturating_sub(small_stack);
    assert!(
        fold_step < extra_items / 4 * WINDOW_BYTES,
        "folding peak grew {fold_step} bytes over {extra_items} more items"
    );
    assert!(
        stack_step < extra_items / 4 * WINDOW_BYTES,
        "stack peak grew {stack_step} bytes over {extra_items} more items"
    );
    let control_items = FITTING - FITTING / 4;
    assert!(
        control_step > control_items / 2 * WINDOW_BYTES,
        "under the budget the stack peak grew only {control_step} bytes over \
         {control_items} more items, the measurement is not sensitive enough \
         to mean anything"
    );
}
