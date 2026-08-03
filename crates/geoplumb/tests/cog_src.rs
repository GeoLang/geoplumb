//! cog-backed source: overview selection, decimation past the pyramid,
//! nan padding outside the file, and the http range transport

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::routing::get;
use geoplumb::element::Source;
use geoplumb::elements::{CogSrc, Hillshade, HttpRange, RasterSrc};
use geoplumb::{Bbox, Crs, Engine, Graph, WindowReq};
use terrano_core::{BandedRaster, CogParams, RangeRead, Raster, write_cog};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

fn dem() -> Raster {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(500.0 + 200.0 * ((lon * 8.0).sin() * (lat * 8.0).cos()));
        }
    }
    Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
}

fn cog_bytes(overview_levels: u32) -> Vec<u8> {
    let params = CogParams {
        tile_width: 64,
        tile_height: 64,
        overview_levels,
        epsg: 4326,
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        deflate: false,
    };
    let mut buf = std::io::Cursor::new(Vec::new());
    write_cog(&dem(), &params, &mut buf).unwrap();
    buf.into_inner()
}

struct MemRange {
    data: Vec<u8>,
    fetched: Arc<AtomicUsize>,
}

impl MemRange {
    fn new(data: Vec<u8>) -> Self {
        MemRange {
            data,
            fetched: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl RangeRead for MemRange {
    fn read_range(&mut self, offset: u64, len: u64) -> Result<Vec<u8>, terrano_core::Error> {
        self.fetched.fetch_add(len as usize, Ordering::SeqCst);
        let mut slice = self.data.as_slice();
        slice.read_range(offset, len)
    }
}

fn mem_src() -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![dem()]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
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

fn engine_of(src: impl Source + 'static) -> (Engine, geoplumb::NodeId) {
    let mut g = Graph::new();
    let n = g.add_source(Box::new(src));
    (Engine::new(g, 64 << 20).unwrap(), n)
}

fn assert_bands_close(a: &geoplumb::RasterChunk, b: &geoplumb::RasterChunk, tol: f64) {
    assert_eq!(a.width(), b.width());
    assert_eq!(a.height(), b.height());
    let (ba, bb) = (a.bands.band(0).unwrap(), b.bands.band(0).unwrap());
    for (i, (x, y)) in ba.data().iter().zip(bb.data()).enumerate() {
        if x.is_nan() && y.is_nan() {
            continue;
        }
        assert!((x - y).abs() < tol, "pixel {i}: cog {x} vs memory {y}");
    }
}

#[tokio::test]
async fn base_level_matches_in_memory_source() {
    let (cog, cn) = engine_of(CogSrc::open(MemRange::new(cog_bytes(2))).unwrap());
    let (mem, mn) = engine_of(mem_src());
    let req = WindowReq {
        bbox: window(20, 20, 340, 230),
        resolution: CELL,
    };
    let a = cog.pull(cn, req).await.unwrap().into_raster().unwrap();
    let b = mem.pull(mn, req).await.unwrap().into_raster().unwrap();
    assert_bands_close(&a, &b, 1e-12);
}

#[tokio::test]
async fn overview_pull_matches_and_fetches_less() {
    let src = CogSrc::open(MemRange::new(cog_bytes(2))).unwrap();
    let (mem, mn) = engine_of(mem_src());
    let full = window(0, 0, W, H);

    let coarse = WindowReq {
        bbox: full,
        resolution: CELL * 2.0,
    };
    let a = src.read(&coarse).await.unwrap().into_raster().unwrap();
    let b = mem.pull(mn, coarse).await.unwrap().into_raster().unwrap();
    assert_bands_close(&a, &b, 1e-12);

    // the coarse read must come from the overview: reopen fresh so header
    // reads don't blur the comparison, then compare tile bytes fetched
    let base_bytes = tile_bytes_fetched(WindowReq {
        bbox: full,
        resolution: CELL,
    })
    .await;
    let coarse_bytes = tile_bytes_fetched(coarse).await;
    assert!(
        coarse_bytes * 3 < base_bytes,
        "coarse pull fetched {coarse_bytes} of base {base_bytes}, overview not used"
    );
}

async fn tile_bytes_fetched(req: WindowReq) -> usize {
    let mem_range = MemRange::new(cog_bytes(2));
    let fetched = mem_range.fetched.clone();
    let src = CogSrc::open(mem_range).unwrap();
    let before = fetched.load(Ordering::SeqCst);
    src.read(&req).await.unwrap().into_raster().unwrap();
    fetched.load(Ordering::SeqCst) - before
}

#[tokio::test]
async fn pull_coarser_than_pyramid_decimates() {
    // one overview (down to 2*CELL), request 4*CELL: read level 1, average 2x2
    let src = CogSrc::open(MemRange::new(cog_bytes(1))).unwrap();
    let (mem, mn) = engine_of(mem_src());
    let req = WindowReq {
        bbox: window(0, 0, 400, 400),
        resolution: CELL * 4.0,
    };
    let a = src.read(&req).await.unwrap().into_raster().unwrap();
    let b = mem.pull(mn, req).await.unwrap().into_raster().unwrap();
    assert_eq!(a.width(), 100);
    // average-of-averages vs one flat average differ only in rounding
    assert_bands_close(&a, &b, 1e-9);
}

#[tokio::test]
async fn window_outside_the_file_pads_nan() {
    let src = CogSrc::open(MemRange::new(cog_bytes(0))).unwrap();
    // 40 px west and north of the origin, 40 px inside
    let req = WindowReq {
        bbox: Bbox {
            min_x: ORIGIN_X - 40.0 * CELL,
            max_x: ORIGIN_X + 40.0 * CELL,
            max_y: ORIGIN_Y + 40.0 * CELL,
            min_y: ORIGIN_Y - 40.0 * CELL,
        },
        resolution: CELL,
    };
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    let band = chunk.bands.band(0).unwrap();
    assert_eq!(band.width(), 80);
    assert!(band.data()[0].is_nan(), "outside pixel not nan");
    assert!(band.data()[45 * 80 + 45].is_finite(), "inside pixel nan");
    let expected = dem().data()[5 * W + 5];
    assert!((band.data()[45 * 80 + 45] - expected).abs() < 1e-12);
}

async fn range_handler(
    State(bytes): State<Arc<Vec<u8>>>,
    headers: HeaderMap,
) -> (StatusCode, Vec<u8>) {
    let Some(range) = headers.get(header::RANGE) else {
        return (StatusCode::OK, bytes.as_slice().to_vec());
    };
    let spec = range.to_str().unwrap().trim_start_matches("bytes=");
    let (s, e) = spec.split_once('-').unwrap();
    let (s, e): (usize, usize) = (s.parse().unwrap(), e.parse().unwrap());
    (
        StatusCode::PARTIAL_CONTENT,
        bytes[s..=e.min(bytes.len() - 1)].to_vec(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn http_transport_feeds_the_engine() {
    let bytes = Arc::new(cog_bytes(2));
    let app = axum::Router::new()
        .route("/dem.tif", get(range_handler))
        .with_state(bytes);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{addr}/dem.tif");
    let src = tokio::task::spawn_blocking(move || CogSrc::open(HttpRange::new(url)))
        .await
        .unwrap()
        .unwrap();

    let mut g = Graph::new();
    let s = g.add_source(Box::new(src));
    let hs = g.add_transform(s, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let mut g = Graph::new();
    let s = g.add_source(Box::new(mem_src()));
    let mhs = g.add_transform(s, Box::new(Hillshade::new(315.0, 45.0)));
    let mem = Engine::new(g, 64 << 20).unwrap();

    let req = WindowReq {
        bbox: window(20, 20, 340, 230),
        resolution: CELL,
    };
    let a = engine.pull(hs, req).await.unwrap().into_raster().unwrap();
    let b = mem.pull(mhs, req).await.unwrap().into_raster().unwrap();
    assert_bands_close(&a, &b, 1e-9);
}
