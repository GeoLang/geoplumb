//! stac source against a mock api: search, crs and band-count filtering,
//! `next`-link pagination, lazy cog opens over range requests,
//! most-recent-first mosaicking with deflate cogs band by band, and lazy
//! per-window block searches past the open bbox

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::routing::get;
use geoplumb::element::Source;
use geoplumb::elements::{Composite, StacSearch, StacSrc, stac::s3_to_https};
use geoplumb::{Bbox, Crs, Engine, Graph, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, CogParams, Raster, write_cog, write_cog_bands};

const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

fn elevation(lon: f64, lat: f64) -> f64 {
    500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos()
}

/// cog over [x0, x0+cols] pixels of the shared scene. `shift` marks the
/// values so tests can tell which item won a pixel. `hole` blanks rows
/// 100..120 to nodata
fn cog(x0: usize, cols: usize, rows: usize, shift: f64, hole: bool) -> Vec<u8> {
    let mut data = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let lon = ORIGIN_X + (x0 + col) as f64 * CELL + 0.5 * CELL;
            let lat = ORIGIN_Y - row as f64 * CELL - 0.5 * CELL;
            data.push(if hole && (100..120).contains(&row) {
                f64::NAN
            } else {
                elevation(lon, lat) + shift
            });
        }
    }
    let raster = Raster::from_vec(cols, rows, data, CELL, f64::NAN).unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    write_cog(&raster, &params(x0), &mut buf).unwrap();
    buf.into_inner()
}

fn params(x0: usize) -> CogParams {
    CogParams {
        tile_width: 64,
        tile_height: 64,
        overview_levels: 2,
        epsg: 4326,
        origin_x: ORIGIN_X + x0 as f64 * CELL,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        deflate: true,
    }
}

const MB_BANDS: usize = 3;

/// rows band `b` of a holed multi-band item leaves as nodata. staggered
/// per band so a fill that keys every band off band 0 shows up
fn hole_rows(b: usize) -> std::ops::Range<usize> {
    let start = 100 + b * 30;
    start..start + 20
}

fn mb_value(b: usize, lon: f64, lat: f64, shift: f64) -> f64 {
    elevation(lon, lat) * (b as f64 + 1.0) + shift
}

/// the multi-band twin of `cog`: `bands` planes, each a scaled copy of the
/// scene, each with its own nodata rows when `hole` is set
fn cog_mb(x0: usize, cols: usize, rows: usize, shift: f64, hole: bool, bands: usize) -> Vec<u8> {
    let planes = (0..bands)
        .map(|b| {
            let mut data = Vec::with_capacity(cols * rows);
            for row in 0..rows {
                for col in 0..cols {
                    let lon = ORIGIN_X + (x0 + col) as f64 * CELL + 0.5 * CELL;
                    let lat = ORIGIN_Y - row as f64 * CELL - 0.5 * CELL;
                    data.push(if hole && hole_rows(b).contains(&row) {
                        f64::NAN
                    } else {
                        mb_value(b, lon, lat, shift)
                    });
                }
            }
            Raster::from_vec(cols, rows, data, CELL, f64::NAN).unwrap()
        })
        .collect();
    let mut buf = std::io::Cursor::new(Vec::new());
    write_cog_bands(&BandedRaster::new(planes).unwrap(), &params(x0), &mut buf).unwrap();
    buf.into_inner()
}

struct Mock {
    base: String,
    cogs: std::collections::HashMap<String, Vec<u8>>,
    features: Vec<serde_json::Value>,
    /// http requests to /search, so a paged search counts more than one
    searches: AtomicUsize,
}

fn item(id: &str, dt: &str, href: &str, bbox: [f64; 4], epsg: u32) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "bbox": bbox,
        "properties": { "datetime": dt, "proj:epsg": epsg },
        "assets": { "data": { "href": href } }
    })
}

/// same, with the asset declaring `raster:bands`
fn banded_item(
    id: &str,
    dt: &str,
    href: &str,
    bbox: [f64; 4],
    epsg: u32,
    bands: usize,
) -> serde_json::Value {
    let mut f = item(id, dt, href, bbox, epsg);
    let decl: Vec<serde_json::Value> = (0..bands)
        .map(|b| serde_json::json!({ "name": format!("b{b}") }))
        .collect();
    f["assets"]["data"]["raster:bands"] = serde_json::Value::Array(decl);
    f
}

/// filters by the bbox param and pages at the limit param, like a real
/// api: each page past the first is reached through a `next` link, so
/// both the lazy block searches and pagination are exercised
async fn serve_search(
    State(mock): State<Arc<Mock>>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> ([(&'static str, &'static str); 1], String) {
    mock.searches.fetch_add(1, Ordering::SeqCst);
    let q: Vec<f64> = params["bbox"]
        .split(',')
        .map(|v| v.parse().unwrap())
        .collect();
    let limit: usize = params["limit"].parse().unwrap();
    let page: usize = params.get("page").map_or(0, |v| v.parse().unwrap());
    let matching: Vec<&serde_json::Value> = mock
        .features
        .iter()
        .filter(|f| {
            let b: Vec<f64> = f["bbox"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            b[0] <= q[2] && b[2] >= q[0] && b[1] <= q[3] && b[3] >= q[1]
        })
        .collect();
    let hits: Vec<&serde_json::Value> = matching
        .iter()
        .skip(page * limit)
        .take(limit)
        .copied()
        .collect();
    let mut body = serde_json::json!({ "type": "FeatureCollection", "features": hits });
    if (page + 1) * limit < matching.len() {
        let mut next = params.clone();
        next.insert("page".into(), (page + 1).to_string());
        let qs: Vec<String> = next.iter().map(|(k, v)| format!("{k}={v}")).collect();
        body["links"] = serde_json::json!([
            { "rel": "self", "href": format!("{}/search", mock.base) },
            { "rel": "next", "href": format!("{}/search?{}", mock.base, qs.join("&")) },
        ]);
    }
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

type Scene = (
    std::collections::HashMap<String, Vec<u8>>,
    Vec<serde_json::Value>,
);

async fn start_mock_with(build: impl FnOnce(&str) -> Scene) -> (String, Arc<Mock>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let (cogs, features) = build(&base);
    let mock = Arc::new(Mock {
        base: base.clone(),
        cogs,
        features,
        searches: AtomicUsize::new(0),
    });
    let app = axum::Router::new()
        .route("/search", get(serve_search))
        .route("/cog/{name}", get(serve_cog))
        .with_state(mock.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, mock)
}

async fn start_mock() -> (String, Arc<Mock>) {
    start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        // the 2024 pair covers lon 7.0..7.3 (with a nodata hole) and 7.3..7.6
        cogs.insert("left.tif".into(), cog(0, 300, 400, 0.0, true));
        cogs.insert("right.tif".into(), cog(300, 300, 400, 0.0, false));
        // the 2020 item covers everything, values shifted so wins are visible
        cogs.insert("old.tif".into(), cog(0, 600, 400, 1000.0, false));
        let features = vec![
            item(
                "old",
                "2020-01-01T00:00:00Z",
                &format!("{base}/cog/old.tif"),
                [7.0, 46.6, 7.6, 47.0],
                4326,
            ),
            item(
                "left",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/left.tif"),
                [7.0, 46.6, 7.3, 47.0],
                4326,
            ),
            item(
                "utm",
                "2019-01-01T00:00:00Z",
                &format!("{base}/cog/missing.tif"),
                [7.0, 46.6, 7.6, 47.0],
                32632,
            ),
            item(
                "right",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/right.tif"),
                [7.3, 46.6, 7.6, 47.0],
                4326,
            ),
        ];
        (cogs, features)
    })
    .await
}

/// four overlapping single-band items so a pixel can be three deep: old
/// (+1000) covers everything, mid (+100) the middle, and the 2024 pair
/// the two halves, left holing out rows 100..120. the shifts are spaced
/// unevenly on purpose, so a median cannot pass as a mean
async fn start_composite_mock() -> (String, Arc<Mock>) {
    start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        cogs.insert("c_old.tif".into(), cog(0, 600, 400, 1000.0, false));
        cogs.insert("c_mid.tif".into(), cog(100, 400, 400, 100.0, false));
        cogs.insert("c_left.tif".into(), cog(0, 300, 400, 0.0, true));
        cogs.insert("c_right.tif".into(), cog(300, 300, 400, 0.0, false));
        let features = vec![
            item(
                "c_left",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/c_left.tif"),
                [7.0, 46.6, 7.3, 47.0],
                4326,
            ),
            item(
                "c_right",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/c_right.tif"),
                [7.3, 46.6, 7.6, 47.0],
                4326,
            ),
            item(
                "c_mid",
                "2022-01-01T00:00:00Z",
                &format!("{base}/cog/c_mid.tif"),
                [7.1, 46.6, 7.5, 47.0],
                4326,
            ),
            item(
                "c_old",
                "2020-01-01T00:00:00Z",
                &format!("{base}/cog/c_old.tif"),
                [7.0, 46.6, 7.6, 47.0],
                4326,
            ),
        ];
        (cogs, features)
    })
    .await
}

/// the same scene in three bands, plus a single-band item the band filter
/// must drop. the newest items hole out different rows in each band
async fn start_mb_mock() -> (String, Arc<Mock>) {
    start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        cogs.insert(
            "mb_left.tif".into(),
            cog_mb(0, 300, 400, 0.0, true, MB_BANDS),
        );
        cogs.insert(
            "mb_right.tif".into(),
            cog_mb(300, 300, 400, 0.0, false, MB_BANDS),
        );
        cogs.insert(
            "mb_old.tif".into(),
            cog_mb(0, 600, 400, 1000.0, false, MB_BANDS),
        );
        cogs.insert("mb_mono.tif".into(), cog_mb(0, 600, 400, 5000.0, false, 1));
        let features = vec![
            banded_item(
                "mb_left",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/mb_left.tif"),
                [7.0, 46.6, 7.3, 47.0],
                4326,
                MB_BANDS,
            ),
            banded_item(
                "mb_right",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/mb_right.tif"),
                [7.3, 46.6, 7.6, 47.0],
                4326,
                MB_BANDS,
            ),
            banded_item(
                "mb_old",
                "2020-01-01T00:00:00Z",
                &format!("{base}/cog/mb_old.tif"),
                [7.0, 46.6, 7.6, 47.0],
                4326,
                MB_BANDS,
            ),
            banded_item(
                "mb_mono",
                "2018-01-01T00:00:00Z",
                &format!("{base}/cog/mb_mono.tif"),
                [7.0, 46.6, 7.6, 47.0],
                4326,
                1,
            ),
        ];
        (cogs, features)
    })
    .await
}

async fn open_src(base: &str) -> StacSrc {
    let search = StacSearch::new(base, "test-dem", "data", [7.0, 46.6, 7.6, 47.0]);
    tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_filters_crs_and_anchors_the_grid() {
    let (base, _mock) = start_mock().await;
    let src = open_src(&base).await;
    // the utm item is filtered, three wgs84 items remain
    assert_eq!(src.item_count(), 3);
    let g = src.grid();
    assert!((g.base_resolution - CELL).abs() < 1e-15);
    assert_eq!(src.crs(), Crs::WGS84);
}

/// spans the left/right item seam at lon 7.3 and the left item's holes
const SEAM: WindowReq = WindowReq {
    bbox: Bbox {
        min_x: 7.2,
        max_x: 7.4,
        max_y: ORIGIN_Y - 0.05,
        min_y: ORIGIN_Y - 0.15,
    },
    resolution: CELL,
};

/// every pixel is the 2024 value except in the left item's hole, where the
/// shifted 2020 value shows through
fn assert_seam_mosaic(chunk: &RasterChunk) {
    let band = chunk.bands.band(0).unwrap();
    let res = chunk.resolution;
    for row in 0..band.height() {
        for col in 0..band.width() {
            let x = chunk.bbox.min_x + (col as f64 + 0.5) * res;
            let y = chunk.bbox.max_y - (row as f64 + 0.5) * res;
            if !(7.2..=7.4).contains(&x) || !(ORIGIN_Y - 0.15..=ORIGIN_Y - 0.05).contains(&y) {
                continue;
            }
            let got = band.data()[row * band.width() + col];
            let scene_row = ((ORIGIN_Y - y) / CELL - 0.5).round() as usize;
            let in_hole = x < 7.3 && (100..120).contains(&scene_row);
            let expected = elevation(x, y) + if in_hole { 1000.0 } else { 0.0 };
            assert!(
                (got - expected).abs() < 1e-6,
                "({x:.4},{y:.4}) hole={in_hole}: {got} vs {expected}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recent_items_win_and_old_fills_their_holes() {
    let (base, _mock) = start_mock().await;
    let src = open_src(&base).await;
    let mut g = Graph::new();
    let n = g.add_source(Box::new(src));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let chunk = engine.pull(n, SEAM).await.unwrap().into_raster().unwrap();
    assert_seam_mosaic(&chunk);
}

/// the open bbox is only an anchor: a pull past it finds its items
/// through a lazy block search, and the blocks are searched once
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pulls_past_the_open_bbox_search_lazily_and_cache_blocks() {
    let (base, mock) = start_mock().await;
    let search = StacSearch::new(&base, "test-dem", "data", [7.0, 46.6, 7.2, 47.0]);
    let src = tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap();
    // the right item starts at lon 7.3, outside the open bbox
    assert_eq!(src.item_count(), 2);
    assert_eq!(mock.searches.load(Ordering::SeqCst), 1);

    let req = WindowReq {
        bbox: Bbox {
            min_x: 7.35,
            max_x: 7.45,
            max_y: 46.95,
            min_y: 46.85,
        },
        resolution: CELL,
    };
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    assert_eq!(src.item_count(), 3, "the right item was not discovered");
    assert_eq!(mock.searches.load(Ordering::SeqCst), 2);

    let band = chunk.bands.band(0).unwrap();
    for row in 0..band.height() {
        for col in 0..band.width() {
            let x = req.bbox.min_x + (col as f64 + 0.5) * CELL;
            let y = req.bbox.max_y - (row as f64 + 0.5) * CELL;
            let got = band.data()[row * band.width() + col];
            // the 2024 right item wins over the shifted 2020 item
            let expected = elevation(x, y);
            assert!(
                (got - expected).abs() < 1e-6,
                "({x:.4},{y:.4}): {got} vs {expected}"
            );
        }
    }

    // another window in the same block reuses the cached search
    let again = WindowReq {
        bbox: Bbox {
            min_x: 7.05,
            max_x: 7.15,
            max_y: 46.95,
            min_y: 46.85,
        },
        resolution: CELL,
    };
    src.read(&again).await.unwrap();
    assert_eq!(
        mock.searches.load(Ordering::SeqCst),
        2,
        "block searched twice"
    );
}

/// a page-sized search used to fail the pull rather than serve partial
/// coverage. now it follows the api's `next` links, so a page size of one
/// finds every item and the mosaic is the same as an unpaged search
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn searches_spanning_pages_still_cover_the_window() {
    let (base, mock) = start_mock().await;
    let mut search = StacSearch::new(&base, "test-dem", "data", [7.0, 46.6, 7.6, 47.0]);
    search.limit = 1;
    let src = tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap();
    // four features over the anchor bbox, one per page, the utm one dropped
    assert_eq!(src.item_count(), 3);
    assert_eq!(
        mock.searches.load(Ordering::SeqCst),
        4,
        "the open search did not page"
    );

    let mut g = Graph::new();
    let n = g.add_source(Box::new(src));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let chunk = engine.pull(n, SEAM).await.unwrap().into_raster().unwrap();
    assert_seam_mosaic(&chunk);
}

/// past the accumulation cap a search is asking for a mosaic of thousands
/// of items, so it fails loud instead of paging on
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_search_past_the_item_cap_fails_loud() {
    let (base, _mock) = start_mock_with(|base| {
        let features = (0..1200)
            .map(|i| {
                item(
                    &format!("f{i}"),
                    "2024-01-01T00:00:00Z",
                    &format!("{base}/cog/f{i}.tif"),
                    [7.0, 46.6, 7.6, 47.0],
                    4326,
                )
            })
            .collect();
        (std::collections::HashMap::new(), features)
    })
    .await;
    let search = StacSearch::new(&base, "test-dem", "data", [7.0, 46.6, 7.6, 47.0]);
    let opened = tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap();
    let err = match opened {
        Ok(_) => panic!("the item cap did not fire"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("1000 items"), "{err}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn search_filters_band_count_and_anchors_it() {
    let (base, _mock) = start_mb_mock().await;
    let search = StacSearch::new(&base, "test-dem", "data", [7.0, 46.6, 7.6, 47.0]);
    let src = tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap();
    // the single-band item is filtered, three three-band items remain
    assert_eq!(src.item_count(), 3);
    assert_eq!(src.bands(), MB_BANDS as u16);

    let mut g = Graph::new();
    let n = g.add_source(Box::new(src));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(n).raster().bands, MB_BANDS as u16);
}

/// the newest item holes out different rows in each band, so each band is
/// filled from the older item independently
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn multi_band_mosaic_fills_holes_per_band() {
    let (base, _mock) = start_mb_mock().await;
    let search = StacSearch::new(&base, "test-dem", "data", [7.0, 46.6, 7.6, 47.0]);
    let src = tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap();
    let mut g = Graph::new();
    let n = g.add_source(Box::new(src));
    let engine = Engine::new(g, 64 << 20).unwrap();

    // spans the seam at lon 7.3 and all three per-band hole ranges
    let req = WindowReq {
        bbox: Bbox {
            min_x: 7.2,
            max_x: 7.4,
            max_y: ORIGIN_Y - 0.05,
            min_y: ORIGIN_Y - 0.20,
        },
        resolution: CELL,
    };
    let chunk = engine.pull(n, req).await.unwrap().into_raster().unwrap();
    assert_eq!(chunk.bands.band_count(), MB_BANDS);
    let res = chunk.resolution;
    let mut holes_hit = 0;
    for (bi, band) in chunk.bands.bands().iter().enumerate() {
        for row in 0..band.height() {
            for col in 0..band.width() {
                let x = chunk.bbox.min_x + (col as f64 + 0.5) * res;
                let y = chunk.bbox.max_y - (row as f64 + 0.5) * res;
                if !(7.2..=7.4).contains(&x) || !(ORIGIN_Y - 0.20..=ORIGIN_Y - 0.05).contains(&y) {
                    continue;
                }
                let got = band.data()[row * band.width() + col];
                let scene_row = ((ORIGIN_Y - y) / CELL - 0.5).round() as usize;
                let in_hole = x < 7.3 && hole_rows(bi).contains(&scene_row);
                holes_hit += usize::from(in_hole);
                let expected = mb_value(bi, x, y, if in_hole { 1000.0 } else { 0.0 });
                assert!(
                    (got - expected).abs() < 1e-6,
                    "band {bi} ({x:.4},{y:.4}) hole={in_hole}: {got} vs {expected}"
                );
            }
        }
    }
    assert!(holes_hit > 1000, "the window missed the holes");
}

#[test]
fn s3_hrefs_rewrite_to_https() {
    assert_eq!(
        s3_to_https("s3://copernicus-dem-30m/some/key.tif"),
        "https://copernicus-dem-30m.s3.amazonaws.com/some/key.tif"
    );
    assert_eq!(s3_to_https("https://x/y.tif"), "https://x/y.tif");
}

async fn open_composite(base: &str, composite: Composite) -> StacSrc {
    let mut search = StacSearch::new(base, "test-dem", "data", [7.0, 46.6, 7.6, 47.0]);
    search.composite = composite;
    tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap()
}

/// walks every band and pixel of an exactly-sized read, handing the check
/// the pixel center and the value there
fn each_pixel(chunk: &RasterChunk, mut f: impl FnMut(usize, f64, f64, f64)) {
    let res = chunk.resolution;
    for (bi, band) in chunk.bands.bands().iter().enumerate() {
        for row in 0..band.height() {
            for col in 0..band.width() {
                let x = chunk.bbox.min_x + (col as f64 + 0.5) * res;
                let y = chunk.bbox.max_y - (row as f64 + 0.5) * res;
                f(bi, x, y, band.data()[row * band.width() + col]);
            }
        }
    }
}

fn scene_row(y: f64) -> usize {
    ((ORIGIN_Y - y) / CELL - 0.5).round() as usize
}

/// lon 7.2..7.3 over rows 50..150: the left item covers it and holes out
/// rows 100..120, the old item covers all of it shifted by 1000
const OVERLAP: WindowReq = WindowReq {
    bbox: Bbox {
        min_x: 7.2,
        max_x: 7.3,
        max_y: ORIGIN_Y - 0.05,
        min_y: ORIGIN_Y - 0.15,
    },
    resolution: CELL,
};

fn close(got: f64, want: f64, at: (f64, f64)) {
    assert!(
        (got - want).abs() < 1e-6,
        "({:.4},{:.4}): {got} vs {want}",
        at.0,
        at.1
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn mean_averages_the_items_and_skips_their_nodata() {
    let (base, _mock) = start_mock().await;
    let src = open_composite(&base, Composite::Mean).await;
    let chunk = src.read(&OVERLAP).await.unwrap().into_raster().unwrap();
    let mut in_hole = 0;
    each_pixel(&chunk, |_, x, y, got| {
        let e = elevation(x, y);
        // outside the hole both items contribute, inside only the old one,
        // and its nodata must not drag the average toward zero
        if (100..120).contains(&scene_row(y)) {
            in_hole += 1;
            close(got, e + 1000.0, (x, y));
        } else {
            close(got, e + 500.0, (x, y));
        }
    });
    assert!(in_hole > 1000, "the window missed the hole");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn median_takes_the_middle_of_three_and_the_mean_of_two() {
    let (base, _mock) = start_composite_mock().await;
    let src = open_composite(&base, Composite::Median).await;
    assert_eq!(src.item_count(), 4);
    let chunk = src.read(&OVERLAP).await.unwrap().into_raster().unwrap();
    let (mut three_deep, mut two_deep) = (0, 0);
    each_pixel(&chunk, |_, x, y, got| {
        let e = elevation(x, y);
        if (100..120).contains(&scene_row(y)) {
            // left holed out, leaving mid and old: the mean of the two
            two_deep += 1;
            close(got, e + 550.0, (x, y));
        } else {
            // left (+0), mid (+100) and old (+1000) stack here, so the
            // middle value is mid, well away from their mean of +366.67
            three_deep += 1;
            close(got, e + 100.0, (x, y));
        }
    });
    assert!(three_deep > 1000 && two_deep > 1000, "window missed a case");

    // the same window under Mean, to pin that the two are distinct
    let src = open_composite(&base, Composite::Mean).await;
    let chunk = src.read(&OVERLAP).await.unwrap().into_raster().unwrap();
    each_pixel(&chunk, |_, x, y, got| {
        let e = elevation(x, y);
        if (100..120).contains(&scene_row(y)) {
            close(got, e + 550.0, (x, y));
        } else {
            close(got, e + 1100.0 / 3.0, (x, y));
        }
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn min_and_max_pick_the_unshifted_and_shifted_items() {
    let (base, _mock) = start_mock().await;
    for (op, unholed, holed) in [
        (Composite::Min, 0.0, 1000.0),
        (Composite::Max, 1000.0, 1000.0),
    ] {
        let src = open_composite(&base, op).await;
        let chunk = src.read(&OVERLAP).await.unwrap().into_raster().unwrap();
        each_pixel(&chunk, |_, x, y, got| {
            let shift = if (100..120).contains(&scene_row(y)) {
                holed
            } else {
                unholed
            };
            close(got, elevation(x, y) + shift, (x, y));
        });
    }
}

/// lon 7.5..7.7 runs off the east edge of every item at 7.6, so the far
/// half has no finite value to reduce and must stay nodata
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_pixel_no_item_covers_stays_nodata() {
    let (base, _mock) = start_mock().await;
    let req = WindowReq {
        bbox: Bbox {
            min_x: 7.5,
            max_x: 7.7,
            max_y: ORIGIN_Y - 0.05,
            min_y: ORIGIN_Y - 0.15,
        },
        resolution: CELL,
    };
    let src = open_composite(&base, Composite::Min).await;
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    let (mut covered, mut bare) = (0, 0);
    each_pixel(&chunk, |_, x, y, got| {
        if x < 7.6 {
            covered += 1;
            close(got, elevation(x, y), (x, y));
        } else {
            bare += 1;
            assert!(got.is_nan(), "({x:.4},{y:.4}) past every item: {got}");
        }
    });
    assert!(covered > 1000 && bare > 1000, "window missed a case");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_multi_band_composite_reduces_each_band_independently() {
    let (base, _mock) = start_mb_mock().await;
    let src = open_composite(&base, Composite::Mean).await;
    assert_eq!(src.bands(), MB_BANDS as u16);
    // taller than OVERLAP: the per-band holes run down to row 180
    let req = WindowReq {
        bbox: Bbox {
            min_y: ORIGIN_Y - 0.20,
            ..OVERLAP.bbox
        },
        ..OVERLAP
    };
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    assert_eq!(chunk.bands.band_count(), MB_BANDS);
    let mut holes_hit = [0usize; MB_BANDS];
    each_pixel(&chunk, |bi, x, y, got| {
        // each band holes out its own rows, so the band that falls back to
        // the old item alone differs row by row
        let shift = if hole_rows(bi).contains(&scene_row(y)) {
            holes_hit[bi] += 1;
            1000.0
        } else {
            500.0
        };
        close(got, mb_value(bi, x, y, shift), (x, y));
    });
    assert!(
        holes_hit.iter().all(|&n| n > 1000),
        "a band's hole was missed: {holes_hit:?}"
    );
}
