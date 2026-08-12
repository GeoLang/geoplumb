//! stac source against a mock api: search, band-count filtering,
//! `next`-link pagination, lazy cog opens over range requests,
//! most-recent-first mosaicking with deflate cogs band by band, items on
//! another crs warped onto the anchor grid, and lazy per-window block
//! searches past the open bbox

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::routing::get;
use geoplumb::element::Source;
use geoplumb::elements::{Composite, StacSearch, StacSrc, stac::s3_to_https};
use geoplumb::{Bbox, Crs, Engine, Graph, RasterChunk, TimeInterval, WindowReq};
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
    // the api's interval is closed at both ends, and every datetime here
    // is fixed-width utc, so string order is time order
    let time = params.get("datetime").map(|dt| {
        let (start, end) = dt.split_once('/').unwrap();
        (start.to_string(), end.to_string())
    });
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
            let dt = f["properties"]["datetime"].as_str().unwrap();
            let in_time = time
                .as_ref()
                .is_none_or(|(start, end)| dt >= start.as_str() && dt <= end.as_str());
            in_time && b[0] <= q[2] && b[2] >= q[0] && b[1] <= q[3] && b[3] >= q[1]
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
async fn open_anchors_the_grid_on_the_newest_item() {
    let (base, _mock) = start_mock().await;
    let src = open_src(&base).await;
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
    time: None,
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

/// a pull carrying an interval overrides the source's own datetime, and
/// each interval gets its own block searches and its own item set: the
/// 2024 window is the recent pair with the left item's hole left open,
/// the 2020 window is the older item alone
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn pull_time_picks_the_items_and_keys_the_block_cache() {
    let (base, mock) = start_mock().await;
    let src = open_src(&base).await;
    assert_eq!(mock.searches.load(Ordering::SeqCst), 1);

    let recent = SEAM.with_time(Some(
        TimeInterval::parse("2024-01-01T00:00:00Z/2025-01-01T00:00:00Z").unwrap(),
    ));
    let older = SEAM.with_time(Some(
        TimeInterval::parse("2019-06-01T00:00:00Z/2020-06-01T00:00:00Z").unwrap(),
    ));

    let chunk = src.read(&recent).await.unwrap().into_raster().unwrap();
    assert_eq!(mock.searches.load(Ordering::SeqCst), 2);
    assert_seam_at(&chunk, |in_hole| match in_hole {
        // nothing older is in this interval to patch the hole
        true => f64::NAN,
        false => 0.0,
    });

    let chunk = src.read(&older).await.unwrap().into_raster().unwrap();
    assert_eq!(mock.searches.load(Ordering::SeqCst), 3);
    assert_seam_at(&chunk, |_| 1000.0);

    // each interval's block is searched once, and the two item sets stay
    // apart: the recent window is still hole-free of the older item
    let chunk = src.read(&recent).await.unwrap().into_raster().unwrap();
    assert_eq!(
        mock.searches.load(Ordering::SeqCst),
        3,
        "the block was searched twice at one time"
    );
    assert_seam_at(&chunk, |in_hole| match in_hole {
        true => f64::NAN,
        false => 0.0,
    });
    // the two searches shared one copy of the items they both matched
    assert_eq!(src.item_count(), 3);
}

/// the seam window with a per-pixel shift over the scene, `None` where the
/// pixel should have no value at all
fn assert_seam_at(chunk: &RasterChunk, shift: impl Fn(bool) -> f64) {
    let band = chunk.bands.band(0).unwrap();
    let res = chunk.resolution;
    for row in 0..band.height() {
        for col in 0..band.width() {
            let x = chunk.bbox.min_x + (col as f64 + 0.5) * res;
            let y = chunk.bbox.max_y - (row as f64 + 0.5) * res;
            let got = band.data()[row * band.width() + col];
            let scene_row = ((ORIGIN_Y - y) / CELL - 0.5).round() as usize;
            let shift = shift(x < 7.3 && (100..120).contains(&scene_row));
            if shift.is_nan() {
                assert!(got.is_nan(), "({x:.4},{y:.4}): {got} should be nodata");
                continue;
            }
            let expected = elevation(x, y) + shift;
            assert!(
                (got - expected).abs() < 1e-6,
                "({x:.4},{y:.4}): {got} vs {expected}"
            );
        }
    }
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
        time: None,
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
        time: None,
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
    // three features over the anchor bbox, one per page
    assert_eq!(src.item_count(), 3);
    assert_eq!(
        mock.searches.load(Ordering::SeqCst),
        3,
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
        time: None,
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

/// item whose bands live in one cog per asset, sentinel-2 style
fn asset_item(
    id: &str,
    dt: &str,
    assets: &[(&str, String)],
    bbox: [f64; 4],
    epsg: u32,
) -> serde_json::Value {
    let mut f = serde_json::json!({
        "id": id,
        "bbox": bbox,
        "properties": { "datetime": dt, "proj:epsg": epsg },
        "assets": {}
    });
    for (key, href) in assets {
        f["assets"][key] = serde_json::json!({ "href": href });
    }
    f
}

/// nir values sit 5000 over red everywhere, so band order is visible
const NIR_SHIFT: f64 = 5000.0;

/// per-band-asset scene: the 2024 pair carries red and nir cogs, red left
/// holed, the 2020 item covers everything shifted by 1000, and one item
/// only has a red asset
async fn start_asset_mock() -> (String, Arc<Mock>) {
    start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        cogs.insert("r_left.tif".into(), cog(0, 300, 400, 0.0, true));
        cogs.insert("n_left.tif".into(), cog(0, 300, 400, NIR_SHIFT, false));
        cogs.insert("r_right.tif".into(), cog(300, 300, 400, 0.0, false));
        cogs.insert("n_right.tif".into(), cog(300, 300, 400, NIR_SHIFT, false));
        cogs.insert("r_old.tif".into(), cog(0, 600, 400, 1000.0, false));
        cogs.insert(
            "n_old.tif".into(),
            cog(0, 600, 400, NIR_SHIFT + 1000.0, false),
        );
        let features = vec![
            asset_item(
                "a_left",
                "2024-06-01T00:00:00Z",
                &[
                    ("red", format!("{base}/cog/r_left.tif")),
                    ("nir", format!("{base}/cog/n_left.tif")),
                ],
                [7.0, 46.6, 7.3, 47.0],
                4326,
            ),
            asset_item(
                "a_right",
                "2024-06-01T00:00:00Z",
                &[
                    ("red", format!("{base}/cog/r_right.tif")),
                    ("nir", format!("{base}/cog/n_right.tif")),
                ],
                [7.3, 46.6, 7.6, 47.0],
                4326,
            ),
            asset_item(
                "a_old",
                "2020-01-01T00:00:00Z",
                &[
                    ("red", format!("{base}/cog/r_old.tif")),
                    ("nir", format!("{base}/cog/n_old.tif")),
                ],
                [7.0, 46.6, 7.6, 47.0],
                4326,
            ),
            asset_item(
                "a_partial",
                "2025-01-01T00:00:00Z",
                &[("red", format!("{base}/cog/r_old.tif"))],
                [7.0, 46.6, 7.6, 47.0],
                4326,
            ),
        ];
        (cogs, features)
    })
    .await
}

async fn open_assets(base: &str, assets: &[&str]) -> Result<StacSrc, geoplumb::Error> {
    let mut search = StacSearch::new(base, "test-s2", "red", [7.0, 46.6, 7.6, 47.0]);
    search.assets = assets.iter().map(|a| a.to_string()).collect();
    tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
}

/// per-band assets stack into one raster in asset order, items missing an
/// asset are skipped, and the mosaic still fills each band independently:
/// the newest red is holed and fills from the old item, nir is not
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn per_asset_cogs_stack_in_asset_order() {
    let (base, _mock) = start_asset_mock().await;
    let src = open_assets(&base, &["red", "nir"]).await.unwrap();
    // the partial item lacks a nir asset, three full items remain
    assert_eq!(src.item_count(), 3);
    assert_eq!(src.bands(), 2);

    let chunk = src.read(&SEAM).await.unwrap().into_raster().unwrap();
    assert_eq!(chunk.bands.band_count(), 2);
    let mut in_hole = 0;
    each_pixel(&chunk, |bi, x, y, got| {
        let holed = x < 7.3 && (100..120).contains(&scene_row(y));
        let expected = match bi {
            // red: the 2024 pair, with the old item showing in the hole
            0 => elevation(x, y) + if holed { 1000.0 } else { 0.0 },
            // nir: nothing holed, the 2024 pair everywhere
            _ => elevation(x, y) + NIR_SHIFT,
        };
        in_hole += usize::from(holed && bi == 0);
        close(got, expected, (x, y));
    });
    assert!(in_hole > 1000, "the window missed the red hole");
}

/// assets at different resolutions cannot stack, the open fails loud
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn assets_on_different_grids_fail_at_open() {
    let (base, _mock) = start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        cogs.insert("fine.tif".into(), cog(0, 300, 400, 0.0, false));
        let coarse = {
            let mut data = Vec::with_capacity(150 * 200);
            for row in 0..200 {
                for col in 0..150 {
                    let lon = ORIGIN_X + col as f64 * 2.0 * CELL + CELL;
                    let lat = ORIGIN_Y - row as f64 * 2.0 * CELL - CELL;
                    data.push(elevation(lon, lat));
                }
            }
            let raster = Raster::from_vec(150, 200, data, 2.0 * CELL, f64::NAN).unwrap();
            let mut params = params(0);
            params.pixel_width = 2.0 * CELL;
            params.pixel_height = 2.0 * CELL;
            let mut buf = std::io::Cursor::new(Vec::new());
            write_cog(&raster, &params, &mut buf).unwrap();
            buf.into_inner()
        };
        cogs.insert("coarse.tif".into(), coarse);
        let features = vec![asset_item(
            "mixed",
            "2024-06-01T00:00:00Z",
            &[
                ("red", format!("{base}/cog/fine.tif")),
                ("nir", format!("{base}/cog/coarse.tif")),
            ],
            [7.0, 46.6, 7.3, 47.0],
            4326,
        )];
        (cogs, features)
    })
    .await;
    let err = match open_assets(&base, &["red", "nir"]).await {
        Ok(_) => panic!("mixed-resolution assets opened"),
        Err(e) => e,
    };
    assert!(err.to_string().contains("different resolutions"), "{err}");
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
    time: None,
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
        time: None,
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

/// seventy co-located items, one per day, shifts spaced unevenly so a
/// scrambled read order or a dropped item shows in the reduce
async fn start_deep_stack_mock() -> (String, Arc<Mock>) {
    start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        let mut features = Vec::new();
        for i in 0..70usize {
            let name = format!("stack_{i}.tif");
            cogs.insert(name.clone(), cog(0, 64, 64, (i * i) as f64, false));
            features.push(item(
                &format!("stack_{i}"),
                &format!("2024-{:02}-{:02}T00:00:00Z", i / 28 + 1, i % 28 + 1),
                &format!("{base}/cog/{name}"),
                [7.0, 46.936, 7.064, 47.0],
                4326,
            ));
        }
        (cogs, features)
    })
    .await
}

/// the deep stack's full footprint
const STACK: WindowReq = WindowReq {
    bbox: Bbox {
        min_x: 7.0,
        max_x: 7.064,
        max_y: ORIGIN_Y,
        min_y: ORIGIN_Y - 0.064,
    },
    resolution: CELL,
    time: None,
};

/// a stack deeper than the parallel read cap, so the reads span more than
/// one wave of permits
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stack_deeper_than_the_read_cap_reduces_and_fills_in_order() {
    let (base, _mock) = start_deep_stack_mock().await;
    let src = open_composite(&base, Composite::Median).await;
    assert_eq!(src.item_count(), 70);
    let chunk = src.read(&STACK).await.unwrap().into_raster().unwrap();
    each_pixel(&chunk, |_, x, y, got| {
        // shifts are 0,1,4,..,4761: the median of the seventy is 1190.5
        close(got, elevation(x, y) + 1190.5, (x, y));
    });

    let src = open_composite(&base, Composite::Latest).await;
    let chunk = src.read(&STACK).await.unwrap().into_raster().unwrap();
    each_pixel(&chunk, |_, x, y, got| {
        // every item covers every pixel, the newest must win them all
        close(got, elevation(x, y) + 4761.0, (x, y));
    });
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn percentile_interpolates_between_the_neighbouring_ranks() {
    let (base, _mock) = start_deep_stack_mock().await;
    // shifts are 0,1,4,..,4761, so rank 62.1 of the seventy sits a tenth of
    // the way from 3844 to 3969, a value no item carries
    for (percent, want_shift) in [(0.0, 0.0), (50.0, 1190.5), (90.0, 3856.5), (100.0, 4761.0)] {
        let src = open_composite(&base, Composite::Percentile(percent)).await;
        let chunk = src.read(&STACK).await.unwrap().into_raster().unwrap();
        each_pixel(&chunk, |_, x, y, got| {
            close(got, elevation(x, y) + want_shift, (x, y));
        });
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_percent_outside_the_range_clamps_to_min_and_max() {
    let (base, _mock) = start_mock().await;
    for (percent, want_shift) in [(-10.0, 0.0), (110.0, 1000.0)] {
        let src = open_composite(&base, Composite::Percentile(percent)).await;
        let chunk = src.read(&OVERLAP).await.unwrap().into_raster().unwrap();
        each_pixel(&chunk, |_, x, y, got| {
            // the hole leaves the old item alone, at +1000 either way
            let shift = if (100..120).contains(&scene_row(y)) {
                1000.0
            } else {
                want_shift
            };
            close(got, elevation(x, y) + shift, (x, y));
        });
    }
}

/// the utm zone pair sentinel-2 straddles at lon 12, the case the anchor
/// crs alone cannot serve
const ZONE_32: u16 = 32632;
const ZONE_33: u16 = 32633;

/// cell of the mixed-crs scene, in metres
const UTM_CELL: f64 = 20.0;

/// top-left corner of the zone 32 anchor item, west of the zone boundary
/// but wide enough to run past it
const MIXED_LON: f64 = 11.9;
const MIXED_LAT: f64 = 47.0;
const MIXED_COLS: usize = 600;
const MIXED_ROWS: usize = 400;

/// cells the zone 33 item reaches past the anchor item's footprint, so a
/// window running off the anchor item still has cover and no test pixel
/// sits within a resampling kernel of the zone 33 item's own edge
const MIXED_MARGIN: usize = 100;

/// marks the zone 33 item's values, so a pixel says which item filled it
const MIXED_SHIFT: f64 = 2000.0;

/// a warped value is one bilinear cell of the scene's curvature away from
/// the scene the cogs were built from
const WARP_TOLERANCE: f64 = 1e-3;

fn to_lonlat(epsg: u16) -> projicio_core::Transform {
    projicio_core::Transform::new(&format!("EPSG:{epsg}"), "EPSG:4326").unwrap()
}

fn from_lonlat(epsg: u16) -> projicio_core::Transform {
    projicio_core::Transform::new("EPSG:4326", &format!("EPSG:{epsg}")).unwrap()
}

/// a utm pixel grid: where its top-left corner sits in its own metres and
/// how many `UTM_CELL` cells it holds
struct UtmGrid {
    epsg: u16,
    origin_x: f64,
    origin_y: f64,
    cols: usize,
    rows: usize,
}

impl UtmGrid {
    /// the grid whose top-left corner is this lon/lat
    fn at(epsg: u16, lon: f64, lat: f64, cols: usize, rows: usize) -> UtmGrid {
        let (origin_x, origin_y) = from_lonlat(epsg).convert(lon, lat).unwrap();
        UtmGrid {
            epsg,
            origin_x,
            origin_y,
            cols,
            rows,
        }
    }

    fn pixel_center(&self, col: usize, row: usize) -> (f64, f64) {
        (
            self.origin_x + (col as f64 + 0.5) * UTM_CELL,
            self.origin_y - (row as f64 + 0.5) * UTM_CELL,
        )
    }

    /// corners plus edge midpoints, the boundary points a rotated grid
    /// needs for an honest envelope in another crs
    fn boundary(&self) -> Vec<(f64, f64)> {
        let xs = [0.0, self.cols as f64 / 2.0, self.cols as f64];
        let ys = [0.0, self.rows as f64 / 2.0, self.rows as f64];
        let mut pts = Vec::with_capacity(9);
        for cols in xs {
            for rows in ys {
                pts.push((
                    self.origin_x + cols * UTM_CELL,
                    self.origin_y - rows * UTM_CELL,
                ));
            }
        }
        pts
    }

    /// the grid's lon/lat footprint, the bbox a stac item declares
    fn footprint(&self) -> [f64; 4] {
        let pts = to_lonlat(self.epsg)
            .convert_batch(&self.boundary())
            .unwrap();
        let mut out = [
            f64::INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NEG_INFINITY,
        ];
        for (lon, lat) in pts {
            out = [
                out[0].min(lon),
                out[1].min(lat),
                out[2].max(lon),
                out[3].max(lat),
            ];
        }
        out
    }

    /// the same ground on another crs's axes, plus `margin` cells all
    /// round: the two grids are rotated against each other, so this is
    /// what an item in the neighbouring zone covering the anchor looks like
    fn covering(&self, epsg: u16, margin: usize) -> UtmGrid {
        let to =
            projicio_core::Transform::new(&format!("EPSG:{}", self.epsg), &format!("EPSG:{epsg}"))
                .unwrap();
        let pts = to.convert_batch(&self.boundary()).unwrap();
        let min_x = pts.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let max_x = pts.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let min_y = pts.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let max_y = pts.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        let margin_m = margin as f64 * UTM_CELL;
        UtmGrid {
            epsg,
            origin_x: min_x - margin_m,
            origin_y: max_y + margin_m,
            cols: ((max_x - min_x) / UTM_CELL).ceil() as usize + 2 * margin,
            rows: ((max_y - min_y) / UTM_CELL).ceil() as usize + 2 * margin,
        }
    }

    /// cog over the shared lon/lat scene on this grid, `shift` marking the
    /// values so tests can tell which item won a pixel, `hole` blanking
    /// rows 100..120 to nodata like `cog`
    fn cog(&self, shift: f64, hole: bool) -> Vec<u8> {
        let mut centers = Vec::with_capacity(self.cols * self.rows);
        for row in 0..self.rows {
            for col in 0..self.cols {
                centers.push(self.pixel_center(col, row));
            }
        }
        let data = to_lonlat(self.epsg)
            .convert_batch(&centers)
            .unwrap()
            .iter()
            .enumerate()
            .map(
                |(i, &(lon, lat))| match hole && (100..120).contains(&(i / self.cols)) {
                    true => f64::NAN,
                    false => elevation(lon, lat) + shift,
                },
            )
            .collect();
        let raster = Raster::from_vec(self.cols, self.rows, data, UTM_CELL, f64::NAN).unwrap();
        let params = CogParams {
            epsg: self.epsg,
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            pixel_width: UTM_CELL,
            pixel_height: UTM_CELL,
            ..params(0)
        };
        let mut buf = std::io::Cursor::new(Vec::new());
        write_cog(&raster, &params, &mut buf).unwrap();
        buf.into_inner()
    }

    /// window over these cells of the grid, aligned to it the way the
    /// engine's chunk windows are
    fn window(&self, col0: usize, row0: usize, cols: usize, rows: usize) -> WindowReq {
        WindowReq {
            bbox: Bbox {
                min_x: self.origin_x + col0 as f64 * UTM_CELL,
                max_x: self.origin_x + (col0 + cols) as f64 * UTM_CELL,
                max_y: self.origin_y - row0 as f64 * UTM_CELL,
                min_y: self.origin_y - (row0 + rows) as f64 * UTM_CELL,
            },
            resolution: UTM_CELL,
            time: None,
        }
    }

    fn row_at(&self, y: f64) -> usize {
        ((self.origin_y - y) / UTM_CELL - 0.5).round() as usize
    }
}

/// zone 32 and zone 33 items over the same ground: the newest item anchors
/// the grid on zone 32 and holes out rows 100..120, the zone 33 item covers
/// all of it plus a margin, shifted so a warped pixel is recognisable
async fn start_mixed_crs_mock() -> (String, UtmGrid) {
    let anchor = UtmGrid::at(ZONE_32, MIXED_LON, MIXED_LAT, MIXED_COLS, MIXED_ROWS);
    let other = anchor.covering(ZONE_33, MIXED_MARGIN);
    let (base, _mock) = start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        cogs.insert("zone32.tif".into(), anchor.cog(0.0, true));
        cogs.insert("zone33.tif".into(), other.cog(MIXED_SHIFT, false));
        let features = vec![
            item(
                "zone32",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/zone32.tif"),
                anchor.footprint(),
                u32::from(ZONE_32),
            ),
            item(
                "zone33",
                "2020-01-01T00:00:00Z",
                &format!("{base}/cog/zone33.tif"),
                other.footprint(),
                u32::from(ZONE_33),
            ),
        ];
        (cogs, features)
    })
    .await;
    (base, anchor)
}

/// covers both footprints, so the open search finds the whole scene
const MIXED_SEARCH_BBOX: [f64; 4] = [11.8, 46.85, 12.15, 47.05];

async fn open_mixed(base: &str) -> StacSrc {
    let search = StacSearch::new(base, "test-s2", "data", MIXED_SEARCH_BBOX);
    tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap()
}

fn close_warped(got: f64, want: f64, at: (f64, f64)) {
    assert!(
        (got - want).abs() < WARP_TOLERANCE,
        "({:.1},{:.1}): {got} vs {want}",
        at.0,
        at.1
    );
}

/// an item on another crs is kept and warped onto the anchor grid, so it
/// fills the anchor-crs item's nodata rows instead of leaving them empty
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_item_in_another_crs_fills_the_anchor_items_nodata() {
    let (base, anchor) = start_mixed_crs_mock().await;
    let src = open_mixed(&base).await;
    assert_eq!(src.item_count(), 2, "the zone 33 item was dropped");
    assert_eq!(src.crs(), Crs(u32::from(ZONE_32)));

    // a hundred cells square over the holed rows, well inside both items
    let req = anchor.window(100, 50, 100, 100);
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    let lonlat = to_lonlat(ZONE_32);
    let (mut plain, mut holed) = (0, 0);
    each_pixel(&chunk, |_, x, y, got| {
        let (lon, lat) = lonlat.convert(x, y).unwrap();
        let in_hole = (100..120).contains(&anchor.row_at(y));
        match in_hole {
            true => holed += 1,
            false => plain += 1,
        }
        let shift = if in_hole { MIXED_SHIFT } else { 0.0 };
        close_warped(got, elevation(lon, lat) + shift, (x, y));
    });
    assert_eq!((plain, holed), (8000, 2000), "window missed a case");
}

/// a window straddling the two items' coverage composites both: the zone
/// 32 item wins its own half, the zone 33 item covers the ground past the
/// zone 32 item's east edge
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_window_straddling_two_crss_composites_both() {
    let (base, anchor) = start_mixed_crs_mock().await;
    let src = open_mixed(&base).await;

    // fifty cells either side of the zone 32 item's east edge, clear of
    // its holed rows
    let req = anchor.window(MIXED_COLS - 50, 150, 100, 100);
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    let east_edge = anchor.origin_x + MIXED_COLS as f64 * UTM_CELL;
    let lonlat = to_lonlat(ZONE_32);
    let (mut zone32, mut zone33) = (0, 0);
    each_pixel(&chunk, |_, x, y, got| {
        let (lon, lat) = lonlat.convert(x, y).unwrap();
        let shift = match x < east_edge {
            true => {
                zone32 += 1;
                0.0
            }
            false => {
                zone33 += 1;
                MIXED_SHIFT
            }
        };
        close_warped(got, elevation(lon, lat) + shift, (x, y));
    });
    assert_eq!((zone32, zone33), (5000, 5000), "window missed a case");
}

/// the easting two opposite corners of a lon/lat footprint reach in the
/// anchor crs, the way item footprints once converted
fn two_corner_max_x(footprint: [f64; 4]) -> f64 {
    let to_native = from_lonlat(ZONE_32);
    let (west, _) = to_native.convert(footprint[0], footprint[1]).unwrap();
    let (east, _) = to_native.convert(footprint[2], footprint[3]).unwrap();
    west.max(east)
}

/// the zone 33 item's footprint comes out rotated in the anchor crs, so
/// its far corner reaches east of what two opposite corners of its
/// lon/lat bbox convert to. a window on that ground is covered by the
/// item and used to be skipped for it, coming back empty
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_window_past_the_two_corner_footprint_still_finds_the_item() {
    let (base, anchor) = start_mixed_crs_mock().await;
    let src = open_mixed(&base).await;

    let other = anchor.covering(ZONE_33, MIXED_MARGIN);
    let first_col = (two_corner_max_x(other.footprint()) - anchor.origin_x) / UTM_CELL;
    // the first anchor column past that easting, over rows where the
    // item's rotated corner still has six cells of cover to spare
    let req = anchor.window(first_col.ceil() as usize, 373, 10, 64);
    assert!(
        req.bbox.min_x > anchor.origin_x + MIXED_COLS as f64 * UTM_CELL,
        "the window is not clear of the zone 32 item"
    );

    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    let lonlat = to_lonlat(ZONE_32);
    let mut seen = 0;
    each_pixel(&chunk, |_, x, y, got| {
        let (lon, lat) = lonlat.convert(x, y).unwrap();
        seen += 1;
        close_warped(got, elevation(lon, lat) + MIXED_SHIFT, (x, y));
    });
    assert_eq!(seen, 10 * 64);
}

/// the two-degree search block boundary the tilted window reaches over
const BLOCK_EDGE_LAT: f64 = 46.0;

/// the window reaching over it: wide enough that the anchor grid's
/// rotation against lon/lat carries its south-east corner past the
/// boundary, tall enough that both corners the conversion once used stay
/// north of it
const BLOCK_EDGE_COLS: usize = 1800;
const BLOCK_EDGE_ROWS: usize = 70;

/// the anchor item sits this many cells north of the window, far enough
/// that its own envelope does not reach it: the window's only cover is
/// the item past the boundary
const BLOCK_EDGE_ANCHOR_ROWS: usize = 40;
const BLOCK_EDGE_GAP_ROWS: usize = 150;

/// the zone 33 item covering the corner past the boundary, and how far
/// south of the boundary its northernmost corner sits, so its whole
/// footprint falls in the block past it
const BLOCK_EDGE_COVER_COLS: usize = 800;
const BLOCK_EDGE_COVER_ROWS: usize = 100;
const BLOCK_EDGE_COVER_CLEARANCE: f64 = 0.002;

/// cells of the cover item a checked pixel keeps clear of its edge, past
/// which a warped sample falls back to the nearest cell it has
const BLOCK_EDGE_INSET: f64 = 2.0;

/// pixels of the window the cover item reaches once that inset is taken
/// off, well under the near three thousand it actually fills
const BLOCK_EDGE_MIN_COVERED: usize = 1000;

/// the anchor grid whose south-east corner tilts past `BLOCK_EDGE_LAT`:
/// the window's south edge crosses the boundary at its midpoint, leaving
/// half the tilt on either side
fn block_edge_anchor() -> UtmGrid {
    let (x, y) = from_lonlat(ZONE_32)
        .convert(MIXED_LON, BLOCK_EDGE_LAT)
        .unwrap();
    let to_south_edge = BLOCK_EDGE_ANCHOR_ROWS + BLOCK_EDGE_GAP_ROWS + BLOCK_EDGE_ROWS;
    UtmGrid {
        epsg: ZONE_32,
        origin_x: x - BLOCK_EDGE_COLS as f64 / 2.0 * UTM_CELL,
        origin_y: y + to_south_edge as f64 * UTM_CELL,
        cols: BLOCK_EDGE_COLS,
        rows: BLOCK_EDGE_ANCHOR_ROWS,
    }
}

/// the zone 33 item covering the window's corner past the boundary. its
/// north edge climbs eastward where the window's south edge falls, so
/// they cross and leave a wedge of shared ground
fn block_edge_cover(req: &WindowReq) -> UtmGrid {
    let (lon, _) = to_lonlat(ZONE_32)
        .convert(req.bbox.max_x, req.bbox.min_y)
        .unwrap();
    let (x, y) = from_lonlat(ZONE_33)
        .convert(lon, BLOCK_EDGE_LAT - BLOCK_EDGE_COVER_CLEARANCE)
        .unwrap();
    UtmGrid {
        epsg: ZONE_33,
        origin_x: x - BLOCK_EDGE_COVER_COLS as f64 * UTM_CELL,
        origin_y: y,
        cols: BLOCK_EDGE_COVER_COLS,
        rows: BLOCK_EDGE_COVER_ROWS,
    }
}

/// a window on the anchor grid is rotated against lon/lat too, so its
/// south-east corner sits in a search block that two opposite corners of
/// its bbox never name. the item covering that corner lives only in that
/// block, and used to be searched for only where the window was not
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_window_tilted_past_a_block_boundary_searches_that_block() {
    let anchor = block_edge_anchor();
    let req = anchor.window(
        0,
        BLOCK_EDGE_ANCHOR_ROWS + BLOCK_EDGE_GAP_ROWS,
        BLOCK_EDGE_COLS,
        BLOCK_EDGE_ROWS,
    );
    let cover = block_edge_cover(&req);

    let corners = to_lonlat(ZONE_32)
        .convert_batch(&[
            (req.bbox.min_x, req.bbox.min_y),
            (req.bbox.max_x, req.bbox.max_y),
            (req.bbox.max_x, req.bbox.min_y),
        ])
        .unwrap();
    let two_corner_min_lat = corners[0].1.min(corners[1].1);
    assert!(
        corners[2].1 < BLOCK_EDGE_LAT && two_corner_min_lat > BLOCK_EDGE_LAT,
        "the window does not straddle the boundary the way the case needs: \
         corner {}, two-corner {two_corner_min_lat}",
        corners[2].1
    );
    assert!(
        cover.footprint()[3] < BLOCK_EDGE_LAT && anchor.footprint()[1] > BLOCK_EDGE_LAT,
        "the two items do not sit on opposite sides of the boundary"
    );

    let (base, _mock) = start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        cogs.insert("edge_anchor.tif".into(), anchor.cog(0.0, false));
        cogs.insert("edge_cover.tif".into(), cover.cog(MIXED_SHIFT, false));
        let features = vec![
            item(
                "edge_anchor",
                "2024-06-01T00:00:00Z",
                &format!("{base}/cog/edge_anchor.tif"),
                anchor.footprint(),
                u32::from(ZONE_32),
            ),
            item(
                "edge_cover",
                "2020-01-01T00:00:00Z",
                &format!("{base}/cog/edge_cover.tif"),
                cover.footprint(),
                u32::from(ZONE_33),
            ),
        ];
        (cogs, features)
    })
    .await;

    // the open search covers the anchor item alone, so the cover item can
    // only arrive through the block search the window drives
    let search = StacSearch::new(&base, "test-s2", "data", anchor.footprint());
    let src = tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(
        src.item_count(),
        1,
        "the open search reached the cover item"
    );

    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    assert_eq!(
        src.item_count(),
        2,
        "the block past the boundary was never searched"
    );

    let lonlat = to_lonlat(ZONE_32);
    let to_cover = projicio_core::Transform::new("EPSG:32632", "EPSG:32633").unwrap();
    let mut covered = 0;
    each_pixel(&chunk, |_, x, y, got| {
        let (cx, cy) = to_cover.convert(x, y).unwrap();
        let col = (cx - cover.origin_x) / UTM_CELL;
        let row = (cover.origin_y - cy) / UTM_CELL;
        let inside = col > BLOCK_EDGE_INSET
            && row > BLOCK_EDGE_INSET
            && col < cover.cols as f64 - BLOCK_EDGE_INSET
            && row < cover.rows as f64 - BLOCK_EDGE_INSET;
        if !inside {
            return;
        }
        let (lon, lat) = lonlat.convert(x, y).unwrap();
        covered += 1;
        close_warped(got, elevation(lon, lat) + MIXED_SHIFT, (x, y));
    });
    assert!(
        covered > BLOCK_EDGE_MIN_COVERED,
        "only {covered} window pixels came back from the cover item"
    );
}

/// the engine splits a pull into chunks, so a mixed-crs item is read once
/// per chunk with its own planned window. those reads must land on the
/// same item pixels the one-shot read of the whole window lands on
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn chunked_pull_equals_the_whole_window_across_crss() {
    let (base, anchor) = start_mixed_crs_mock().await;
    // two chunks wide at the source's 256 px chunking, taking in the holed
    // rows the zone 33 item fills
    let req = anchor.window(0, 0, 512, 256);

    let whole = open_mixed(&base)
        .await
        .read(&req)
        .await
        .unwrap()
        .into_raster()
        .unwrap();

    let mut graph = Graph::new();
    let node = graph.add_source(Box::new(open_mixed(&base).await));
    let engine = Engine::new(graph, 64 << 20).unwrap();
    let chunked = engine.pull(node, req).await.unwrap().into_raster().unwrap();
    assert_eq!(chunked.bbox, whole.bbox);

    let lonlat = to_lonlat(ZONE_32);
    let mut holed = 0;
    each_pixel(&chunked, |_, x, y, got| {
        let (lon, lat) = lonlat.convert(x, y).unwrap();
        let in_hole = (100..120).contains(&anchor.row_at(y));
        holed += usize::from(in_hole);
        let shift = if in_hole { MIXED_SHIFT } else { 0.0 };
        close_warped(got, elevation(lon, lat) + shift, (x, y));
    });
    assert_eq!(holed, 20 * 512, "the pull missed the holed rows");

    let one = whole.bands.band(0).unwrap().data();
    let many = chunked.bands.band(0).unwrap().data();
    assert_eq!(one.len(), many.len());
    for (i, (a, b)) in one.iter().zip(many).enumerate() {
        assert!(
            (a - b).abs() < 1e-9,
            "pixel {i}: whole {a} vs chunked {b}, the two plans disagree"
        );
    }
}

/// the same zone pair with one cog per band, the sentinel-2 layout: both
/// assets of the zone 33 item are warped, so a window past the zone 32
/// item comes back with the bands still in asset order
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn per_asset_cogs_of_an_item_in_another_crs_stay_in_order() {
    let anchor = UtmGrid::at(ZONE_32, MIXED_LON, MIXED_LAT, MIXED_COLS, MIXED_ROWS);
    let other = anchor.covering(ZONE_33, MIXED_MARGIN);
    let (base, _mock) = start_mock_with(|base| {
        let mut cogs = std::collections::HashMap::new();
        cogs.insert("z32_red.tif".into(), anchor.cog(0.0, false));
        cogs.insert("z32_nir.tif".into(), anchor.cog(NIR_SHIFT, false));
        cogs.insert("z33_red.tif".into(), other.cog(MIXED_SHIFT, false));
        cogs.insert(
            "z33_nir.tif".into(),
            other.cog(NIR_SHIFT + MIXED_SHIFT, false),
        );
        let features = vec![
            asset_item(
                "z32",
                "2024-06-01T00:00:00Z",
                &[
                    ("red", format!("{base}/cog/z32_red.tif")),
                    ("nir", format!("{base}/cog/z32_nir.tif")),
                ],
                anchor.footprint(),
                u32::from(ZONE_32),
            ),
            asset_item(
                "z33",
                "2020-01-01T00:00:00Z",
                &[
                    ("red", format!("{base}/cog/z33_red.tif")),
                    ("nir", format!("{base}/cog/z33_nir.tif")),
                ],
                other.footprint(),
                u32::from(ZONE_33),
            ),
        ];
        (cogs, features)
    })
    .await;

    let mut search = StacSearch::new(&base, "test-s2", "red", MIXED_SEARCH_BBOX);
    search.assets = vec!["red".into(), "nir".into()];
    let src = tokio::task::spawn_blocking(move || StacSrc::open(&search))
        .await
        .unwrap()
        .unwrap();
    assert_eq!(src.item_count(), 2);
    assert_eq!(src.bands(), 2);

    // fifty cells past the zone 32 item, where only the warped item covers
    let req = anchor.window(MIXED_COLS, 150, 50, 50);
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    assert_eq!(chunk.bands.band_count(), 2);
    let lonlat = to_lonlat(ZONE_32);
    let mut seen = 0;
    each_pixel(&chunk, |bi, x, y, got| {
        let (lon, lat) = lonlat.convert(x, y).unwrap();
        let nir = if bi == 0 { 0.0 } else { NIR_SHIFT };
        seen += 1;
        close_warped(got, elevation(lon, lat) + MIXED_SHIFT + nir, (x, y));
    });
    assert_eq!(seen, 2 * 50 * 50);
}

fn population_std_dev(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / values.len() as f64).sqrt()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn std_dev_is_the_population_spread_of_the_stack() {
    // two shifts 900 apart spread 450 either side of their mean, which pins
    // the divisor at n rather than n-1
    assert_eq!(population_std_dev(&[100.0, 1000.0]), 450.0);

    let (base, _mock) = start_composite_mock().await;
    let src = open_composite(&base, Composite::StdDev).await;
    let chunk = src.read(&OVERLAP).await.unwrap().into_raster().unwrap();
    let (mut three_deep, mut two_deep) = (0, 0);
    each_pixel(&chunk, |_, x, y, got| {
        // the elevation is common to every item, so only the shifts spread
        if (100..120).contains(&scene_row(y)) {
            two_deep += 1;
            close(got, population_std_dev(&[100.0, 1000.0]), (x, y));
        } else {
            three_deep += 1;
            close(got, population_std_dev(&[0.0, 100.0, 1000.0]), (x, y));
        }
    });
    assert!(three_deep > 1000 && two_deep > 1000, "window missed a case");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn count_reports_how_many_items_had_a_value() {
    let (base, _mock) = start_mock().await;
    let src = open_composite(&base, Composite::Count).await;
    let chunk = src.read(&OVERLAP).await.unwrap().into_raster().unwrap();
    let (mut both, mut old_only) = (0, 0);
    each_pixel(&chunk, |_, _, y, got| {
        // left and old both cover this window, left holing out rows 100..120
        if (100..120).contains(&scene_row(y)) {
            old_only += 1;
            assert_eq!(got, 1.0);
        } else {
            both += 1;
            assert_eq!(got, 2.0);
        }
    });
    assert!(both > 1000 && old_only > 1000, "window missed a case");

    // past the east edge of every item nothing is counted, the pixel is
    // nodata rather than zero
    let req = WindowReq {
        bbox: Bbox {
            min_x: 7.5,
            max_x: 7.7,
            ..OVERLAP.bbox
        },
        ..OVERLAP
    };
    let chunk = src.read(&req).await.unwrap().into_raster().unwrap();
    let mut bare = 0;
    each_pixel(&chunk, |_, x, _, got| {
        if x < 7.6 {
            assert_eq!(got, 2.0);
        } else {
            bare += 1;
            assert!(got.is_nan(), "({x:.4}) past every item: {got}");
        }
    });
    assert!(bare > 1000, "window missed the uncovered half");
}
