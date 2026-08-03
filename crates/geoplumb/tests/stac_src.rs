//! stac source against a mock api: search, crs filtering, lazy cog opens
//! over range requests, and most-recent-first mosaicking with deflate cogs

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{HeaderMap, StatusCode, header};
use axum::routing::get;
use geoplumb::element::Source;
use geoplumb::elements::{StacSearch, StacSrc, stac::s3_to_https};
use geoplumb::{Bbox, Crs, Engine, Graph, WindowReq};
use terrano_core::{CogParams, Raster, write_cog};

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
    let params = CogParams {
        tile_width: 64,
        tile_height: 64,
        overview_levels: 2,
        epsg: 4326,
        origin_x: ORIGIN_X + x0 as f64 * CELL,
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
    search: serde_json::Value,
}

fn item(id: &str, dt: &str, href: &str, bbox: [f64; 4], epsg: u32) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "bbox": bbox,
        "properties": { "datetime": dt, "proj:epsg": epsg },
        "assets": { "data": { "href": href } }
    })
}

async fn serve_search(
    State(mock): State<Arc<Mock>>,
) -> ([(&'static str, &'static str); 1], String) {
    (
        [("content-type", "application/geo+json")],
        mock.search.to_string(),
    )
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

async fn start_mock() -> (String, Arc<Mock>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let base = format!("http://{addr}");
    let mut cogs = std::collections::HashMap::new();
    // the 2024 pair covers lon 7.0..7.3 (with a nodata hole) and 7.3..7.6
    cogs.insert("left.tif".into(), cog(0, 300, 400, 0.0, true));
    cogs.insert("right.tif".into(), cog(300, 300, 400, 0.0, false));
    // the 2020 item covers everything, values shifted so wins are visible
    cogs.insert("old.tif".into(), cog(0, 600, 400, 1000.0, false));
    let search = serde_json::json!({ "type": "FeatureCollection", "features": [
        item("old", "2020-01-01T00:00:00Z", &format!("{base}/cog/old.tif"), [7.0, 46.6, 7.6, 47.0], 4326),
        item("left", "2024-06-01T00:00:00Z", &format!("{base}/cog/left.tif"), [7.0, 46.6, 7.3, 47.0], 4326),
        item("utm", "2019-01-01T00:00:00Z", &format!("{base}/cog/missing.tif"), [7.0, 46.6, 7.6, 47.0], 32632),
        item("right", "2024-06-01T00:00:00Z", &format!("{base}/cog/right.tif"), [7.3, 46.6, 7.6, 47.0], 4326),
    ]});
    let mock = Arc::new(Mock { cogs, search });
    let app = axum::Router::new()
        .route("/search", get(serve_search))
        .route("/cog/{name}", get(serve_cog))
        .with_state(mock.clone());
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (base, mock)
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn recent_items_win_and_old_fills_their_holes() {
    let (base, _mock) = start_mock().await;
    let src = open_src(&base).await;
    let mut g = Graph::new();
    let n = g.add_source(Box::new(src));
    let engine = Engine::new(g, 64 << 20).unwrap();

    // spans the left/right item seam at lon 7.3 and the left item's hole
    let req = WindowReq {
        bbox: Bbox {
            min_x: 7.2,
            max_x: 7.4,
            max_y: ORIGIN_Y - 0.05,
            min_y: ORIGIN_Y - 0.15,
        },
        resolution: CELL,
    };
    let chunk = engine.pull(n, req).await.unwrap();
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

#[test]
fn s3_hrefs_rewrite_to_https() {
    assert_eq!(
        s3_to_https("s3://copernicus-dem-30m/some/key.tif"),
        "https://copernicus-dem-30m.s3.amazonaws.com/some/key.tif"
    );
    assert_eq!(s3_to_https("https://x/y.tif"), "https://x/y.tif");
}
