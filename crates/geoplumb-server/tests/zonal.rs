//! the zonal endpoints over a cog layer written at test time: reductions
//! whose numbers are checkable by hand, request-order ids, and a rejection
//! for every cap and every malformed-input class. nothing touches the
//! network, and every zone is given in lon/lat the way geojson carries it

use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use geoplumb_server::config::Config;
use geoplumb_server::{Layer, router};
use serde_json::{Value, json};
use terrano_core::{CogParams, Raster, write_cog};
use tower::ServiceExt;

const WIDTH: usize = 600;
const HEIGHT: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

/// the top rows hold nodata, so a zone can reduce to nothing at all
const NODATA_ROWS: usize = 100;
/// columns left of this hold `WEST`, the rest hold `EAST`
const SEAM_COL: usize = 300;
const WEST: f64 = 10.0;
const EAST: f64 = 30.0;

/// ground metres, coarse enough that a whole-raster window is a few chunks
const RESOLUTION: f64 = 500.0;

const STEP: &str = "2024-06-01T00:00:00Z/2024-07-01T00:00:00Z";
const NEXT_STEP: &str = "2024-07-01T00:00:00Z/2024-08-01T00:00:00Z";

/// the caps the endpoints refuse past, restated so a test can sit on the
/// boundary of each one
const FEATURES_CAP: usize = 256;
const STEPS_CAP: usize = 64;
const POSITIONS_CAP: usize = 20_000;
const REDUCTION_SLOTS: usize = 4;

const WEB_MERCATOR_EXTENT: f64 = 20037508.342789244;

fn web_mercator(lon: f64, lat: f64) -> (f64, f64) {
    let x = lon * WEB_MERCATOR_EXTENT / 180.0;
    let y =
        ((90.0 + lat).to_radians() / 2.0).tan().ln() * WEB_MERCATOR_EXTENT / std::f64::consts::PI;
    (x, y)
}

/// a dem of two flat halves under a nodata strip, as a deflate cog. flat
/// halves survive reprojection, so a zone inside one of them still reduces
/// to exactly that half's value
fn split_cog() -> Vec<u8> {
    let mut data = Vec::with_capacity(WIDTH * HEIGHT);
    for row in 0..HEIGHT {
        for col in 0..WIDTH {
            data.push(match (row < NODATA_ROWS, col < SEAM_COL) {
                (true, _) => f64::NAN,
                (false, true) => WEST,
                (false, false) => EAST,
            });
        }
    }
    let raster = Raster::from_vec(WIDTH, HEIGHT, data, CELL, f64::NAN).unwrap();
    let params = CogParams {
        tile_width: 256,
        tile_height: 256,
        overview_levels: 2,
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

/// a one-layer file with no ops, so the served values are the cog's own
fn split_layer_file(dir: &Path) -> String {
    let path = dir.join("split.tif");
    std::fs::write(&path, split_cog()).unwrap();
    format!(
        r#"
[[layer]]
name = "dem"
source = {{ kind = "cog", path = "{}" }}
"#,
        path.display().to_string().replace('\\', "/")
    )
}

fn app(text: &str) -> Router {
    let config = Config::parse(text).expect("the layer file must parse");
    let layers = Layer::build_all(&config, 64 << 20, None).expect("the layers must build");
    router(layers)
}

/// a rectangular zone in lon/lat degrees
fn zone(west: f64, south: f64, east: f64, north: f64) -> Value {
    let ring: Vec<Value> = [
        (west, south),
        (east, south),
        (east, north),
        (west, north),
        (west, south),
    ]
    .iter()
    .map(|(lon, lat)| json!([lon, lat]))
    .collect();
    json!({
        "type": "Feature",
        "properties": {},
        "geometry": { "type": "Polygon", "coordinates": [ring] }
    })
}

/// the same rectangle with its southern edge subdivided, so one zone gives
/// the burn that many edges to walk per raster row
fn dense_zone(west: f64, south: f64, east: f64, north: f64, positions: usize) -> Value {
    let step = (east - west) / positions as f64;
    let mut ring: Vec<Value> = (0..positions)
        .map(|index| json!([west + index as f64 * step, south]))
        .collect();
    ring.push(json!([east, south]));
    ring.push(json!([east, north]));
    ring.push(json!([west, north]));
    ring.push(json!([west, south]));
    json!({
        "type": "Feature",
        "properties": {},
        "geometry": { "type": "Polygon", "coordinates": [ring] }
    })
}

/// the same rectangle as a request bbox, the geojson corner order
fn window(west: f64, south: f64, east: f64, north: f64) -> [f64; 4] {
    [west, south, east, north]
}

async fn post(app: &Router, uri: &str, body: Value) -> (StatusCode, Vec<u8>) {
    post_raw(app, uri, serde_json::to_vec(&body).unwrap()).await
}

async fn post_raw(app: &Router, uri: &str, body: Vec<u8>) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(uri)
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(body))
                .unwrap(),
        )
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 8 << 20)
        .await
        .unwrap();
    (status, body.to_vec())
}

fn rows(body: &[u8]) -> Vec<Value> {
    let parsed: Value = serde_json::from_slice(body).expect("a json response");
    parsed["rows"].as_array().expect("a rows array").clone()
}

fn count_of(row: &Value) -> u64 {
    row["count"].as_u64().expect("a count")
}

#[tokio::test]
async fn a_zonal_reduction_holds_one_row_per_request_index() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    let west = zone(7.05, 46.65, 7.25, 46.85);
    let east = zone(7.35, 46.65, 7.55, 46.85);
    // straddles the seam without touching the other two: zones that overlap
    // share their pixels out to whichever came last, so a zone containing
    // another would leave that one empty
    let seam = zone(7.26, 46.65, 7.34, 46.85);

    let (status, body) = post(
        &app,
        "/zonal/dem",
        json!({
            "type": "FeatureCollection",
            "features": [west, east, seam],
            "resolution": RESOLUTION,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let rows = rows(&body);
    assert_eq!(rows.len(), 3);
    assert_eq!(
        rows.iter()
            .map(|row| row["id"].as_u64())
            .collect::<Vec<_>>(),
        vec![Some(0), Some(1), Some(2)]
    );

    for (index, value) in [(0, WEST), (1, EAST)] {
        let row = &rows[index];
        let count = count_of(row);
        assert!(count > 0, "zone {index} caught no pixel");
        assert_eq!(row["minimum"].as_f64(), Some(value), "zone {index}");
        assert_eq!(row["maximum"].as_f64(), Some(value), "zone {index}");
        assert_eq!(row["mean"].as_f64(), Some(value), "zone {index}");
        assert_eq!(
            row["sum"].as_f64(),
            Some(count as f64 * value),
            "zone {index}"
        );
    }

    assert_eq!(rows[2]["minimum"].as_f64(), Some(WEST));
    assert_eq!(rows[2]["maximum"].as_f64(), Some(EAST));
    let mean = rows[2]["mean"].as_f64().expect("a mean");
    assert!(mean > WEST && mean < EAST, "the seam zone means {mean}");
    // the seam zone is the narrow one, so a shuffled id would show up here
    assert!(count_of(&rows[2]) < count_of(&rows[0]));
}

#[tokio::test]
async fn a_zone_that_caught_no_pixel_serializes_null_not_nan() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    // inside the nodata strip, so every pixel of it drops out
    let empty = zone(7.1, 46.93, 7.5, 46.97);

    let (status, body) = post(
        &app,
        "/zonal/dem",
        json!({ "features": [empty], "resolution": RESOLUTION }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let rows = rows(&body);
    assert_eq!(rows.len(), 1);
    assert_eq!(count_of(&rows[0]), 0);
    assert!(rows[0]["mean"].is_null(), "{}", rows[0]);
    assert!(rows[0]["minimum"].is_null(), "{}", rows[0]);
    assert!(rows[0]["maximum"].is_null(), "{}", rows[0]);
    assert_eq!(rows[0]["sum"].as_f64(), Some(0.0));

    let text = String::from_utf8(body).unwrap();
    assert!(
        !text.to_lowercase().contains("nan"),
        "the response carries a spelling json cannot read back: {text}"
    );
}

#[tokio::test]
async fn a_reduction_with_no_zones_covers_the_whole_bbox() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    let inside_west = window(7.05, 46.65, 7.25, 46.85);

    for features in [json!(null), json!([])] {
        let mut body = json!({ "bbox": inside_west, "resolution": RESOLUTION });
        if !features.is_null() {
            body["features"] = features.clone();
        }
        let (status, body) = post(&app, "/zonal/dem", body).await;
        assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

        let rows = rows(&body);
        assert_eq!(rows.len(), 1, "features {features}");
        assert!(rows[0]["id"].is_null(), "a whole-window row has no id");
        let count = count_of(&rows[0]);
        assert!(count > 0, "the window caught no pixel");
        assert_eq!(rows[0]["minimum"].as_f64(), Some(WEST));
        assert_eq!(rows[0]["maximum"].as_f64(), Some(WEST));
        assert_eq!(rows[0]["mean"].as_f64(), Some(WEST));
        assert_eq!(rows[0]["sum"].as_f64(), Some(count as f64 * WEST));
    }
}

#[tokio::test]
async fn a_series_returns_one_entry_per_step_in_order() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));

    let (status, body) = post(
        &app,
        "/zonal/dem/series",
        json!({
            "features": [zone(7.05, 46.65, 7.25, 46.85)],
            "resolution": RESOLUTION,
            "steps": [STEP, NEXT_STEP],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let parsed: Value = serde_json::from_slice(&body).unwrap();
    let steps = parsed["steps"].as_array().expect("a steps array");
    assert_eq!(
        steps
            .iter()
            .map(|step| step["t"].as_str())
            .collect::<Vec<_>>(),
        vec![Some(STEP), Some(NEXT_STEP)]
    );
    // the cog has no time axis, so every step reduces the same pixels
    for step in steps {
        let rows = step["rows"].as_array().expect("a rows array");
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["id"].as_u64(), Some(0));
        assert!(count_of(&rows[0]) > 0);
        assert_eq!(rows[0]["mean"].as_f64(), Some(WEST));
    }
}

#[tokio::test]
async fn a_whole_window_series_needs_no_zones() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));

    let (status, body) = post(
        &app,
        "/zonal/dem/series",
        json!({
            "bbox": window(7.35, 46.65, 7.55, 46.85),
            "resolution": RESOLUTION,
            "steps": [STEP, NEXT_STEP],
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let parsed: Value = serde_json::from_slice(&body).unwrap();
    let steps = parsed["steps"].as_array().expect("a steps array");
    assert_eq!(steps.len(), 2);
    for step in steps {
        let rows = step["rows"].as_array().expect("a rows array");
        assert_eq!(rows.len(), 1);
        assert!(rows[0]["id"].is_null());
        assert_eq!(rows[0]["mean"].as_f64(), Some(EAST));
    }
}

#[tokio::test]
async fn a_lon_lat_body_reduces_the_place_it_names() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));

    // the same rectangle as a zone and as a bbox, which have to land on the
    // same half of the dem: a projection error would put one of them off it
    let (status, body) = post(
        &app,
        "/zonal/dem",
        json!({ "features": [zone(7.05, 46.65, 7.25, 46.85)], "resolution": RESOLUTION }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let burned = rows(&body)[0].clone();

    let (status, body) = post(
        &app,
        "/zonal/dem",
        json!({ "bbox": window(7.05, 46.65, 7.25, 46.85), "resolution": RESOLUTION }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let whole = rows(&body)[0].clone();

    assert_eq!(burned["mean"].as_f64(), Some(WEST));
    assert_eq!(whole["mean"].as_f64(), Some(WEST));
    // the burn takes the pixels centred inside the rectangle and the window
    // takes the rectangle snapped outward, so one is a hair short of the other
    let ratio = count_of(&burned) as f64 / count_of(&whole) as f64;
    assert!(
        ratio > 0.85 && ratio <= 1.0,
        "the zone and the bbox landed differently: {ratio}"
    );

    // web mercator metres are what the layer grid speaks, not what a body does
    let (x, y) = web_mercator(7.15, 46.75);
    let (status, message) = post(
        &app,
        "/zonal/dem",
        json!({
            "features": [{
                "type": "Feature",
                "properties": {},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x, y], [x + 1000.0, y], [x + 1000.0, y + 1000.0], [x, y]]],
                },
            }],
            "resolution": RESOLUTION,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = String::from_utf8(message).unwrap();
    assert!(message.contains("longitude"), "{message}");
}

#[tokio::test]
async fn a_request_past_a_cap_is_refused_naming_it() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    let one = zone(7.05, 46.65, 7.25, 46.85);
    let inside_west = window(7.05, 46.65, 7.25, 46.85);
    let whole_world = window(-180.0, -85.0, 180.0, 85.0);
    let long_ring: Vec<Value> = (0..POSITIONS_CAP + 2)
        .map(|step| json!([7.05 + (step % 128) as f64 * 1e-4, 46.7]))
        .collect();

    let cases = [
        (
            "the pixel budget",
            "/zonal/dem",
            json!({ "bbox": whole_world, "resolution": RESOLUTION }),
            "pixels",
        ),
        (
            "the feature cap",
            "/zonal/dem",
            json!({
                "features": vec![one.clone(); FEATURES_CAP + 1],
                "resolution": RESOLUTION,
            }),
            "features is past",
        ),
        (
            "the position cap",
            "/zonal/dem",
            json!({
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": { "type": "Polygon", "coordinates": [long_ring] },
                }],
                "resolution": RESOLUTION,
            }),
            "position cap",
        ),
        (
            "the step cap",
            "/zonal/dem/series",
            json!({
                "bbox": inside_west,
                "resolution": RESOLUTION,
                "steps": vec![STEP; STEPS_CAP + 1],
            }),
            "steps is past",
        ),
        (
            "the pixel budget over a series",
            "/zonal/dem/series",
            json!({
                "bbox": whole_world,
                "resolution": RESOLUTION,
                "steps": [STEP, NEXT_STEP],
            }),
            "pixels",
        ),
    ];

    for (what, uri, body, reason) in cases {
        let (status, body) = post(&app, uri, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{what}");
        let message = String::from_utf8(body).unwrap();
        assert!(message.contains(reason), "{what} answered {message}");
    }
}

#[tokio::test]
async fn the_pixel_budget_counts_every_step_of_a_series() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    // a window one pull can afford and a full series of pulls cannot
    let wide = window(7.0, 45.0, 11.0, 47.7);

    let (status, body) = post(
        &app,
        "/zonal/dem/series",
        json!({ "bbox": wide, "resolution": RESOLUTION, "steps": [STEP] }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));

    let (status, body) = post(
        &app,
        "/zonal/dem/series",
        json!({ "bbox": wide, "resolution": RESOLUTION, "steps": vec![STEP; STEPS_CAP] }),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = String::from_utf8(body).unwrap();
    assert!(message.contains("pixels"), "{message}");
}

#[tokio::test]
async fn malformed_zonal_bodies_are_refused_before_any_pull() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    let one = zone(7.05, 46.65, 7.25, 46.85);
    let inside_west = window(7.05, 46.65, 7.25, 46.85);
    let ring: Vec<Value> = (0..4).map(|_| json!([0.0, 0.0])).collect();

    let bodies = [
        (
            "a resolution of zero",
            "/zonal/dem",
            json!({ "features": [one.clone()], "resolution": 0.0 }),
        ),
        (
            "a negative resolution",
            "/zonal/dem",
            json!({ "features": [one.clone()], "resolution": -30.0 }),
        ),
        (
            "no resolution at all",
            "/zonal/dem",
            json!({ "features": [one.clone()] }),
        ),
        (
            "an unknown field",
            "/zonal/dem",
            json!({
                "bbox": inside_west,
                "resolution": RESOLUTION,
                "statistic": "mean",
            }),
        ),
        (
            "a collection that is not one",
            "/zonal/dem",
            json!({ "type": "Feature", "bbox": inside_west, "resolution": RESOLUTION }),
        ),
        (
            "no bbox behind no features",
            "/zonal/dem",
            json!({ "resolution": RESOLUTION }),
        ),
        (
            "no bbox behind an empty features array",
            "/zonal/dem",
            json!({ "features": [], "resolution": RESOLUTION }),
        ),
        (
            "an inverted bbox",
            "/zonal/dem",
            json!({ "bbox": [1.0, 1.0, 0.0, 0.0], "resolution": RESOLUTION }),
        ),
        (
            "a bbox of three numbers",
            "/zonal/dem",
            json!({ "bbox": [0.0, 0.0, 1.0], "resolution": RESOLUTION }),
        ),
        (
            "a t the parser refuses",
            "/zonal/dem",
            json!({ "features": [one.clone()], "resolution": RESOLUTION, "t": "last summer" }),
        ),
        (
            "a member that is not a Feature",
            "/zonal/dem",
            json!({
                "features": [{ "type": "Polygon", "coordinates": [ring.clone()] }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a null geometry",
            "/zonal/dem",
            json!({
                "features": [{ "type": "Feature", "properties": {}, "geometry": null }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a point zone",
            "/zonal/dem",
            json!({
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": { "type": "Point", "coordinates": [0.0, 0.0] },
                }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a line zone",
            "/zonal/dem",
            json!({
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": { "type": "LineString", "coordinates": [[0.0, 0.0], [1.0, 1.0]] },
                }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a geometry with no coordinates",
            "/zonal/dem",
            json!({
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": { "type": "Polygon" },
                }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a ring that cannot close",
            "/zonal/dem",
            json!({
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]],
                    },
                }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a position of one number",
            "/zonal/dem",
            json!({
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [[[0.0, 0.0], [1.0], [1.0, 1.0], [0.0, 0.0]]],
                    },
                }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a multipolygon of no polygons",
            "/zonal/dem",
            json!({
                "features": [{
                    "type": "Feature",
                    "properties": {},
                    "geometry": { "type": "MultiPolygon", "coordinates": [] },
                }],
                "resolution": RESOLUTION,
            }),
        ),
        (
            "a latitude past the web mercator limit",
            "/zonal/dem",
            json!({ "features": [zone(7.05, 85.1, 7.25, 85.4)], "resolution": RESOLUTION }),
        ),
        (
            "a bbox latitude past the web mercator limit",
            "/zonal/dem",
            json!({ "bbox": window(7.05, 46.65, 7.25, 88.0), "resolution": RESOLUTION }),
        ),
        (
            "a longitude off the earth",
            "/zonal/dem",
            json!({ "features": [zone(179.9, 46.65, 180.4, 46.85)], "resolution": RESOLUTION }),
        ),
        (
            "a bbox in the wrong units",
            "/zonal/dem",
            json!({ "bbox": [780000.0, 5900000.0, 800000.0, 5920000.0], "resolution": RESOLUTION }),
        ),
        (
            "a series of no steps",
            "/zonal/dem/series",
            json!({ "bbox": inside_west, "resolution": RESOLUTION, "steps": [] }),
        ),
        (
            "a step the parser refuses",
            "/zonal/dem/series",
            json!({ "bbox": inside_west, "resolution": RESOLUTION, "steps": ["last summer"] }),
        ),
        (
            "a t on a series",
            "/zonal/dem/series",
            json!({
                "bbox": inside_west,
                "resolution": RESOLUTION,
                "steps": [STEP],
                "t": STEP,
            }),
        ),
    ];

    for (what, uri, body) in bodies {
        let (status, message) = post(&app, uri, body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{what}");
        assert!(!message.is_empty(), "{what} gave no reason");
    }

    let raw = [
        ("a body that is not json", b"{not json".to_vec()),
        (
            "a resolution out of f64 range",
            br#"{"bbox":[0,0,1,1],"resolution":1e400}"#.to_vec(),
        ),
        (
            "a body nested past what serde_json will read",
            format!(
                r#"{{"resolution":500.0,"features":[{}{}]}}"#,
                "[".repeat(200),
                "]".repeat(200)
            )
            .into_bytes(),
        ),
    ];
    for (what, body) in raw {
        let (status, message) = post_raw(&app, "/zonal/dem", body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{what}");
        assert!(!message.is_empty(), "{what} gave no reason");
    }
}

#[tokio::test]
async fn a_body_past_the_buffered_limit_is_refused_unparsed() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    // axum buffers a body through a 2 MiB limit of its own, and the zonal
    // endpoints read theirs through that same extractor
    let padded = format!("{}{{}}", " ".repeat(3 << 20));

    let (status, _) = post_raw(&app, "/zonal/dem", padded.into_bytes()).await;
    assert_eq!(status, StatusCode::PAYLOAD_TOO_LARGE);
}

#[tokio::test]
async fn overlapping_zones_go_to_whichever_burned_last() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    // both wholly inside the west half, the second wholly inside the first
    let outer = zone(7.05, 46.65, 7.25, 46.85);
    let inner = zone(7.10, 46.70, 7.15, 46.75);

    let (status, body) = post(
        &app,
        "/zonal/dem",
        json!({
            "features": [inner.clone(), outer.clone()],
            "resolution": RESOLUTION,
        }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let contained_first = rows(&body);
    assert_eq!(
        count_of(&contained_first[0]),
        0,
        "the contained zone kept pixels the zone after it burned over"
    );
    assert!(
        contained_first[0]["mean"].is_null(),
        "{}",
        contained_first[0]
    );
    let whole = count_of(&contained_first[1]);
    assert!(whole > 0, "the containing zone caught no pixel");

    let (status, body) = post(
        &app,
        "/zonal/dem",
        json!({ "features": [outer, inner], "resolution": RESOLUTION }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let contained_last = rows(&body);
    let inner_count = count_of(&contained_last[1]);
    assert!(
        inner_count > 0,
        "the contained zone burned last caught nothing"
    );
    assert_eq!(
        count_of(&contained_last[0]) + inner_count,
        whole,
        "the overlap was counted twice or lost"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_burst_of_reductions_is_refused_past_the_slots_rather_than_queued() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    // a burn long enough that the first requests still hold their slots when
    // the rest of the burst arrives
    let body = json!({
        "features": [dense_zone(7.02, 46.62, 7.58, 46.88, POSITIONS_CAP - 1000)],
        "resolution": 1.0,
    });

    let mut burst = tokio::task::JoinSet::new();
    for _ in 0..REDUCTION_SLOTS * 4 {
        let app = app.clone();
        let body = body.clone();
        burst.spawn(async move { post(&app, "/zonal/dem", body).await });
    }

    let mut refused = 0;
    while let Some(answer) = burst.join_next().await {
        let (status, body) = answer.expect("a request task");
        match status {
            StatusCode::OK => assert_eq!(rows(&body).len(), 1),
            StatusCode::SERVICE_UNAVAILABLE => {
                refused += 1;
                let message = String::from_utf8(body).unwrap();
                assert!(message.contains("reduction slots are busy"), "{message}");
            }
            other => panic!(
                "the burst answered {other}: {}",
                String::from_utf8_lossy(&body)
            ),
        }
    }
    assert!(refused > 0, "the burst never found the slots full");

    // the slots come back, so a refused burst does not leave the endpoint dead
    let (status, body) = post(&app, "/zonal/dem", body).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
}

#[tokio::test]
async fn a_zonal_request_for_an_unknown_layer_is_a_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&split_layer_file(dir.path()));
    let body = json!({ "bbox": window(7.05, 46.65, 7.25, 46.85), "resolution": RESOLUTION });

    for uri in ["/zonal/nope", "/zonal/nope/series"] {
        let (status, _) = post(&app, uri, body.clone()).await;
        assert_eq!(status, StatusCode::NOT_FOUND, "{uri}");
    }
}
