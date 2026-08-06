//! config parsing, the `t` parameter, and the routes over a cog layer
//! built from a synthetic geotiff written at test time, so nothing here
//! touches the network

use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use geoplumb_server::config::{Config, OpConfig, SourceConfig};
use geoplumb_server::{Layer, parse_time, router};
use terrano_core::{CogParams, Raster, write_cog};
use tower::ServiceExt;

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

/// an alpine-ish dem as a deflate cog, the fixture a `cog` layer serves
fn dem_cog() -> Vec<u8> {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos());
        }
    }
    let raster = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
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

/// the xyz tile holding a lon/lat, the slippy-map formula
fn tile_of(z: u8, lon: f64, lat: f64) -> (u32, u32) {
    let n = f64::from(1u32 << z);
    let x = ((lon + 180.0) / 360.0 * n).floor() as u32;
    let r = lat.to_radians();
    let y =
        ((1.0 - (r.tan() + 1.0 / r.cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor() as u32;
    (x, y)
}

/// a one-layer file over a cog written into `dir`
fn cog_layer_file(dir: &Path) -> String {
    let path = dir.join("dem.tif");
    std::fs::write(&path, dem_cog()).unwrap();
    format!(
        r#"
[[layer]]
name = "dem"
source = {{ kind = "cog", path = "{}" }}

[[layer.op]]
kind = "hillshade"
azimuth = 315.0
altitude = 45.0
"#,
        path.display()
    )
}

fn app(text: &str) -> Router {
    let config = Config::parse(text).expect("the layer file must parse");
    let layers = Layer::build_all(&config, 64 << 20, None).expect("the layers must build");
    router(layers)
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let body = axum::body::to_bytes(response.into_body(), 8 << 20)
        .await
        .unwrap();
    (status, body.to_vec())
}

/// pixels the png marks opaque, so an empty tile is distinguishable from
/// one that actually rendered data
fn opaque_pixels(bytes: &[u8]) -> usize {
    let mut reader = png::Decoder::new(bytes).read_info().expect("a png");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::GrayscaleAlpha);
    buf[..info.buffer_size()]
        .chunks_exact(2)
        .filter(|px| px[1] == 255)
        .count()
}

const FULL: &str = r#"
[[layer]]
name = "ndvi"
source = { kind = "stac", api = "https://example.test/v1", collection = "sentinel-2-l2a", assets = ["red", "nir"], bbox = [7.0, 46.3, 8.0, 46.9], datetime = "2025-06-01T00:00:00Z/2025-09-01T00:00:00Z", composite = "median" }

[[layer.op]]
kind = "bandmath"
expr = "(b1 - b0) / (b1 + b0)"
min = -1.0
max = 1.0

[[layer]]
name = "dem"
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.op]]
kind = "hillshade"
azimuth = 315.0
altitude = 45.0
"#;

#[test]
fn config_parses_sources_ops_and_their_order() {
    let config = Config::parse(FULL).unwrap();
    assert_eq!(config.layers.len(), 2);

    let ndvi = &config.layers[0];
    assert_eq!(ndvi.name, "ndvi");
    let search = ndvi.source.stac_search().expect("a stac source");
    assert_eq!(search.collection, "sentinel-2-l2a");
    assert_eq!(search.assets, ["red", "nir"]);
    assert_eq!(
        search.datetime.as_deref(),
        Some("2025-06-01T00:00:00Z/2025-09-01T00:00:00Z")
    );
    assert_eq!(search.composite, geoplumb::elements::Composite::Median);
    assert_eq!(search.bbox, [7.0, 46.3, 8.0, 46.9]);
    match &ndvi.ops[..] {
        [OpConfig::Bandmath { expr, min, max }] => {
            assert_eq!(expr, "(b1 - b0) / (b1 + b0)");
            assert_eq!((*min, *max), (-1.0, 1.0));
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    let dem = &config.layers[1];
    assert!(dem.source.stac_search().is_none());
    match &dem.source {
        SourceConfig::Cog { path } => assert_eq!(path.to_str(), Some("/data/dem.tif")),
        other => panic!("unexpected source: {other:?}"),
    }
    assert!(matches!(dem.ops[..], [OpConfig::Hillshade { .. }]));
}

#[test]
fn config_rejects_files_it_cannot_serve() {
    let cases = [
        ("not toml at all {{{", "syntax"),
        ("", "no layers"),
        (
            r#"[[layer]]
name = "a"
source = { kind = "wms", url = "x" }"#,
            "unknown source kind",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "cog", path = "/a.tif" }
[[layer.op]]
kind = "slope""#,
            "op outside the two supported",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "cog", path = "/a.tif" }
[[layer.op]]
kind = "hillshade"
azimuth = 315.0"#,
            "op missing a field",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "cog", path = "/a.tif" }
[[layer]]
name = "a"
source = { kind = "cog", path = "/b.tif" }"#,
            "duplicate names",
        ),
        (
            r#"[[layer]]
name = "a/b"
source = { kind = "cog", path = "/a.tif" }"#,
            "name is not a path segment",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "stac", api = "x", collection = "c", assets = [], bbox = [0.0, 0.0, 1.0, 1.0] }"#,
            "stac source with no assets",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "cog", path = "/a.tif" }
colour = "blue""#,
            "unknown layer field",
        ),
    ];
    for (text, why) in cases {
        assert!(Config::parse(text).is_err(), "{why} should not have parsed");
    }
}

#[test]
fn the_t_parameter_is_an_rfc3339_interval_or_nothing() {
    assert_eq!(parse_time(None).unwrap(), None);
    let t = parse_time(Some("2024-06-01T00:00:00Z/2024-07-01T00:00:00Z"))
        .unwrap()
        .expect("an interval");
    assert_eq!(t.end_ms - t.start_ms, 30 * 86_400_000);
    assert!(parse_time(Some("2024-07-01T00:00:00Z/2024-06-01T00:00:00Z")).is_err());
    assert!(parse_time(Some("2024-06-01T00:00:00Z")).is_err());
    assert!(parse_time(Some("last summer")).is_err());
    assert!(parse_time(Some("")).is_err());
}

#[tokio::test]
async fn health_and_layers_describe_the_service() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&cog_layer_file(dir.path()));

    let (status, body) = get(&app, "/health").await;
    assert_eq!(status, StatusCode::OK);
    let health: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(health["status"], "ok");

    let (status, body) = get(&app, "/layers").await;
    assert_eq!(status, StatusCode::OK);
    let layers: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(layers.as_array().unwrap().len(), 1);
    assert_eq!(layers[0]["name"], "dem");
    assert_eq!(layers[0]["source"], "cog");
    // a cog layer has no collection and no time of its own, and the keys
    // are present so a client does not have to guess
    assert!(layers[0]["collection"].is_null());
    assert!(layers[0]["default_datetime"].is_null());
    assert!(layers[0]["temporal_extent"].is_null());
}

#[tokio::test]
async fn a_tile_renders_at_any_time_and_a_bad_one_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&cog_layer_file(dir.path()));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let (status, png) = get(&app, &format!("/tiles/dem/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
    let opaque = opaque_pixels(&png);
    assert!(opaque > 0, "the tile rendered nothing but nodata");

    // the layer's source has no time axis, so both instants serve the
    // same tile, and neither is an error
    for t in [
        "2024-06-01T00:00:00Z/2024-07-01T00:00:00Z",
        "2020-01-01T00:00:00Z/2021-01-01T00:00:00Z",
    ] {
        let (status, timed) = get(&app, &format!("/tiles/dem/12/{x}/{y}.png?t={t}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(opaque_pixels(&timed), opaque);
    }

    let (status, body) = get(&app, &format!("/tiles/dem/12/{x}/{y}.png?t=yesterday")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let message = String::from_utf8(body).unwrap();
    assert!(
        message.contains("t parameter"),
        "a rejection should name the parameter: {message}"
    );
}

#[tokio::test]
async fn malformed_tile_requests_are_rejected_not_rendered() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&cog_layer_file(dir.path()));

    let cases = [
        ("/tiles/nope/12/2131/1443.png", StatusCode::NOT_FOUND),
        ("/tiles/dem/12/2131/north.png", StatusCode::BAD_REQUEST),
        // past the zoom cap, where the tile arithmetic stops working
        ("/tiles/dem/40/0/0.png", StatusCode::BAD_REQUEST),
        // inside the cap but outside the zoom's tile grid
        ("/tiles/dem/2/9/0.png", StatusCode::BAD_REQUEST),
        ("/tiles/dem/zoom/0/0.png", StatusCode::BAD_REQUEST),
    ];
    for (uri, expected) in cases {
        let (status, _) = get(&app, uri).await;
        assert_eq!(status, expected, "{uri}");
    }
}
