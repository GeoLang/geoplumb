use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use geoplumb_server::config::Config;
use geoplumb_server::{Layer, router};
use projicio_core::Transform;
use terrano_core::{CogParams, CogReader, Raster, write_cog};
use tower::ServiceExt;

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

fn dem_at(lon: f64, lat: f64) -> f64 {
    500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos()
}

const CODES: [f64; 4] = [10.0, 20.0, 30.0, 80.0];

// classes change every one to three cells, so resampling twice moves many pixels to a neighbour's code
fn class_at(col: usize, row: usize) -> f64 {
    CODES[(col / 3 + row / 2 + col % 2) % CODES.len()]
}

fn write_cog_of(dir: &Path, name: &str, value: impl Fn(usize, usize) -> f64) -> String {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            data.push(value(col, row));
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
        ..CogParams::default()
    };
    let path = dir.join(name);
    let mut file = std::fs::File::create(&path).unwrap();
    write_cog(&raster, &params, &mut file).unwrap();
    // windows temp paths hold `\U`, a unicode escape in a basic toml string
    path.display().to_string().replace('\\', "/")
}

fn write_dem(dir: &Path) -> String {
    write_cog_of(dir, "dem.tif", |col, row| {
        let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
        let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
        dem_at(lon, lat)
    })
}

fn app(dir: &Path) -> Router {
    let text = format!(
        r#"
[[layer]]
name = "dem"
source = {{ kind = "cog", path = "{}" }}
gray = {{ min = 300.0, max = 700.0 }}

[[layer]]
name = "classes"
source = {{ kind = "cog", path = "{}", value_kind = "categorical" }}
"#,
        write_dem(dir),
        write_cog_of(dir, "classes.tif", class_at)
    );
    let config = Config::parse(&text).expect("the layer file must parse");
    let layers = Layer::build_all(config, 64 << 20, None).expect("the layers must build");
    router(layers)
}

struct Answer {
    status: StatusCode,
    content_type: String,
    content_disposition: String,
    body: Vec<u8>,
}

async fn get(app: &Router, uri: &str) -> Answer {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let header_text = |name: header::HeaderName| {
        response
            .headers()
            .get(name)
            .map(|value| value.to_str().unwrap().to_string())
            .unwrap_or_default()
    };
    let status = response.status();
    let content_type = header_text(header::CONTENT_TYPE);
    let content_disposition = header_text(header::CONTENT_DISPOSITION);
    let body = axum::body::to_bytes(response.into_body(), 64 << 20)
        .await
        .unwrap()
        .to_vec();
    Answer {
        status,
        content_type,
        content_disposition,
        body,
    }
}

// a 10 km by 6 km utm 32n window around 7.3 e 46.8 n, its corners on whole 100 m
fn utm_window() -> [f64; 4] {
    let to_utm = Transform::new("EPSG:4326", "EPSG:32632").unwrap();
    let (easting, northing) = to_utm.convert(7.3, 46.8).unwrap();
    let (easting, northing) = (
        (easting / 100.0).round() * 100.0,
        (northing / 100.0).round() * 100.0,
    );
    [
        easting - 5000.0,
        northing - 3000.0,
        easting + 5000.0,
        northing + 3000.0,
    ]
}

#[tokio::test(flavor = "multi_thread")]
async fn a_wgs84_layer_exports_onto_the_utm_grid_it_was_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let [min_x, min_y, max_x, max_y] = utm_window();
    let answer = get(
        &app,
        &format!("/export/dem?bbox={min_x},{min_y},{max_x},{max_y}&crs=EPSG:32632&resolution=100"),
    )
    .await;
    assert_eq!(
        answer.status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&answer.body)
    );
    assert_eq!(answer.content_type, "image/tiff");
    assert_eq!(
        answer.content_disposition,
        "attachment; filename=\"dem.tif\""
    );

    let reader = CogReader::open(answer.body.as_slice()).unwrap();
    let meta = reader.meta();
    assert_eq!(meta.epsg, 32632);
    assert_eq!((meta.origin_x, meta.origin_y), (min_x, max_y));
    assert_eq!((meta.pixel_width, meta.pixel_height), (100.0, 100.0));
    let base = &reader.levels()[0];
    assert_eq!((base.width, base.height), (100, 60));

    // every pixel center, taken back to lon/lat, has to hold the dem there
    let to_wgs84 = Transform::new("EPSG:32632", "EPSG:4326").unwrap();
    let pixels = reader.read_window(0, 0, 0, 100, 60).unwrap();
    for row in 0..60 {
        for col in 0..100 {
            let x = min_x + (col as f64 + 0.5) * 100.0;
            let y = max_y - (row as f64 + 0.5) * 100.0;
            let (lon, lat) = to_wgs84.convert(x, y).unwrap();
            let got = pixels.get(row, col).unwrap();
            assert!(
                (got - dem_at(lon, lat)).abs() < 0.05,
                "pixel {col},{row} at {lon},{lat} holds {got}, the dem is {}",
                dem_at(lon, lat)
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn a_categorical_export_samples_the_source_once_at_each_pixel_center() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let [min_x, min_y, max_x, max_y] = utm_window();
    let answer = get(
        &app,
        &format!(
            "/export/classes?bbox={min_x},{min_y},{max_x},{max_y}&crs=EPSG:32632&resolution=50"
        ),
    )
    .await;
    assert_eq!(
        answer.status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&answer.body)
    );
    let reader = CogReader::open(answer.body.as_slice()).unwrap();
    let (cols, rows) = (200, 120);
    assert_eq!(
        (reader.levels()[0].width, reader.levels()[0].height),
        (cols, rows)
    );
    let pixels = reader.read_window(0, 0, 0, cols, rows).unwrap();

    let to_wgs84 = Transform::new("EPSG:32632", "EPSG:4326").unwrap();
    let mut mismatches = 0;
    for row in 0..rows {
        for col in 0..cols {
            let x = min_x + (col as f64 + 0.5) * 50.0;
            let y = max_y - (row as f64 + 0.5) * 50.0;
            let (lon, lat) = to_wgs84.convert(x, y).unwrap();
            let source_col = ((lon - ORIGIN_X) / CELL).floor() as usize;
            let source_row = ((ORIGIN_Y - lat) / CELL).floor() as usize;
            if pixels.get(row, col).unwrap() != class_at(source_col, source_row) {
                mismatches += 1;
            }
        }
    }
    // a center within rounding of a cell edge can land in either cell
    assert!(
        mismatches * 1000 < cols * rows,
        "{mismatches} of {} pixels hold another cell's class",
        cols * rows
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_export_sizes_its_grid_the_way_gdalwarp_sizes_te_and_tr() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let [min_x, min_y, max_x, max_y] = utm_window();
    // gdalwarp rounds the 10 km by 6 km window's pixel counts
    for (resolution, size) in [(104.0, (96, 58)), (30.0, (333, 200))] {
        let answer = get(
            &app,
            &format!(
                "/export/dem?bbox={min_x},{min_y},{max_x},{max_y}&crs=EPSG:32632&resolution={resolution}"
            ),
        )
        .await;
        assert_eq!(
            answer.status,
            StatusCode::OK,
            "{}",
            String::from_utf8_lossy(&answer.body)
        );
        let reader = CogReader::open(answer.body.as_slice()).unwrap();
        assert_eq!(
            (reader.meta().origin_x, reader.meta().origin_y),
            (min_x, max_y)
        );
        let base = &reader.levels()[0];
        assert_eq!((base.width, base.height), size, "resolution {resolution}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn an_export_it_cannot_serve_is_refused_naming_the_problem() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path());
    let [min_x, min_y, max_x, max_y] = utm_window();
    let bbox = format!("{min_x},{min_y},{max_x},{max_y}");
    let cases = [
        (
            "/export/dem?bbox=1,2,3&crs=EPSG:32632&resolution=100".to_string(),
            "four numbers",
        ),
        (
            "/export/dem?bbox=a,b,c,d&crs=EPSG:32632&resolution=100".to_string(),
            "four numbers",
        ),
        (
            format!(
                "/export/dem?bbox={max_x},{min_y},{min_x},{max_y}&crs=EPSG:32632&resolution=100"
            ),
            "no area",
        ),
        (
            format!("/export/dem?bbox={bbox}&crs=32632&resolution=100"),
            "EPSG:<code>",
        ),
        (
            format!("/export/dem?bbox={bbox}&crs=EPSG:1&resolution=100"),
            "unknown to projicio",
        ),
        (format!("/export/dem?bbox={bbox}&resolution=100"), "crs"),
        (
            format!("/export/dem?bbox={bbox}&crs=EPSG:32632&resolution=0"),
            "positive",
        ),
        (
            format!("/export/dem?bbox={bbox}&crs=EPSG:32632&resolution=-100"),
            "positive",
        ),
        (
            format!("/export/dem?bbox={bbox}&crs=EPSG:32632&resolution=0.01"),
            "value export cap",
        ),
        (
            format!("/export/dem?bbox={bbox}&crs=EPSG:32632&resolution=100&t=yesterday"),
            "bad t parameter",
        ),
    ];
    for (uri, problem) in cases {
        let answer = get(&app, &uri).await;
        let reason = String::from_utf8_lossy(&answer.body);
        assert_eq!(answer.status, StatusCode::BAD_REQUEST, "{uri}: {reason}");
        assert!(reason.contains(problem), "{uri} answered {reason:?}");
    }

    let answer = get(
        &app,
        &format!("/export/nowhere?bbox={bbox}&crs=EPSG:32632&resolution=100"),
    )
    .await;
    assert_eq!(answer.status, StatusCode::NOT_FOUND);
}
