use std::path::Path;
use std::process::{Command, Output};

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use geoplumb_server::config::Config;
use geoplumb_server::{Layer, router};
use projicio_core::Transform;
use terrano_core::{CogParams, CogReader, RangeRead, Raster, SampleFormat, write_cog};
use topoi_core::geojson::{FeatureGeometry, read_geojson};
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

fn app(dir: &Path, export_dir: Option<&Path>) -> Router {
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
    router(layers, export_dir.map(Path::to_path_buf))
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
    let app = app(dir.path(), Some(dir.path()));
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

    assert_holds_the_dem_on_the_100_m_utm_grid(&CogReader::open(answer.body.as_slice()).unwrap());
}

fn assert_holds_the_dem_on_the_100_m_utm_grid<R: RangeRead>(reader: &CogReader<R>) {
    let [min_x, _, _, max_y] = utm_window();
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
    let app = app(dir.path(), Some(dir.path()));
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
    let app = app(dir.path(), Some(dir.path()));
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
    let app = app(dir.path(), Some(dir.path()));
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

#[tokio::test(flavor = "multi_thread")]
async fn a_lowercase_epsg_authority_names_the_same_crs() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(dir.path(), Some(dir.path()));
    let [min_x, min_y, max_x, max_y] = utm_window();
    let answer = get(
        &app,
        &format!("/export/dem?bbox={min_x},{min_y},{max_x},{max_y}&crs=epsg:32632&resolution=100"),
    )
    .await;
    assert_eq!(
        answer.status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&answer.body)
    );
    assert_eq!(
        CogReader::open(answer.body.as_slice()).unwrap().meta().epsg,
        32632
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn an_export_writes_its_file_under_the_export_dir() {
    let dir = tempfile::tempdir().unwrap();
    let [min_x, min_y, max_x, max_y] = utm_window();
    let uri =
        format!("/export/dem?bbox={min_x},{min_y},{max_x},{max_y}&crs=EPSG:32632&resolution=100");

    let answer = get(&app(dir.path(), None), &uri).await;
    let reason = String::from_utf8_lossy(&answer.body);
    assert_eq!(answer.status, StatusCode::INTERNAL_SERVER_ERROR, "{reason}");
    assert!(reason.contains("GEOPLUMB_EXPORT_DIR"), "{reason}");

    let export_dir = tempfile::tempdir().unwrap();
    let app = app(dir.path(), Some(export_dir.path()));
    std::fs::remove_dir_all(export_dir.path()).unwrap();
    let answer = get(&app, &uri).await;
    let reason = String::from_utf8_lossy(&answer.body);
    assert_eq!(answer.status, StatusCode::INTERNAL_SERVER_ERROR, "{reason}");
    assert!(reason.contains("no temp file for the export"), "{reason}");
}

#[tokio::test(flavor = "multi_thread")]
async fn an_export_of_a_clip_raster_layer_is_nodata_past_its_cutline() {
    let dir = tempfile::tempdir().unwrap();
    let (west, south, east, north) = (7.1302, 46.8202, 7.1698, 46.8798);
    let cutline = serde_json::json!({
        "type": "Polygon",
        "coordinates": [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    });
    let cutline_path = dir.path().join("cutline.geojson");
    std::fs::write(&cutline_path, cutline.to_string()).unwrap();
    let text = format!(
        r#"
[[layer]]
name = "cutline"
source = {{ kind = "geojson", path = "{}" }}

[[layer]]
name = "clipped"
source = {{ kind = "cog", path = "{}" }}
gray = {{ min = 300.0, max = 700.0 }}

[[layer.op]]
kind = "clip_raster"
cutline = "cutline"
"#,
        cutline_path.display().to_string().replace('\\', "/"),
        write_dem(dir.path()),
    );
    let config = Config::parse(&text).expect("the layer file must parse");
    let layers = Layer::build_all(config, 64 << 20, None).expect("the layers must build");
    let app = router(layers, Some(dir.path().to_path_buf()));

    let answer = get(
        &app,
        "/export/clipped?bbox=7.1,46.8,7.2,46.9&crs=EPSG:4326&resolution=0.001",
    )
    .await;
    assert_eq!(
        answer.status,
        StatusCode::OK,
        "{}",
        String::from_utf8_lossy(&answer.body)
    );
    let reader = CogReader::open(answer.body.as_slice()).unwrap();
    let pixels = reader.read_window(0, 0, 0, 100, 100).unwrap();
    // two cells clear of the cut, where the export's resampling reads no neighbour across it
    let margin = 0.002;
    let (mut kept, mut blanked) = (0, 0);
    for row in 0..100 {
        for col in 0..100 {
            let lon = 7.1 + (col as f64 + 0.5) * CELL;
            let lat = 46.9 - (row as f64 + 0.5) * CELL;
            let got = pixels.get(row, col).unwrap();
            let inside = west + margin < lon
                && lon < east - margin
                && south + margin < lat
                && lat < north - margin;
            let outside = lon < west - margin
                || lon > east + margin
                || lat < south - margin
                || lat > north + margin;
            if inside {
                kept += 1;
                assert!(
                    (got - dem_at(lon, lat)).abs() < 1e-6,
                    "pixel {col},{row} holds {got}"
                );
            } else if outside {
                blanked += 1;
                assert!(got.is_nan(), "pixel {col},{row} holds {got}");
            }
        }
    }
    assert!(kept > 0 && blanked > 0, "{kept} kept, {blanked} blanked");
}

fn geoplumb(arguments: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_geoplumb"))
        .args(arguments)
        .output()
        .unwrap()
}

fn stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).into_owned()
}

fn write_command_layers(dir: &Path) -> String {
    let sites = serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": { "name": "pier", "kind": "keep" },
                "geometry": { "type": "Point", "coordinates": [-122.45, 37.75] },
            },
            {
                "type": "Feature",
                "properties": { "name": "dropped", "kind": "drop" },
                "geometry": { "type": "Point", "coordinates": [-122.40, 37.80] },
            },
            {
                "type": "Feature",
                "properties": { "name": "outside", "kind": "keep" },
                "geometry": { "type": "Point", "coordinates": [-122.20, 37.75] },
            },
            {
                "type": "Feature",
                "properties": { "name": "park", "kind": "keep" },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [-122.48, 37.72], [-122.42, 37.72], [-122.42, 37.78],
                        [-122.48, 37.78], [-122.48, 37.72],
                    ]],
                },
            },
            {
                "type": "Feature",
                "properties": { "name": "ferry", "kind": "keep" },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[-122.35, 37.85], [-122.25, 37.85]],
                },
            },
        ],
    });
    let sites_path = dir.join("sites.geojson");
    std::fs::write(&sites_path, sites.to_string()).unwrap();
    let text = format!(
        r#"
[[layer]]
name = "dem"
source = {{ kind = "cog", path = "{dem}" }}

[[layer]]
name = "contours"
source = {{ kind = "cog", path = "{dem}" }}

[[layer.op]]
kind = "contours"
interval = 50.0

[[layer]]
name = "sites"
source = {{ kind = "geojson", path = "{sites}" }}

[[layer.op]]
kind = "vec_filter"
field = "kind"
equals = "keep"
"#,
        dem = write_dem(dir),
        sites = sites_path.display().to_string().replace('\\', "/"),
    );
    let path = dir.join("layers.toml");
    std::fs::write(&path, text).unwrap();
    path.display().to_string()
}

#[test]
fn geoplumb_run_writes_a_raster_layer_as_a_cog_on_the_grid_it_was_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    let layers = write_command_layers(dir.path());
    let [min_x, min_y, max_x, max_y] = utm_window();
    let out = dir.path().join("dem-utm.tif");
    let output = geoplumb(&[
        "run",
        &layers,
        "dem",
        "--bbox",
        &format!("{min_x},{min_y},{max_x},{max_y}"),
        "--crs",
        "EPSG:32632",
        "--resolution",
        "100",
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", stderr(&output));
    let file = std::fs::File::open(&out).unwrap();
    assert_holds_the_dem_on_the_100_m_utm_grid(&CogReader::open(file).unwrap());
}

#[test]
fn geoplumb_run_writes_contour_lines_as_geojson_on_their_elevation() {
    let dir = tempfile::tempdir().unwrap();
    let layers = write_command_layers(dir.path());
    let out = dir.path().join("contours.geojson");
    let output = geoplumb(&[
        "run",
        &layers,
        "contours",
        "--bbox",
        "7.2,46.7,7.4,46.9",
        "--crs",
        "EPSG:4326",
        "--resolution",
        "0.001",
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", stderr(&output));

    let collection = read_geojson(&std::fs::read_to_string(&out).unwrap()).unwrap();
    // the dem runs from about 300 to 480 m over this window
    let mut elevations: Vec<f64> = Vec::new();
    for feature in &collection.features {
        let elevation = feature.properties["elevation"].as_f64().unwrap();
        assert_eq!(elevation % 50.0, 0.0, "elevation {elevation}");
        elevations.push(elevation);
        let lines = match feature.geometry.as_ref().unwrap() {
            FeatureGeometry::LineString(line) => vec![line.clone()],
            FeatureGeometry::MultiLineString(lines) => lines.linestrings().to_vec(),
            other => panic!("contour {elevation} is {other:?}"),
        };
        for coord in lines.iter().flat_map(|line| line.coords()) {
            assert!(
                (7.2 - 1e-9..=7.4 + 1e-9).contains(&coord.x)
                    && (46.7 - 1e-9..=46.9 + 1e-9).contains(&coord.y),
                "contour {elevation} has a vertex at {},{} outside the bbox",
                coord.x,
                coord.y
            );
            let dem = dem_at(coord.x, coord.y);
            assert!(
                (dem - elevation).abs() < 0.05,
                "contour {elevation} passes {},{} where the dem is {dem}",
                coord.x,
                coord.y
            );
        }
    }
    elevations.sort_by(f64::total_cmp);
    elevations.dedup();
    assert_eq!(elevations, [350.0, 400.0, 450.0]);
}

#[test]
fn geoplumb_run_writes_a_filtered_vector_layer_clipped_to_a_western_bbox() {
    let dir = tempfile::tempdir().unwrap();
    let layers = write_command_layers(dir.path());
    let out = dir.path().join("sites.geojson");
    let output = geoplumb(&[
        "run",
        &layers,
        "sites",
        "--bbox",
        "-122.5,37.7,-122.3,37.9",
        "--crs",
        "EPSG:4326",
        "--resolution",
        "0.001",
        "-o",
        out.to_str().unwrap(),
    ]);
    assert!(output.status.success(), "{}", stderr(&output));

    let collection = read_geojson(&std::fs::read_to_string(&out).unwrap()).unwrap();
    let mut names: Vec<&str> = collection
        .features
        .iter()
        .map(|feature| feature.properties["name"].as_str().unwrap())
        .collect();
    names.sort_unstable();
    assert_eq!(names, ["ferry", "park", "pier"]);
    let geometry_of = |name: &str| {
        collection
            .features
            .iter()
            .find(|feature| feature.properties["name"] == name)
            .and_then(|feature| feature.geometry.clone())
            .unwrap()
    };

    let FeatureGeometry::Point(pier) = geometry_of("pier") else {
        panic!("the pier is no longer a point");
    };
    assert_eq!((pier.0.x, pier.0.y), (-122.45, 37.75));

    let park = match geometry_of("park") {
        FeatureGeometry::Polygon(polygon) => polygon,
        FeatureGeometry::MultiPolygon(multi) if multi.polygons().len() == 1 => {
            multi.polygons()[0].clone()
        }
        other => panic!("the park is {other:?}"),
    };
    let ring = park.exterior().coords();
    let west = ring
        .iter()
        .map(|coord| coord.x)
        .fold(f64::INFINITY, f64::min);
    let east = ring
        .iter()
        .map(|coord| coord.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let south = ring
        .iter()
        .map(|coord| coord.y)
        .fold(f64::INFINITY, f64::min);
    let north = ring
        .iter()
        .map(|coord| coord.y)
        .fold(f64::NEG_INFINITY, f64::max);
    for (got, want) in [
        (west, -122.48),
        (east, -122.42),
        (south, 37.72),
        (north, 37.78),
    ] {
        assert!(
            (got - want).abs() < 1e-6,
            "park edge {got}, the source has {want}"
        );
    }

    // the ferry line runs east past the bbox and is cut at its east edge
    let ferry = match geometry_of("ferry") {
        FeatureGeometry::LineString(line) => line.coords().to_vec(),
        FeatureGeometry::MultiLineString(lines) => lines
            .linestrings()
            .iter()
            .flat_map(|line| line.coords().to_vec())
            .collect(),
        other => panic!("the ferry is {other:?}"),
    };
    let east_end = ferry
        .iter()
        .map(|coord| coord.x)
        .fold(f64::NEG_INFINITY, f64::max);
    let west_end = ferry
        .iter()
        .map(|coord| coord.x)
        .fold(f64::INFINITY, f64::min);
    assert!(
        (east_end - -122.3).abs() < 1e-6,
        "the ferry ends at {east_end}"
    );
    assert!(
        (west_end - -122.35).abs() < 1e-6,
        "the ferry starts at {west_end}"
    );
    assert!(ferry.iter().all(|coord| (coord.y - 37.85).abs() < 1e-6));
}

#[test]
fn geoplumb_run_refuses_a_job_it_cannot_write_and_writes_no_file() {
    let dir = tempfile::tempdir().unwrap();
    let layers = write_command_layers(dir.path());
    let [min_x, min_y, max_x, max_y] = utm_window();
    let utm_bbox = format!("{min_x},{min_y},{max_x},{max_y}");
    let lon_lat_bbox = "7.2,46.7,7.4,46.9";
    let missing = dir.path().join("missing.toml").display().to_string();
    let cases: [(&str, &str, &str, &str, &str, &str); 9] = [
        (
            &layers,
            "dem",
            &utm_bbox,
            "EPSG:32632",
            "dem.gpkg",
            ".geojson for a vector layer",
        ),
        (
            &layers,
            "nowhere",
            &utm_bbox,
            "EPSG:32632",
            "dem.tif",
            "no layer is named nowhere, the file names dem, contours, sites",
        ),
        (
            &layers,
            "dem",
            lon_lat_bbox,
            "EPSG:4326",
            "dem.geojson",
            "raster layer, its output is a .tif",
        ),
        (
            &layers,
            "contours",
            lon_lat_bbox,
            "EPSG:4326",
            "contours.tif",
            "vector layer, its output is a .geojson",
        ),
        (
            &layers,
            "contours",
            &utm_bbox,
            "EPSG:32632",
            "contours.geojson",
            "--crs EPSG:4326",
        ),
        (
            &layers,
            "dem",
            "1,2,3",
            "EPSG:32632",
            "dem.tif",
            "four numbers",
        ),
        (&layers, "dem", &utm_bbox, "32632", "dem.tif", "EPSG:<code>"),
        (
            &layers,
            "dem",
            &utm_bbox,
            "EPSG:32632",
            "dem.tiff.txt",
            ".tif for a raster layer",
        ),
        (
            &missing,
            "dem",
            &utm_bbox,
            "EPSG:32632",
            "dem.tif",
            "missing.toml",
        ),
    ];
    let out_dir = dir.path().join("out");
    std::fs::create_dir(&out_dir).unwrap();
    for (layer_file, layer, bbox, crs, name, problem) in cases {
        let out = out_dir.join(name);
        let output = geoplumb(&[
            "run",
            layer_file,
            layer,
            "--bbox",
            bbox,
            "--crs",
            crs,
            "--resolution",
            "100",
            "-o",
            out.to_str().unwrap(),
        ]);
        let message = stderr(&output);
        assert_eq!(
            output.status.code(),
            Some(1),
            "{layer} to {name}: {message}"
        );
        assert!(message.contains(problem), "{layer} to {name}: {message}");
        assert!(!out.exists(), "{layer} to {name} left a file behind");
    }

    let output = geoplumb(&[
        "run",
        &layers,
        "dem",
        "--bbox",
        &utm_bbox,
        "--crs",
        "EPSG:32632",
        "--resolution",
        "100",
        "--t",
        "yesterday",
        "-o",
        out_dir.join("dem.tif").to_str().unwrap(),
    ]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("bad t parameter"),
        "{}",
        stderr(&output)
    );
    assert!(!out_dir.join("dem.tif").exists());

    let output = geoplumb(&["info", &layers]);
    assert_eq!(output.status.code(), Some(1));
    assert!(
        stderr(&output).contains("not a little-endian tiff"),
        "{}",
        stderr(&output)
    );
}

#[test]
fn geoplumb_info_prints_what_the_cog_declares() {
    let dir = tempfile::tempdir().unwrap();
    let raster = Raster::from_vec(W, H, vec![500.0; W * H], 20.0, f64::NAN).unwrap();
    let params = CogParams {
        tile_width: 256,
        tile_height: 256,
        overview_levels: 2,
        epsg: 32632,
        origin_x: 380000.0,
        origin_y: 5190000.0,
        pixel_width: 20.0,
        pixel_height: 20.0,
        deflate: true,
        format: SampleFormat::U16,
        nodata: Some(0.0),
        palette: None,
    };
    let path = dir.path().join("dem-u16.tif");
    write_cog(&raster, &params, &mut std::fs::File::create(&path).unwrap()).unwrap();

    let output = geoplumb(&["info", path.to_str().unwrap()]);
    assert!(output.status.success(), "{}", stderr(&output));
    assert_eq!(
        String::from_utf8_lossy(&output.stdout),
        "size: 600 x 400\n\
         bands: 1\n\
         sample format: u16\n\
         bits: 16\n\
         crs: EPSG:32632\n\
         geotransform: 380000, 20, 0, 5190000, 0, -20\n\
         compression: deflate\n\
         predictor: none\n\
         nodata: 0\n\
         overviews:\n\
         - 300 x 200\n\
         - 150 x 100\n"
    );
}
