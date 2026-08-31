//! config parsing, the `t` parameter, and the routes over a cog layer
//! built from a synthetic geotiff written at test time, so nothing here
//! touches the network

use std::collections::{HashMap, HashSet};
use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use geoplumb_server::config::{
    BinaryOpConfig, BurnConfig, Config, FaninConfig, FocalOpConfig, LayerConfig, OpConfig,
    SourceConfig, UnaryOpConfig,
};
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

/// vertex spacing of the geojson fixture, close enough to a z12 tile's
/// ground resolution that the source's ladder starts there
const PARCEL_STEP: f64 = 0.0005;
const PARCEL_MIN_LAT: f64 = 46.70;
const PARCEL_MAX_LAT: f64 = 46.85;

fn parcel_ring(min_lon: f64, max_lon: f64) -> Vec<[f64; 2]> {
    let corners = [
        (min_lon, PARCEL_MIN_LAT),
        (max_lon, PARCEL_MIN_LAT),
        (max_lon, PARCEL_MAX_LAT),
        (min_lon, PARCEL_MAX_LAT),
        (min_lon, PARCEL_MIN_LAT),
    ];
    let mut ring = Vec::new();
    for edge in corners.windows(2) {
        let (a, b) = (edge[0], edge[1]);
        let steps = ((b.0 - a.0).hypot(b.1 - a.1) / PARCEL_STEP).ceil() as usize;
        for step in 0..steps {
            let t = step as f64 / steps as f64;
            ring.push([a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)]);
        }
    }
    ring.push([min_lon, PARCEL_MIN_LAT]);
    ring
}

/// two squares meeting at 7.35, each with its own depth, together covering
/// the tile the render test asks for
fn parcels_geojson() -> String {
    let parcel = |min_lon: f64, max_lon: f64, depth: f64| {
        serde_json::json!({
            "type": "Feature",
            "properties": { "depth": depth },
            "geometry": {
                "type": "Polygon",
                "coordinates": [parcel_ring(min_lon, max_lon)],
            },
        })
    };
    serde_json::json!({
        "type": "FeatureCollection",
        "features": [parcel(7.25, 7.35, 1.0), parcel(7.35, 7.45, 3.0)],
    })
    .to_string()
}

fn source_of(layer: &LayerConfig) -> &SourceConfig {
    layer.source.as_ref().expect("a layer with one source")
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

/// writes the fixture into `dir` and gives back the path a layer file
/// names it by
fn write_dem(dir: &Path) -> String {
    let path = dir.join("dem.tif");
    std::fs::write(&path, dem_cog()).unwrap();
    // windows temp paths hold `\U`, a unicode escape in a basic toml string
    path.display().to_string().replace('\\', "/")
}

/// writes the geojson fixture into `dir` and gives back the path a layer
/// file names it by
fn write_parcels(dir: &Path) -> String {
    let path = dir.join("parcels.geojson");
    std::fs::write(&path, parcels_geojson()).unwrap();
    path.display().to_string().replace('\\', "/")
}

/// a one-layer file over a cog written into `dir`
fn cog_layer_file(dir: &Path) -> String {
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
        write_dem(dir)
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

/// the gray value of every pixel the png marks opaque, so an empty tile
/// is distinguishable from one that actually rendered data and two tiles
/// are comparable pixel by pixel
fn opaque_grays(bytes: &[u8]) -> Vec<u8> {
    let mut reader = png::Decoder::new(bytes).read_info().expect("a png");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::GrayscaleAlpha);
    buf[..info.buffer_size()]
        .chunks_exact(2)
        .filter(|px| px[1] == 255)
        .map(|px| px[0])
        .collect()
}

fn opaque_pixels(bytes: &[u8]) -> usize {
    opaque_grays(bytes).len()
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

[[layer]]
name = "slope"
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.op]]
kind = "slope"

[[layer]]
name = "aspect"
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.op]]
kind = "aspect"
"#;

/// one layer per single-input raster op, each over the same cog, with the
/// path filled in at test time. the dem is a smooth 300 to 700 surface and
/// the tile the render test asks for holds roughly 310 to 393 of it, which
/// is what the class break and every gray range here are picked against
const EVERY_OP: &str = r#"
[[layer]]
name = "plain"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 300.0, max = 400.0 }

[[layer]]
name = "focal"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 300.0, max = 400.0 }

[[layer.op]]
kind = "focal"
op = "mean"
radius = 2

[[layer]]
name = "reclassified"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 0.0, max = 3.0 }

[[layer.op]]
kind = "reclassify"
classes = [
    { min = 0.0, max = 350.0, value = 1.0 },
    { min = 350.0, max = 1000.0, value = 2.0 },
]

[[layer]]
name = "masked"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 0.0, max = 3.0 }

[[layer.op]]
kind = "reclassify"
classes = [
    { min = 0.0, max = 350.0, value = 1.0 },
    { min = 350.0, max = 1000.0, value = 2.0 },
]

[[layer.op]]
kind = "mask"
band = 0
valid_values = [1.0]

[[layer]]
name = "rooted"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 17.0, max = 20.0 }

[[layer.op]]
kind = "unary"
op = "sqrt"

[[layer]]
name = "convolved"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 300.0, max = 400.0 }

[[layer.op]]
kind = "convolve"
kernel = [
    [0.0625, 0.125, 0.0625],
    [0.125, 0.25, 0.125],
    [0.0625, 0.125, 0.0625],
]
"#;

const EVERY_OP_LAYER: [&str; 6] = [
    "plain",
    "focal",
    "reclassified",
    "masked",
    "rooted",
    "convolved",
];

/// class 1 and class 2 stretched over the `gray = { min = 0.0, max = 3.0 }`
/// the reclassified layers name
const CLASS_ONE_GRAY: u8 = 85;
const CLASS_TWO_GRAY: u8 = 170;

fn every_op_layer_file(dir: &Path) -> String {
    EVERY_OP.replace("COG_PATH", &write_dem(dir))
}

/// the dem plain, mosaicked against itself, and subtracted from itself
/// lifted 50 m, so both fan-in elements have a reference to be checked
/// against
const FAN_IN: &str = r#"
[[layer]]
name = "plain"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 300.0, max = 400.0 }

[[layer]]
name = "mosaicked"
fanin = { kind = "mosaic" }
gray = { min = 300.0, max = 400.0 }

[[layer.input]]
source = { kind = "cog", path = "COG_PATH" }

[[layer.input]]
source = { kind = "cog", path = "COG_PATH" }

[[layer]]
name = "dropped"
fanin = { kind = "combine", op = "subtract" }
gray = { min = -200.0, max = 0.0 }

[[layer.input]]
source = { kind = "cog", path = "COG_PATH" }

[[layer.input]]
source = { kind = "cog", path = "COG_PATH" }

[[layer.input.op]]
kind = "unary"
op = { add = 50.0 }
"#;

/// -50 m stretched over the -200 to 0 range the `dropped` layer names
const DROPPED_GRAY: u8 = 191;

const GEOJSON: &str = r#"
[[layer]]
name = "parcels"
source = { kind = "geojson", path = "GEOJSON_PATH" }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "rasterize"
burn = { property = "depth" }
"#;

/// the two parcel depths, 1.0 and 3.0, over the 0 to 4 range the layer names
const SHALLOW_GRAY: u8 = 64;
const DEEP_GRAY: u8 = 191;

#[test]
fn config_parses_sources_ops_and_their_order() {
    let config = Config::parse(FULL).unwrap();
    assert_eq!(config.layers.len(), 4);

    let ndvi = &config.layers[0];
    assert_eq!(ndvi.name, "ndvi");
    let search = source_of(ndvi).stac_search().expect("a stac source");
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
    assert!(source_of(dem).stac_search().is_none());
    match source_of(dem) {
        SourceConfig::Cog { path } => assert_eq!(path.to_str(), Some("/data/dem.tif")),
        other => panic!("unexpected source: {other:?}"),
    }
    assert!(matches!(dem.ops[..], [OpConfig::Hillshade { .. }]));
    assert!(matches!(config.layers[2].ops[..], [OpConfig::Slope]));
    assert!(matches!(config.layers[3].ops[..], [OpConfig::Aspect]));
}

#[test]
fn config_parses_every_single_input_op() {
    let config = Config::parse(EVERY_OP).unwrap();
    let named = |name: &str| {
        config
            .layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("no layer {name}"))
    };
    assert_eq!(config.layers.len(), EVERY_OP_LAYER.len());
    assert!(named("plain").ops.is_empty());

    match &named("focal").ops[..] {
        [OpConfig::Focal { op, radius }] => {
            assert!(matches!(op, FocalOpConfig::Mean));
            assert_eq!(*radius, 2);
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &named("reclassified").ops[..] {
        [OpConfig::Reclassify { classes }] => {
            assert_eq!(classes.len(), 2);
            assert_eq!(
                (classes[0].min, classes[0].max, classes[0].value),
                (0.0, 350.0, 1.0)
            );
            assert_eq!(
                (classes[1].min, classes[1].max, classes[1].value),
                (350.0, 1000.0, 2.0)
            );
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &named("masked").ops[..] {
        [
            OpConfig::Reclassify { .. },
            OpConfig::Mask { band, valid_values },
        ] => {
            assert_eq!(*band, 0);
            assert_eq!(valid_values, &[1.0]);
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &named("rooted").ops[..] {
        [OpConfig::Unary { op }] => assert!(matches!(op, UnaryOpConfig::Sqrt)),
        other => panic!("unexpected ops: {other:?}"),
    }

    match &named("convolved").ops[..] {
        [
            OpConfig::Convolve {
                kernel,
                scales,
                offsets,
            },
        ] => {
            assert_eq!(kernel[1][1], 0.25);
            assert_eq!(kernel[0], [0.0625, 0.125, 0.0625]);
            // a convolve naming neither runs one channel through unchanged
            assert_eq!(scales, &[1.0]);
            assert_eq!(offsets, &[0.0]);
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    let gray = named("plain").gray.expect("a gray range");
    assert_eq!((gray.min, gray.max), (300.0, 400.0));
}

#[test]
fn the_shipped_example_layer_file_parses() {
    let text = include_str!("../examples/layers.toml");
    Config::parse(text).expect("the file the readme tells people to copy");
}

#[test]
fn a_unary_op_takes_its_constant_where_it_has_one() {
    let text = r#"
[[layer]]
name = "doubled"
source = { kind = "cog", path = "/a.tif" }

[[layer.op]]
kind = "unary"
op = { multiply = 2.0 }
"#;
    let config = Config::parse(text).unwrap();
    match &config.layers[0].ops[..] {
        [OpConfig::Unary { op }] => assert!(matches!(op, UnaryOpConfig::Multiply(2.0))),
        other => panic!("unexpected ops: {other:?}"),
    }
}

#[test]
fn a_reclassify_layer_has_to_name_its_own_gray_range() {
    let text = r#"
[[layer]]
name = "classes"
source = { kind = "cog", path = "/a.tif" }

[[layer.op]]
kind = "reclassify"
classes = [{ min = 0.0, max = 350.0, value = 1.0 }]
"#;
    let error = Config::parse(text).expect_err("class values have no range to stretch over");
    assert!(
        error.contains("gray"),
        "the error should name the field that is missing: {error}"
    );

    let with_range = text.replace(
        "source = { kind = \"cog\", path = \"/a.tif\" }",
        "source = { kind = \"cog\", path = \"/a.tif\" }\ngray = { min = 0.0, max = 3.0 }",
    );
    Config::parse(&with_range).expect("a named gray range makes it servable");
}

#[test]
fn config_parses_a_fan_in_layer_with_a_chain_on_each_side_of_the_join() {
    let text = r#"
[[layer]]
name = "burn_change"
fanin = { kind = "mosaic" }
gray = { min = -1.0, max = 1.0 }

[[layer.input]]
source = { kind = "cog", path = "/before.tif" }

[[layer.input.op]]
kind = "unary"
op = { multiply = 2.0 }

[[layer.input]]
source = { kind = "cog", path = "/after.tif" }

[[layer]]
name = "shaded"
fanin = { kind = "mosaic" }

[[layer.input]]
source = { kind = "cog", path = "/a.tif" }

[[layer.input]]
source = { kind = "cog", path = "/b.tif" }

[[layer.op]]
kind = "slope"
"#;
    let config = Config::parse(text).unwrap();
    let change = &config.layers[0];
    assert!(change.source.is_none());
    assert!(matches!(change.fanin, Some(FaninConfig::Mosaic)));
    assert_eq!(change.inputs.len(), 2);
    assert!(matches!(
        change.inputs[0].ops[..],
        [OpConfig::Unary {
            op: UnaryOpConfig::Multiply(2.0)
        }]
    ));
    assert!(change.inputs[1].ops.is_empty());
    // the layer's own ops are the chain after the join, not any input's
    assert!(change.ops.is_empty());
    assert!(matches!(config.layers[1].ops[..], [OpConfig::Slope]));
    assert_eq!(change.gray_range().unwrap(), (-1.0, 1.0));
    assert_eq!(config.layers[1].gray_range().unwrap(), (0.0, 90.0));
}

#[test]
fn config_parses_a_combine_layer_and_its_binary_op() {
    let text = r#"
[[layer]]
name = "difference"
fanin = { kind = "combine", op = "subtract" }
gray = { min = -100.0, max = 100.0 }

[[layer.input]]
source = { kind = "cog", path = "/a.tif" }

[[layer.input]]
source = { kind = "cog", path = "/b.tif" }
"#;
    let config = Config::parse(text).unwrap();
    assert!(matches!(
        config.layers[0].fanin,
        Some(FaninConfig::Combine {
            op: BinaryOpConfig::Subtract
        })
    ));

    for spelling in ["add", "subtract", "multiply", "divide", "min", "max"] {
        let text = text.replace("\"subtract\"", &format!("\"{spelling}\""));
        Config::parse(&text).unwrap_or_else(|e| panic!("{spelling}: {e}"));
    }
}

#[test]
fn config_parses_a_geojson_layer_burning_a_property_or_a_constant() {
    let text = r#"
[[layer]]
name = "parcels"
source = { kind = "geojson", path = "/parcels.geojson" }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "rasterize"
burn = { property = "depth" }
"#;
    let config = Config::parse(text).unwrap();
    match source_of(&config.layers[0]) {
        SourceConfig::Geojson { path } => assert_eq!(path.to_str(), Some("/parcels.geojson")),
        other => panic!("unexpected source: {other:?}"),
    }
    match &config.layers[0].ops[..] {
        [OpConfig::Rasterize { burn }] => match burn {
            BurnConfig::Property(name) => assert_eq!(name, "depth"),
            other => panic!("unexpected burn: {other:?}"),
        },
        other => panic!("unexpected ops: {other:?}"),
    }

    let constant = text.replace("{ property = \"depth\" }", "{ constant = 1.0 }");
    let config = Config::parse(&constant).unwrap();
    match &config.layers[0].ops[..] {
        [
            OpConfig::Rasterize {
                burn: BurnConfig::Constant(value),
            },
        ] => assert_eq!(*value, 1.0),
        other => panic!("unexpected ops: {other:?}"),
    }
}

#[test]
fn a_layer_that_leaves_the_gray_range_open_has_to_name_one() {
    // a rasterize leaves burned values, and a fanin carries no range across
    // the join, so both need the layer to say what to stretch over
    let cases = [
        r#"
[[layer]]
name = "parcels"
source = { kind = "geojson", path = "/parcels.geojson" }

[[layer.op]]
kind = "rasterize"
burn = { constant = 1.0 }
"#,
        r#"
[[layer]]
name = "mosaicked"
fanin = { kind = "mosaic" }

[[layer.input]]
source = { kind = "cog", path = "/a.tif" }

[[layer.input]]
source = { kind = "cog", path = "/b.tif" }
"#,
    ];
    for text in cases {
        let error = Config::parse(text).expect_err("no op fixes a range here");
        assert!(error.contains("gray"), "{error}");
        let with_range = text.replace("name = ", "gray = { min = 0.0, max = 4.0 }\nname = ");
        Config::parse(&with_range).expect("a named gray range makes it servable");
    }

    // an op after the fanin that fixes a range is the other way out
    let after_the_join = r#"
[[layer]]
name = "mosaicked"
fanin = { kind = "mosaic" }

[[layer.input]]
source = { kind = "cog", path = "/a.tif" }

[[layer.input]]
source = { kind = "cog", path = "/b.tif" }

[[layer.op]]
kind = "slope"
"#;
    let config = Config::parse(after_the_join).unwrap();
    assert_eq!(config.layers[0].gray_range().unwrap(), (0.0, 90.0));
}

#[test]
fn every_composite_spelling_reaches_the_search() {
    let cases = [
        ("\"latest\"", geoplumb::elements::Composite::Latest),
        ("\"mean\"", geoplumb::elements::Composite::Mean),
        ("\"min\"", geoplumb::elements::Composite::Min),
        ("\"max\"", geoplumb::elements::Composite::Max),
        ("\"stddev\"", geoplumb::elements::Composite::StdDev),
        ("\"count\"", geoplumb::elements::Composite::Count),
        (
            "{ percentile = 90.0 }",
            geoplumb::elements::Composite::Percentile(90.0),
        ),
    ];
    for (toml_value, want) in cases {
        let text = format!(
            r#"
[[layer]]
name = "stack"
source = {{ kind = "stac", api = "https://example.test/v1", collection = "c", assets = ["data"], bbox = [7.0, 46.3, 8.0, 46.9], composite = {toml_value} }}
"#
        );
        let config = Config::parse(&text).unwrap_or_else(|e| panic!("{toml_value}: {e}"));
        let search = source_of(&config.layers[0])
            .stac_search()
            .expect("a stac source");
        assert_eq!(search.composite, want, "{toml_value}");
    }
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
kind = "contours""#,
            "op outside the supported set",
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
        (
            r#"[[layer]]
name = "a"
source = { kind = "cog", path = "/a.tif" }
fanin = { kind = "combine", op = "subtract" }
[[layer.input]]
source = { kind = "cog", path = "/b.tif" }"#,
            "a source beside inputs",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "cog", path = "/a.tif" }
[[layer.input]]
source = { kind = "cog", path = "/b.tif" }"#,
            "a source beside inputs with no fanin",
        ),
        (
            r#"[[layer]]
name = "a"
fanin = { kind = "mosaic" }"#,
            "a fanin with no inputs",
        ),
        (
            r#"[[layer]]
name = "a"
gray = { min = 0.0, max = 1.0 }
[[layer.input]]
source = { kind = "cog", path = "/a.tif" }
[[layer.input]]
source = { kind = "cog", path = "/b.tif" }"#,
            "inputs with no fanin",
        ),
        (
            r#"[[layer]]
name = "a"
fanin = { kind = "combine", op = "subtract" }
gray = { min = 0.0, max = 1.0 }
[[layer.input]]
source = { kind = "cog", path = "/a.tif" }"#,
            "combine with one input",
        ),
        (
            r#"[[layer]]
name = "a"
fanin = { kind = "combine", op = "subtract" }
gray = { min = 0.0, max = 1.0 }
[[layer.input]]
source = { kind = "cog", path = "/a.tif" }
[[layer.input]]
source = { kind = "cog", path = "/b.tif" }
[[layer.input]]
source = { kind = "cog", path = "/c.tif" }"#,
            "combine with three inputs",
        ),
        (
            r#"[[layer]]
name = "a"
fanin = { kind = "mosaic" }
gray = { min = 0.0, max = 1.0 }
[[layer.input]]
source = { kind = "cog", path = "/a.tif" }"#,
            "mosaic with one input",
        ),
        (
            r#"[[layer]]
name = "a"
fanin = { kind = "nearest" }
gray = { min = 0.0, max = 1.0 }
[[layer.input]]
source = { kind = "cog", path = "/a.tif" }
[[layer.input]]
source = { kind = "cog", path = "/b.tif" }"#,
            "unknown fanin kind",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "geojson", path = "/a.geojson" }
gray = { min = 0.0, max = 1.0 }"#,
            "a geojson layer with no rasterize",
        ),
        (
            r#"[[layer]]
name = "a"
fanin = { kind = "mosaic" }
gray = { min = 0.0, max = 1.0 }
[[layer.input]]
source = { kind = "geojson", path = "/a.geojson" }
[[layer.input]]
source = { kind = "cog", path = "/b.tif" }
[[layer.op]]
kind = "rasterize"
burn = { constant = 1.0 }"#,
            "a geojson input rasterized only after the join",
        ),
    ];
    for (text, why) in cases {
        assert!(Config::parse(text).is_err(), "{why} should not have parsed");
    }
}

#[test]
fn a_geojson_layer_without_a_rasterize_is_named_at_parse() {
    let text = r#"
[[layer]]
name = "parcels"
fanin = { kind = "mosaic" }
gray = { min = 0.0, max = 4.0 }

[[layer.input]]
source = { kind = "cog", path = "/a.tif" }

[[layer.input]]
source = { kind = "geojson", path = "/parcels.geojson" }
"#;
    let error = Config::parse(text).expect_err("a png encoder cannot take features");
    assert!(error.contains("parcels"), "{error}");
    assert!(error.contains("input 2"), "{error}");
    assert!(error.contains("rasterize"), "{error}");
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
async fn every_op_renders_a_tile_and_the_pixels_show_it_ran() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&every_op_layer_file(dir.path()));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let mut grays: HashMap<&str, Vec<u8>> = HashMap::new();
    for name in EVERY_OP_LAYER {
        let (status, png) = get(&app, &format!("/tiles/{name}/12/{x}/{y}.png")).await;
        assert_eq!(status, StatusCode::OK, "{name}");
        assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n", "{name}");
        let opaque = opaque_grays(&png);
        assert!(!opaque.is_empty(), "{name} rendered nothing but nodata");
        grays.insert(name, opaque);
    }

    let distinct = |name: &str| grays[name].iter().collect::<HashSet<_>>().len();
    assert!(
        distinct("plain") > 10,
        "the untransformed dem is a smooth ramp, not {} values",
        distinct("plain")
    );

    // reclassify collapses that ramp onto its two class values, and only
    // the bilinear reprojection across a class boundary lands between them
    let classed = grays["reclassified"]
        .iter()
        .filter(|g| **g == CLASS_ONE_GRAY || **g == CLASS_TWO_GRAY)
        .count();
    assert!(
        grays["reclassified"].contains(&CLASS_ONE_GRAY)
            && grays["reclassified"].contains(&CLASS_TWO_GRAY),
        "both classes should reach the tile"
    );
    assert!(
        classed * 10 > grays["reclassified"].len() * 9,
        "only class boundaries should sit off a class value, {classed} of {} did not",
        grays["reclassified"].len()
    );

    // the mask keeps class one and drops class two, so it renders a strict
    // subset of the same layer without it
    assert!(
        grays["masked"].len() < grays["reclassified"].len(),
        "the mask dropped nothing"
    );
    assert!(
        !grays["masked"].contains(&CLASS_TWO_GRAY),
        "class two survived the mask"
    );

    // sqrt of a 300 to 400 dem lands in the 17 to 20 range the layer
    // stretches over, so the tile is neither all black nor all white
    assert!(distinct("rooted") > 10);
    assert_ne!(grays["rooted"], grays["plain"]);

    // both are smoothings of the dem, so they stay in its range and differ
    // from it pixel by pixel
    assert_ne!(grays["focal"], grays["plain"]);
    assert_ne!(grays["convolved"], grays["plain"]);
    assert_eq!(grays["focal"].len(), grays["plain"].len());
    assert_eq!(grays["convolved"].len(), grays["plain"].len());
}

#[tokio::test]
async fn a_fan_in_layer_renders_the_tile_its_inputs_add_up_to() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&FAN_IN.replace("COG_PATH", &write_dem(dir.path())));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let mut grays: HashMap<&str, Vec<u8>> = HashMap::new();
    for name in ["plain", "mosaicked", "dropped"] {
        let (status, png) = get(&app, &format!("/tiles/{name}/12/{x}/{y}.png")).await;
        assert_eq!(status, StatusCode::OK, "{name}");
        let opaque = opaque_grays(&png);
        assert!(!opaque.is_empty(), "{name} rendered nothing but nodata");
        grays.insert(name, opaque);
    }

    // a mosaic of the dem against itself is the dem, first input winning
    // every pixel
    assert_eq!(grays["mosaicked"].len(), grays["plain"].len());
    for (i, (a, b)) in grays["mosaicked"].iter().zip(&grays["plain"]).enumerate() {
        assert!(
            a.abs_diff(*b) <= 1,
            "pixel {i}: mosaicked {a} against plain {b}"
        );
    }

    // the second input is the same dem lifted 50 m, so the combine leaves
    // -50 everywhere the dem has a value
    assert_eq!(grays["dropped"].len(), grays["plain"].len());
    for (i, g) in grays["dropped"].iter().enumerate() {
        assert_eq!(*g, DROPPED_GRAY, "pixel {i}");
    }
}

#[tokio::test]
async fn a_geojson_layer_burns_its_features_into_the_tile() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&GEOJSON.replace("GEOJSON_PATH", &write_parcels(dir.path())));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let (status, png) = get(&app, &format!("/tiles/parcels/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    let grays = opaque_grays(&png);
    // the two parcels cover the tile between them
    assert!(grays.len() > 60_000, "only {} pixels burned", grays.len());
    assert!(grays.contains(&SHALLOW_GRAY), "the 1.0 parcel is missing");
    assert!(grays.contains(&DEEP_GRAY), "the 3.0 parcel is missing");
    // only the seam between them, where the reprojection samples across
    // both, lands off a burned value
    let burned = grays
        .iter()
        .filter(|g| **g == SHALLOW_GRAY || **g == DEEP_GRAY)
        .count();
    assert!(
        burned * 10 > grays.len() * 9,
        "{burned} of {} pixels hold a parcel depth",
        grays.len()
    );
}

#[tokio::test]
async fn layers_publishes_a_fan_in_and_a_geojson_layer_by_what_they_read() {
    let dir = tempfile::tempdir().unwrap();
    let text = format!(
        "{}{}",
        FAN_IN.replace("COG_PATH", &write_dem(dir.path())),
        GEOJSON.replace("GEOJSON_PATH", &write_parcels(dir.path()))
    );
    let app = app(&text);

    let (status, body) = get(&app, "/layers").await;
    assert_eq!(status, StatusCode::OK);
    let layers: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let named = |name: &str| {
        layers
            .as_array()
            .unwrap()
            .iter()
            .find(|l| l["name"] == name)
            .unwrap_or_else(|| panic!("no layer {name}"))
            .clone()
    };
    assert_eq!(named("mosaicked")["source"], "composite");
    assert!(named("mosaicked")["collection"].is_null());
    assert!(named("mosaicked")["default_datetime"].is_null());
    assert_eq!(named("parcels")["source"], "geojson");
    assert!(named("parcels")["collection"].is_null());
    assert!(named("parcels")["temporal_extent"].is_null());
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
