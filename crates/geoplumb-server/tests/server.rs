//! config parsing, the `t` parameter, and the routes over a cog layer
//! built from a synthetic geotiff written at test time, so nothing here
//! touches the network

#[path = "../../geoplumb/tests/gpkg_fixture/mod.rs"]
mod gpkg_fixture;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::Path;

use axum::Router;
use axum::body::Body;
use axum::http::{Request, StatusCode};
use geoplumb::caps::ValueKind;
use geoplumb_server::config::{
    BinaryOpConfig, BurnConfig, ColorSelectionConfig, Config, FaninConfig, FocalOpConfig,
    LayerConfig, OpConfig, SourceConfig, UnaryOpConfig, ValueKindConfig,
};
use geoplumb_server::{Layer, parse_time, router};
use jung_mvt::{AttributeValue, TileGeometry, VectorLayer, decode_tile};
use nubis_core::{Point3, PointCloud, write_las};
use rusqlite::types::Value;
use terrano_core::{
    BandedRaster, CogParams, CogReader, GeoTiffMetadata, Raster, write_cog, write_cog_bands,
};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{Coord, Polygon};
use tower::ServiceExt;

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

/// an alpine-ish dem, a smooth 300 to 700 surface
fn dem_values() -> Vec<f64> {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos());
        }
    }
    data
}

fn cog_params() -> CogParams {
    CogParams {
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
    }
}

/// the dem as a deflate cog, the fixture a `cog` layer serves
fn dem_cog() -> Vec<u8> {
    let raster = Raster::from_vec(W, H, dem_values(), CELL, f64::NAN).unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    write_cog(&raster, &cog_params(), &mut buf).unwrap();
    buf.into_inner()
}

/// how far the second band of the two-band fixture sits above the first,
/// a gap the resampling leaves alone because both bands warp together
const BAND_LIFT: f64 = 1000.0;

/// the dem beside a copy of it lifted, so a reader of a served tile can
/// tell the two bands apart and check both survived
fn banded_dem_cog() -> Vec<u8> {
    let base = dem_values();
    let lifted = base.iter().map(|v| v + BAND_LIFT).collect();
    let bands = BandedRaster::new(vec![
        Raster::from_vec(W, H, base, CELL, f64::NAN).unwrap(),
        Raster::from_vec(W, H, lifted, CELL, f64::NAN).unwrap(),
    ])
    .unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    write_cog_bands(&bands, &cog_params(), &mut buf).unwrap();
    buf.into_inner()
}

/// vertex spacing of the geojson fixture, close enough to a z12 tile's
/// ground resolution that the source's ladder starts there
const PARCEL_STEP: f64 = 0.0005;
const PARCEL_MIN_LAT: f64 = 46.70;
const PARCEL_MAX_LAT: f64 = 46.85;

fn box_ring(min_lon: f64, min_lat: f64, max_lon: f64, max_lat: f64) -> Vec<[f64; 2]> {
    let corners = [
        (min_lon, min_lat),
        (max_lon, min_lat),
        (max_lon, max_lat),
        (min_lon, max_lat),
        (min_lon, min_lat),
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
    ring.push([min_lon, min_lat]);
    ring
}

/// two squares meeting at 7.35, each with its own depth and zone name,
/// together covering the tile the render test asks for
fn parcels_geojson() -> String {
    let parcel = |min_lon: f64, max_lon: f64, depth: f64, zone: &str| {
        serde_json::json!({
            "type": "Feature",
            "properties": { "depth": depth, "zone": zone },
            "geometry": {
                "type": "Polygon",
                "coordinates": [box_ring(min_lon, PARCEL_MIN_LAT, max_lon, PARCEL_MAX_LAT)],
            },
        })
    };
    serde_json::json!({
        "type": "FeatureCollection",
        "features": [
            parcel(7.25, 7.35, 1.0, "shallow"),
            parcel(7.35, 7.45, 3.0, "deep"),
        ],
    })
    .to_string()
}

/// the eastern edge of the clip boundary, inside the rendered tile so the
/// cut takes a visible bite out of it
const BOUNDARY_MAX_LON: f64 = 7.32;

/// one polygon over the western part of the tile, the fixture a `vec_clip`
/// cuts against
fn boundary_geojson() -> String {
    serde_json::json!({
        "type": "Polygon",
        "coordinates": [box_ring(7.20, PARCEL_MIN_LAT, BOUNDARY_MAX_LON, PARCEL_MAX_LAT)],
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

/// half the side of the web mercator square, the extent an xyz grid covers
const HALF_WORLD: f64 = 20_037_508.342_789_244;

/// the web mercator bounds of an xyz tile, min x, min y, max x, max y
fn tile_bounds(z: u8, x: u32, y: u32) -> (f64, f64, f64, f64) {
    let side = 2.0 * HALF_WORLD / f64::from(1u32 << z);
    let min_x = -HALF_WORLD + f64::from(x) * side;
    let max_y = HALF_WORLD - f64::from(y) * side;
    (min_x, max_y - side, min_x + side, max_y)
}

/// how far past the rendered tile the point cloud reaches, room for the
/// idw search radius at the tile's own edge
const LAS_MARGIN: f64 = 400.0;
/// point spacing in metres, under the 38 m a z12 tile's pixel covers, so
/// the source's ladder starts at the tile rather than above it
const LAS_STEP: f64 = 30.0;
/// the one height every point carries, so the idw grid over them has a
/// value the test can name
const LAS_HEIGHT: f64 = 120.0;
/// shifts the cloud off the tile's own cell grid, so the points do not
/// line up with the cell nodes the idw interpolates at
const LAS_OFFSET: f64 = 7.0;

/// a flat cloud over the rendered tile, written as a real las file. its
/// coordinates are web mercator metres, where the 1 mm las quantisation is
/// far under the spacing, which lon/lat degrees would not be
fn heights_las() -> Vec<u8> {
    let (tile_x, tile_y) = tile_of(12, 7.3, 46.8);
    let (min_x, min_y, max_x, max_y) = tile_bounds(12, tile_x, tile_y);
    let mut points = Vec::new();
    let mut y = min_y - LAS_MARGIN + LAS_OFFSET;
    while y <= max_y + LAS_MARGIN {
        let mut x = min_x - LAS_MARGIN + LAS_OFFSET;
        while x <= max_x + LAS_MARGIN {
            points.push(Point3::new(x, y, LAS_HEIGHT));
            x += LAS_STEP;
        }
        y += LAS_STEP;
    }
    let mut out = Vec::new();
    write_las(&PointCloud::from_points(points), &mut out).unwrap();
    out
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

// the geojson parcels as a geopackage feature table
fn write_parcels_gpkg(dir: &Path) -> String {
    let parcel = |id: i64, min_lon: f64, max_lon: f64, depth: f64, zone: &str| {
        let ring: Vec<Coord> = box_ring(min_lon, PARCEL_MIN_LAT, max_lon, PARCEL_MAX_LAT)
            .into_iter()
            .map(|[x, y]| Coord::new(x, y))
            .collect();
        gpkg_fixture::FixtureFeature {
            id,
            geometry: FeatureGeometry::Polygon(Polygon::from_coords(&ring)),
            properties: vec![Value::Real(depth), Value::Text(zone.to_string())],
        }
    };
    let path = dir.join("parcels.gpkg");
    gpkg_fixture::write_geopackage(
        &path,
        &gpkg_fixture::Layer {
            srs: gpkg_fixture::WGS84,
            columns: &[("depth", "REAL"), ("zone", "TEXT")],
            features: &[
                parcel(1, 7.25, 7.35, 1.0, "shallow"),
                parcel(2, 7.35, 7.45, 3.0, "deep"),
            ],
            rtree: true,
        },
    );
    path.display().to_string().replace('\\', "/")
}

/// writes a geojson file into `dir` and gives back the path a layer file
/// names it by
fn write_geojson(dir: &Path, name: &str, text: String) -> String {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
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
    let layers = Layer::build_all(config, 64 << 20, None).expect("the layers must build");
    router(layers, None)
}

/// the response's content type beside its body, which only the encoding
/// test looks at
async fn get_typed(app: &Router, uri: &str) -> (StatusCode, String, Vec<u8>) {
    let response = app
        .clone()
        .oneshot(Request::builder().uri(uri).body(Body::empty()).unwrap())
        .await
        .unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get(axum::http::header::CONTENT_TYPE)
        .map(|v| v.to_str().unwrap().to_string())
        .unwrap_or_default();
    let body = axum::body::to_bytes(response.into_body(), 8 << 20)
        .await
        .unwrap();
    (status, content_type, body.to_vec())
}

async fn get(app: &Router, uri: &str) -> (StatusCode, Vec<u8>) {
    let (status, _, body) = get_typed(app, uri).await;
    (status, body)
}

/// every pixel of a grayscale-alpha png as gray beside alpha
fn png_pixels(bytes: &[u8]) -> Vec<(u8, u8)> {
    let mut reader = png::Decoder::new(bytes).read_info().expect("a png");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::GrayscaleAlpha);
    buf[..info.buffer_size()]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|px| (px[0], px[1]))
        .collect()
}

/// the gray value of every pixel the png marks opaque, so an empty tile
/// is distinguishable from one that actually rendered data and two tiles
/// are comparable pixel by pixel
fn opaque_grays(bytes: &[u8]) -> Vec<u8> {
    png_pixels(bytes)
        .into_iter()
        .filter(|(_, alpha)| *alpha == 255)
        .map(|(gray, _)| gray)
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

const GPKG: &str = r#"
[[layer]]
name = "parcels"
source = { kind = "gpkg", path = "GPKG_PATH", table = "TABLE_NAME", crs = 4326 }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "rasterize"
burn = { property = "depth" }
"#;

fn gpkg_layer_file(path: &str) -> String {
    GPKG.replace("GPKG_PATH", path)
        .replace("TABLE_NAME", gpkg_fixture::TABLE)
}

/// the two parcel depths, 1.0 and 3.0, over the 0 to 4 range the layer names
const SHALLOW_GRAY: u8 = 64;
const DEEP_GRAY: u8 = 191;

/// one layer per vector op, each over the same parcels and each ending in
/// the rasterize that turns what the op left into pixels. every layer
/// stretches over the same 0 to 4 range the plain `parcels` layer does
const VEC_OPS: &str = r#"
[[layer]]
name = "filtered"
source = { kind = "geojson", path = "GEOJSON_PATH" }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "vec_filter"
field = "zone"
equals = "shallow"

[[layer.op]]
kind = "rasterize"
burn = { property = "depth" }

[[layer]]
name = "renamed"
source = { kind = "geojson", path = "GEOJSON_PATH" }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "vec_schema"
rename = { depth = "elevation" }

[[layer.op]]
kind = "rasterize"
burn = { property = "elevation" }

[[layer]]
name = "dropped"
source = { kind = "geojson", path = "GEOJSON_PATH" }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "vec_schema"
drop = ["depth"]

[[layer.op]]
kind = "rasterize"
burn = { property = "depth" }

[[layer]]
name = "filled"
source = { kind = "geojson", path = "GEOJSON_PATH" }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "vec_schema"
add = { fill = 2.5 }

[[layer.op]]
kind = "rasterize"
burn = { property = "fill" }

[[layer]]
name = "clipped"
source = { kind = "geojson", path = "GEOJSON_PATH" }
gray = { min = 0.0, max = 4.0 }

[[layer.op]]
kind = "vec_clip"
boundary = "BOUNDARY_PATH"

[[layer.op]]
kind = "rasterize"
burn = { property = "depth" }
"#;

/// the 2.5 the `filled` layer adds to every feature, over the same 0 to 4.
/// a value the gray scaling does not land halfway between two bytes, so
/// the reprojection's last bit cannot tip it either way
const FILL_GRAY: u8 = 159;

/// the same flat cloud gridded three ways: the idw defaults, a min_points
/// no cell of this cloud reaches, and a search radius under the spacing
const LAS: &str = r#"
[[layer]]
name = "heights"
source = { kind = "las", path = "LAS_PATH", crs = 3857 }
gray = { min = 0.0, max = 300.0 }

[[layer.op]]
kind = "idw"

[[layer]]
name = "sparse"
source = { kind = "las", path = "LAS_PATH", crs = 3857 }
gray = { min = 0.0, max = 300.0 }

[[layer.op]]
kind = "idw"
min_points = 1000

[[layer]]
name = "pinpoint"
source = { kind = "las", path = "LAS_PATH", crs = 3857 }
gray = { min = 0.0, max = 300.0 }

[[layer.op]]
kind = "idw"
power = 3.0
radius_px = 0.25
"#;

/// the cloud's one height over the 0 to 300 range the las layers name
const HEIGHT_GRAY: u8 = 102;

fn las_layer_file(dir: &Path) -> String {
    let path = dir.join("heights.las");
    std::fs::write(&path, heights_las()).unwrap();
    LAS.replace("LAS_PATH", &path.display().to_string().replace('\\', "/"))
}

const COPC: &str = r#"
[[layer]]
name = "copc-heights"
source = { kind = "copc", path = "COPC_PATH", crs = 3857 }
gray = { min = 0.0, max = 300.0 }

[[layer.op]]
kind = "idw"
"#;

fn copc_layer_file(dir: &Path) -> String {
    let las_path = dir.join("heights.las");
    std::fs::write(&las_path, heights_las()).unwrap();
    let copc_path = dir.join("heights.copc.laz");
    copc_writer::convert_las_to_copc_streaming(
        &las_path,
        &copc_path,
        &copc_writer::CopcWriterParams::default(),
        dir,
        &copc_core::NeverCancel,
    )
    .unwrap();
    COPC.replace(
        "COPC_PATH",
        &copc_path.display().to_string().replace('\\', "/"),
    )
}

/// the plain parcels layer beside one layer per vector op, all over the
/// same fixtures written into `dir`
fn vec_ops_layer_file(dir: &Path) -> String {
    let parcels = write_parcels(dir);
    let boundary = write_geojson(dir, "boundary.geojson", boundary_geojson());
    format!("{GEOJSON}{VEC_OPS}")
        .replace("GEOJSON_PATH", &parcels)
        .replace("BOUNDARY_PATH", &boundary)
}

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
        SourceConfig::Cog {
            path,
            value_kind,
            nodata,
        } => {
            assert_eq!(path.to_str(), Some("/data/dem.tif"));
            assert!(matches!(value_kind, ValueKindConfig::Continuous));
            assert_eq!(*nodata, None);
        }
        other => panic!("unexpected source: {other:?}"),
    }
    assert!(matches!(dem.ops[..], [OpConfig::Hillshade { .. }]));
    assert!(matches!(config.layers[2].ops[..], [OpConfig::Slope]));
    assert!(matches!(config.layers[3].ops[..], [OpConfig::Aspect]));
}

const CATEGORICAL: &str = r#"
[[layer]]
name = "landcover"
source = { kind = "cog", path = "/data/landcover.tif", value_kind = "categorical" }
gray = { min = 0.0, max = 100.0 }

[[layer]]
name = "scenes"
source = { kind = "stac", api = "https://example.test", collection = "landcover", assets = ["map"], bbox = [7.0, 46.3, 8.0, 46.9], value_kind = "categorical" }
gray = { min = 0.0, max = 100.0 }
"#;

#[test]
fn config_parses_a_categorical_value_kind() {
    let config = Config::parse(CATEGORICAL).unwrap();
    assert!(matches!(
        source_of(&config.layers[0]),
        SourceConfig::Cog {
            value_kind: ValueKindConfig::Categorical,
            ..
        }
    ));
    let search = source_of(&config.layers[1])
        .stac_search()
        .expect("a stac source");
    assert_eq!(
        search.value_kinds,
        BTreeMap::from([("map".to_string(), ValueKind::Categorical)])
    );

    let err = Config::parse(&CATEGORICAL.replacen("\"categorical\"", "\"ordinal\"", 1))
        .expect_err("an unknown value kind parsed");
    assert!(err.contains("ordinal"), "{err}");
}

#[test]
fn config_parses_a_peak_value_kind() {
    let config = Config::parse(&CATEGORICAL.replace("\"categorical\"", "\"peak\"")).unwrap();
    assert!(matches!(
        source_of(&config.layers[0]),
        SourceConfig::Cog {
            value_kind: ValueKindConfig::Peak,
            ..
        }
    ));
    let search = source_of(&config.layers[1])
        .stac_search()
        .expect("a stac source");
    assert_eq!(
        search.value_kinds,
        BTreeMap::from([("map".to_string(), ValueKind::Peak)])
    );

    let per_asset = PER_ASSET_KINDS.replace(
        "value_kind = { scl = \"categorical\" }",
        "composite = \"median\", value_kind = { red = \"peak\" }",
    );
    let config = Config::parse(&per_asset).unwrap_or_else(|e| panic!("{e}"));
    let search = source_of(&config.layers[0])
        .stac_search()
        .expect("a stac source");
    assert_eq!(
        search.asset_value_kinds().unwrap(),
        [ValueKind::Peak, ValueKind::Continuous]
    );
}

const PER_ASSET_KINDS: &str = r#"
[[layer]]
name = "clear"
source = { kind = "stac", api = "https://example.test", collection = "sentinel-2-l2a", assets = ["red", "scl"], bbox = [7.0, 46.3, 8.0, 46.9], value_kind = { scl = "categorical" } }
gray = { min = 0.0, max = 3000.0 }
"#;

#[test]
fn config_parses_a_value_kind_per_stac_asset() {
    let config = Config::parse(PER_ASSET_KINDS).unwrap();
    let search = source_of(&config.layers[0])
        .stac_search()
        .expect("a stac source");
    assert_eq!(
        search.asset_value_kinds().unwrap(),
        [ValueKind::Continuous, ValueKind::Categorical]
    );

    let err = Config::parse(&PER_ASSET_KINDS.replace("scl = ", "qa = "))
        .expect_err("a value kind for no asset parsed");
    assert!(err.contains("qa"), "{err}");

    let err = Config::parse(
        &PER_ASSET_KINDS.replace("value_kind = {", "composite = \"median\", value_kind = {"),
    )
    .expect_err("a median over a categorical asset parsed");
    assert!(err.contains("median") && err.contains("scl"), "{err}");
}

const CATEGORICAL_STAC: &str = r#"
[[layer]]
name = "scenes"
source = { kind = "stac", api = "https://example.test", collection = "landcover", assets = ["map"], bbox = [7.0, 46.3, 8.0, 46.9], value_kind = "categorical", composite = COMPOSITE }
gray = { min = 0.0, max = 100.0 }
"#;

#[test]
fn a_categorical_stac_layer_refuses_the_composites_that_blend_values() {
    for (composite, name) in [
        ("\"mean\"", "mean"),
        ("\"median\"", "median"),
        ("{ percentile = 90.0 }", "percentile"),
        ("\"stddev\"", "stddev"),
    ] {
        let err = Config::parse(&CATEGORICAL_STAC.replace("COMPOSITE", composite))
            .expect_err("a blending composite parsed over categorical values");
        assert!(err.contains(name), "{err}");
        assert!(err.contains("layer scenes"), "{err}");
    }
    for composite in ["\"latest\"", "\"min\"", "\"max\"", "\"count\""] {
        Config::parse(&CATEGORICAL_STAC.replace("COMPOSITE", composite))
            .unwrap_or_else(|e| panic!("{composite}: {e}"));
    }
}

fn write_classes(dir: &Path) -> String {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            data.push(1.0 + ((col / 7 + row / 5) % 2) as f64);
        }
    }
    let raster = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
    let params = CogParams {
        // terrano averages the overviews it writes
        overview_levels: 0,
        ..cog_params()
    };
    let path = dir.join("classes.tif");
    let mut file = std::fs::File::create(&path).unwrap();
    write_cog(&raster, &params, &mut file).unwrap();
    path.display().to_string().replace('\\', "/")
}

const CLASS_LAYERS: &str = r#"
[[layer]]
name = "classes"
source = { kind = "cog", path = "CLASS_PATH", value_kind = "categorical" }
gray = { min = 0.0, max = 3.0 }

[[layer]]
name = "blended"
source = { kind = "cog", path = "CLASS_PATH" }
gray = { min = 0.0, max = 3.0 }
"#;

#[tokio::test]
async fn a_categorical_cog_layer_serves_only_its_codes() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&CLASS_LAYERS.replace("CLASS_PATH", &write_classes(dir.path())));
    let off_class = |grays: &[u8]| {
        grays
            .iter()
            .filter(|g| **g != CLASS_ONE_GRAY && **g != CLASS_TWO_GRAY)
            .count()
    };

    for z in [9, 12] {
        let (x, y) = tile_of(z, 7.3, 46.8);
        let (status, png) = get(&app, &format!("/tiles/classes/{z}/{x}/{y}.png")).await;
        assert_eq!(status, StatusCode::OK);
        let grays = opaque_grays(&png);
        assert!(!grays.is_empty(), "z{z} rendered nothing but nodata");
        assert_eq!(off_class(&grays), 0, "z{z} holds a blended value");
    }

    let (x, y) = tile_of(12, 7.3, 46.8);
    let (status, png) = get(&app, &format!("/tiles/blended/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        off_class(&opaque_grays(&png)) > 0,
        "the continuous layer blends no classes"
    );
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
fn the_example_ndvi_layer_takes_a_median_under_an_item_mask() {
    let config = Config::parse(include_str!("../examples/layers.toml")).unwrap();
    let ndvi = config
        .layers
        .iter()
        .find(|layer| layer.name == "ndvi")
        .expect("an ndvi layer");
    let search = source_of(ndvi).stac_search().expect("a stac source");
    assert_eq!(search.assets, ["red", "nir"]);
    assert_eq!(search.composite, geoplumb::elements::Composite::Median);
    assert_eq!(
        search.item_mask,
        Some(geoplumb::elements::ItemMask {
            asset: "scl".into(),
            valid_values: vec![4.0, 5.0, 6.0],
        })
    );
    assert_eq!(
        search.asset_value_kinds().unwrap(),
        [ValueKind::Continuous, ValueKind::Continuous]
    );
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
source = { kind = "las", path = "/a.las", crs = 3857 }
gray = { min = 0.0, max = 1.0 }"#,
            "a las layer with no idw",
        ),
        (
            r#"[[layer]]
name = "a"
source = { kind = "las", path = "/a.las" }
gray = { min = 0.0, max = 1.0 }
[[layer.op]]
kind = "idw""#,
            "a las source with no crs",
        ),
    ];
    for (text, why) in cases {
        assert!(Config::parse(text).is_err(), "{why} should not have parsed");
    }
}

#[test]
fn config_parses_a_gpkg_layer() {
    let config = Config::parse(&gpkg_layer_file("/parcels.gpkg")).unwrap();
    match source_of(&config.layers[0]) {
        SourceConfig::Gpkg { path, table, crs } => {
            assert_eq!(path.to_str(), Some("/parcels.gpkg"));
            assert_eq!(table.as_deref(), Some(gpkg_fixture::TABLE));
            assert_eq!(*crs, Some(4326));
        }
        other => panic!("unexpected source: {other:?}"),
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

    // reclassify collapses that ramp onto its two class values, and its
    // categorical output reprojects and resamples without blending them
    let classed = grays["reclassified"]
        .iter()
        .filter(|g| **g == CLASS_ONE_GRAY || **g == CLASS_TWO_GRAY)
        .count();
    assert!(
        grays["reclassified"].contains(&CLASS_ONE_GRAY)
            && grays["reclassified"].contains(&CLASS_TWO_GRAY),
        "both classes should reach the tile"
    );
    assert_eq!(
        classed,
        grays["reclassified"].len(),
        "a pixel sits off a class value"
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
    // rasterize output is categorical, so the seam between them reprojects
    // without blending the two depths
    let burned = grays
        .iter()
        .filter(|g| **g == SHALLOW_GRAY || **g == DEEP_GRAY)
        .count();
    assert_eq!(burned, grays.len(), "a pixel sits off a parcel depth");
}

#[tokio::test]
async fn a_gpkg_layer_burns_its_features_into_the_tile() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&gpkg_layer_file(&write_parcels_gpkg(dir.path())));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let (status, png) = get(&app, &format!("/tiles/parcels/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    let grays = opaque_grays(&png);
    assert!(grays.len() > 60_000, "only {} pixels burned", grays.len());
    assert!(grays.contains(&SHALLOW_GRAY), "the 1.0 parcel is missing");
    assert!(grays.contains(&DEEP_GRAY), "the 3.0 parcel is missing");
    let burned = grays
        .iter()
        .filter(|g| **g == SHALLOW_GRAY || **g == DEEP_GRAY)
        .count();
    assert_eq!(burned, grays.len(), "a pixel sits off a parcel depth");

    let (status, body) = get(&app, "/layers").await;
    assert_eq!(status, StatusCode::OK);
    let layers: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(layers[0]["source"], "gpkg", "{layers}");
}

#[tokio::test]
async fn every_vector_op_changes_the_features_the_rasterize_burns() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&vec_ops_layer_file(dir.path()));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let mut grays: HashMap<&str, Vec<u8>> = HashMap::new();
    for name in [
        "parcels", "filtered", "renamed", "dropped", "filled", "clipped",
    ] {
        let (status, png) = get(&app, &format!("/tiles/{name}/12/{x}/{y}.png")).await;
        assert_eq!(status, StatusCode::OK, "{name}");
        grays.insert(name, opaque_grays(&png));
    }

    // the filter keeps the shallow zone, so the deep parcel's half of the
    // tile burns nothing
    assert!(!grays["filtered"].is_empty());
    assert!(
        !grays["filtered"].contains(&DEEP_GRAY),
        "the deep parcel survived the filter"
    );
    assert!(grays["filtered"].contains(&SHALLOW_GRAY));
    assert!(grays["filtered"].len() < grays["parcels"].len());

    // the rename is what the rasterize's `elevation` finds, so the layer
    // renders exactly what burning `depth` unrenamed does
    assert_eq!(grays["renamed"], grays["parcels"]);

    // dropping `depth` leaves the rasterize nothing to read, and a feature
    // without the property is skipped
    assert!(
        grays["dropped"].is_empty(),
        "{} pixels burned without the property",
        grays["dropped"].len()
    );

    // every feature carries the schema's added value, so the tile is uniform
    assert_eq!(grays["filled"].len(), grays["parcels"].len());
    for (i, g) in grays["filled"].iter().enumerate() {
        assert_eq!(*g, FILL_GRAY, "pixel {i}");
    }

    // the boundary stops at 7.32, inside the tile, and holds only shallow
    // parcel, so the clip drops the rest
    assert!(!grays["clipped"].is_empty());
    assert!(grays["clipped"].len() < grays["parcels"].len());
    assert!(
        !grays["clipped"].contains(&DEEP_GRAY),
        "the clip left features past its boundary"
    );
}

#[test]
fn config_parses_every_vector_op() {
    let dir = tempfile::tempdir().unwrap();
    let config = Config::parse(&vec_ops_layer_file(dir.path())).unwrap();
    let ops_of = |name: &str| {
        &config
            .layers
            .iter()
            .find(|l| l.name == name)
            .unwrap_or_else(|| panic!("no layer {name}"))
            .ops
    };

    match &ops_of("filtered")[..] {
        [
            OpConfig::VecFilter { field, equals },
            OpConfig::Rasterize { .. },
        ] => {
            assert_eq!(field, "zone");
            assert_eq!(equals, "shallow");
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &ops_of("renamed")[..] {
        [
            OpConfig::VecSchema { drop, rename, add },
            OpConfig::Rasterize { .. },
        ] => {
            assert_eq!(rename["depth"], "elevation");
            assert!(drop.is_empty());
            assert!(add.is_empty());
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &ops_of("dropped")[..] {
        [OpConfig::VecSchema { drop, .. }, OpConfig::Rasterize { .. }] => {
            assert_eq!(drop, &["depth"]);
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &ops_of("filled")[..] {
        [OpConfig::VecSchema { add, .. }, OpConfig::Rasterize { .. }] => {
            assert_eq!(add["fill"], 2.5);
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &ops_of("clipped")[..] {
        [OpConfig::VecClip { boundary }, OpConfig::Rasterize { .. }] => {
            assert!(boundary.ends_with("boundary.geojson"), "{boundary:?}");
        }
        other => panic!("unexpected ops: {other:?}"),
    }
}

#[test]
fn a_clip_boundary_that_is_not_polygonal_is_named_at_parse() {
    let dir = tempfile::tempdir().unwrap();
    let line = serde_json::json!({
        "type": "LineString",
        "coordinates": [[7.2, 46.7], [7.4, 46.9]],
    })
    .to_string();
    let text = format!(
        r#"
[[layer]]
name = "clipped"
source = {{ kind = "geojson", path = "{}" }}
gray = {{ min = 0.0, max = 4.0 }}

[[layer.op]]
kind = "vec_clip"
boundary = "{}"

[[layer.op]]
kind = "rasterize"
burn = {{ property = "depth" }}
"#,
        write_parcels(dir.path()),
        write_geojson(dir.path(), "line.geojson", line),
    );
    let error = Config::parse(&text).expect_err("a linestring bounds no area");
    assert!(error.contains("polygon"), "{error}");
    assert!(error.contains("linestring"), "{error}");

    let missing = text.replace("line.geojson", "absent.geojson");
    Config::parse(&missing).expect_err("a boundary file that is not there");
}

#[test]
fn an_idw_radius_that_bounds_no_neighbourhood_is_named_at_parse() {
    let dir = tempfile::tempdir().unwrap();
    for radius in ["0.0", "-5.0", "nan"] {
        let text = las_layer_file(dir.path()).replace(
            "kind = \"idw\"\n",
            &format!("kind = \"idw\"\nradius_px = {radius}\n"),
        );
        let error = Config::parse(&text).expect_err("a radius that bounds no neighbourhood");
        assert!(error.contains("radius_px"), "{error}");
    }
}

#[tokio::test]
async fn a_las_layer_grids_its_points_into_the_tile() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&las_layer_file(dir.path()));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let mut grays: HashMap<&str, Vec<u8>> = HashMap::new();
    for name in ["heights", "sparse", "pinpoint"] {
        let (status, png) = get(&app, &format!("/tiles/{name}/12/{x}/{y}.png")).await;
        assert_eq!(status, StatusCode::OK, "{name}");
        grays.insert(name, opaque_grays(&png));
    }

    // the cloud is flat and covers the whole tile, so the idw defaults
    // grid it into that one height everywhere
    assert!(
        grays["heights"].len() > 60_000,
        "only {} pixels gridded",
        grays["heights"].len()
    );
    for (i, g) in grays["heights"].iter().enumerate() {
        assert_eq!(*g, HEIGHT_GRAY, "pixel {i}");
    }

    // the default radius reaches tens of points, so a min_points of 1000
    // leaves the tile nodata but for the handful of cells a point sits on,
    // which take their height whatever min_points says
    assert!(
        grays["sparse"].len() * 100 < grays["heights"].len(),
        "{} of {} pixels reached 1000 points",
        grays["sparse"].len(),
        grays["heights"].len()
    );

    // a quarter-pixel radius is under the point spacing, so only the cells
    // that happen to sit next to a point get a value
    assert!(!grays["pinpoint"].is_empty());
    assert!(
        grays["pinpoint"].len() * 2 < grays["heights"].len(),
        "{} of {} pixels found a point within a quarter of a pixel",
        grays["pinpoint"].len(),
        grays["heights"].len()
    );

    let (status, body) = get(&app, "/layers").await;
    assert_eq!(status, StatusCode::OK);
    let layers: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(layers[0]["source"], "las");
    assert!(layers[0]["temporal_extent"].is_null());
}

#[tokio::test]
async fn a_copc_layer_grids_its_points_into_the_tile() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&copc_layer_file(dir.path()));
    let (x, y) = tile_of(12, 7.3, 46.8);
    let (status, png) = get(&app, &format!("/tiles/copc-heights/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    let grays = opaque_grays(&png);
    assert!(grays.len() > 60_000, "only {} pixels gridded", grays.len());
    for (i, g) in grays.iter().enumerate() {
        assert_eq!(*g, HEIGHT_GRAY, "pixel {i}");
    }

    let (status, body) = get(&app, "/layers").await;
    assert_eq!(status, StatusCode::OK);
    let layers: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(layers[0]["source"], "copc");
}

#[test]
fn a_copc_source_names_exactly_one_of_path_and_url() {
    let both = COPC.replace(
        "path = \"COPC_PATH\"",
        "path = \"/data/heights.copc.laz\", url = \"http://127.0.0.1:1/heights.copc.laz\"",
    );
    let neither = COPC.replace("path = \"COPC_PATH\", ", "");
    for text in [both, neither] {
        let error = Config::parse(&text).expect_err("a copc source needs one location");
        assert!(error.contains("layer copc-heights"), "{error}");
        assert!(error.contains("exactly one of path and url"), "{error}");
    }
}

#[test]
fn config_parses_a_las_source_and_the_idw_over_it() {
    let dir = tempfile::tempdir().unwrap();
    let config = Config::parse(&las_layer_file(dir.path())).unwrap();
    match source_of(&config.layers[0]) {
        SourceConfig::Las { path, crs } => {
            assert!(path.ends_with("heights.las"), "{path:?}");
            assert_eq!(*crs, 3857);
        }
        other => panic!("unexpected source: {other:?}"),
    }
    assert!(source_of(&config.layers[0]).stac_search().is_none());

    // an idw naming no field takes every one of the element's defaults
    match &config.layers[0].ops[..] {
        [
            OpConfig::Idw {
                power,
                radius_px,
                min_points,
            },
        ] => {
            assert_eq!(*power, None);
            assert_eq!(*radius_px, None);
            assert_eq!(*min_points, None);
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    match &config.layers[2].ops[..] {
        [
            OpConfig::Idw {
                power,
                radius_px,
                min_points,
            },
        ] => {
            assert_eq!(*power, Some(3.0));
            assert_eq!(*radius_px, Some(0.25));
            assert_eq!(*min_points, None);
        }
        other => panic!("unexpected ops: {other:?}"),
    }
}

#[test]
fn an_idw_layer_has_to_name_its_own_gray_range() {
    let text = r#"
[[layer]]
name = "heights"
source = { kind = "las", path = "/a.las", crs = 3857 }

[[layer.op]]
kind = "idw"
"#;
    let error = Config::parse(text).expect_err("interpolated heights have no range of their own");
    assert!(error.contains("gray"), "{error}");

    let with_range = text.replace(
        "[[layer.op]]",
        "gray = { min = 0.0, max = 300.0 }\n\n[[layer.op]]",
    );
    Config::parse(&with_range).expect("a named gray range makes it servable");
}

const CONTOURS: &str = r#"
[[layer]]
name = "contour-lines"
source = { kind = "cog", path = "COG_PATH" }

[[layer.op]]
kind = "contours"
interval = 10.0
"#;

#[tokio::test]
async fn a_layer_ending_in_contours_serves_its_lines_as_mvt() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&CONTOURS.replace("COG_PATH", &write_dem(dir.path())));
    let (x, y) = tile_of(12, 7.3, 46.8);

    let (status, body) = get(&app, &format!("/tiles/contour-lines/12/{x}/{y}.mvt")).await;
    assert_eq!(status, StatusCode::OK);
    let layer = decoded_layer(&body, "contour-lines");
    assert!(layer.features.len() >= 2, "{:?}", layer.features);
    for feature in &layer.features {
        assert!(
            matches!(feature.geometry, TileGeometry::Lines(_)),
            "{:?}",
            feature.geometry
        );
        match feature.attributes.get("elevation") {
            Some(AttributeValue::Number(elevation)) => {
                assert_eq!(elevation % 10.0, 0.0, "elevation {elevation}")
            }
            Some(AttributeValue::Integer(elevation)) => {
                assert_eq!(elevation % 10, 0, "elevation {elevation}")
            }
            other => panic!("elevation is {other:?}"),
        }
    }

    let (status, _) = get(&app, &format!("/tiles/contour-lines/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[test]
fn chains_ending_in_features_parse_without_gray_and_burned_ones_need_it() {
    let text = r#"
[[layer]]
name = "classes-as-polygons"
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.op]]
kind = "reclassify"
classes = [
    { min = 0.0, max = 350.0, value = 1.0 },
    { min = 350.0, max = 1000.0, value = 2.0 },
]

[[layer.op]]
kind = "polygonize"

[[layer]]
name = "point-heights-as-contours"
source = { kind = "las", path = "/data/points.las", crs = 3857 }

[[layer.op]]
kind = "idw"

[[layer.op]]
kind = "contours"
interval = 10.0

[[layer]]
name = "mosaic-as-contours"
fanin = { kind = "mosaic" }

[[layer.input]]
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.input]]
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.op]]
kind = "contours"
interval = 10.0
"#;
    let config = Config::parse(text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(config.layers.len(), 3);

    let burned = r#"
[[layer]]
name = "burned-contours"
source = { kind = "cog", path = "/data/dem.tif" }

[[layer.op]]
kind = "contours"
interval = 10.0

[[layer.op]]
kind = "rasterize"
burn = { property = "elevation" }
"#;
    let error = Config::parse(burned).expect_err("burned contours carry no range");
    assert!(error.contains("gray"), "{error}");
}

const VIEWSHED: &str = r#"
[[layer]]
name = "seen"
source = { kind = "cog", path = "COG_PATH" }

[[layer.op]]
kind = "viewshed"
lon = 7.3
lat = 46.8
height = 2.0
radius = 0.05
"#;

#[tokio::test]
async fn a_viewshed_layer_serves_only_hidden_and_visible() {
    let dir = tempfile::tempdir().unwrap();
    let text = VIEWSHED.replace("COG_PATH", &write_dem(dir.path()));
    let (x, y) = tile_of(12, 7.3, 46.8);
    let (status, png) = get(&app(&text), &format!("/tiles/seen/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    let grays = opaque_grays(&png);
    assert!(
        !grays.is_empty(),
        "the tile around the observer is all nodata"
    );
    assert!(grays.iter().all(|g| *g == 0 || *g == 255), "{grays:?}");

    let config = Config::parse(&text).unwrap();
    assert_eq!(config.layers[0].gray_range().unwrap(), (0.0, 1.0));
    match &config.layers[0].ops[..] {
        [
            OpConfig::Viewshed {
                lon,
                lat,
                height,
                radius,
            },
        ] => assert_eq!((*lon, *lat, *height, *radius), (7.3, 46.8, 2.0, 0.05)),
        other => panic!("unexpected ops: {other:?}"),
    }
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
    // features burned into pixels serve as a raster
    assert_eq!(named("parcels")["kind"], "raster");
    assert!(named("parcels")["collection"].is_null());
    assert!(named("parcels")["temporal_extent"].is_null());
}

/// the dem as a cog layer and as a gdal layer over a striped copy, so the
/// two tiles can be compared
const GDAL_BESIDE_COG: &str = r#"
[[layer]]
name = "dem"
source = { kind = "cog", path = "COG_PATH" }

[[layer.op]]
kind = "hillshade"
azimuth = 315.0
altitude = 45.0

[[layer]]
name = "dem-gdal"
source = { kind = "gdal", path = "GDAL_PATH", band = 1 }

[[layer.op]]
kind = "hillshade"
azimuth = 315.0
altitude = 45.0
"#;

#[cfg(feature = "gdal")]
#[test]
fn config_parses_a_gdal_layer_and_its_band() {
    let config = Config::parse(GDAL_BESIDE_COG).unwrap();
    match source_of(&config.layers[1]) {
        SourceConfig::Gdal { path, band, .. } => {
            assert_eq!(path, Path::new("GDAL_PATH"));
            assert_eq!(*band, Some(1));
        }
        other => panic!("expected a gdal source, got {other:?}"),
    }
}

#[cfg(feature = "gdal")]
#[test]
fn config_parses_a_gdal_layers_value_kind() {
    let config =
        Config::parse(&GDAL_BESIDE_COG.replace("band = 1", "value_kind = \"categorical\""))
            .unwrap();
    assert!(matches!(
        source_of(&config.layers[1]),
        SourceConfig::Gdal {
            band: None,
            value_kind: ValueKindConfig::Categorical,
            ..
        }
    ));
    assert!(matches!(
        source_of(&Config::parse(GDAL_BESIDE_COG).unwrap().layers[1]),
        SourceConfig::Gdal {
            value_kind: ValueKindConfig::Continuous,
            ..
        }
    ));
}

#[cfg(not(feature = "gdal"))]
#[test]
fn a_gdal_layer_fails_to_parse_without_the_feature() {
    let message = Config::parse(GDAL_BESIDE_COG).unwrap_err();
    assert!(message.contains("layer dem-gdal"), "{message}");
    assert!(message.contains("gdal feature"), "{message}");
}

/// the dem as a striped geotiff. gdal reads no epsg code from a terrano
/// cog, whose geokeys lack the model type
#[cfg(feature = "gdal")]
fn write_striped_dem(dir: &Path) -> String {
    let raster = Raster::from_vec(W, H, dem_values(), CELL, f64::NAN).unwrap();
    let meta = terrano_core::GeoTiffMetadata {
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        epsg: 4326,
    };
    let mut bytes = Vec::new();
    terrano_core::write_geotiff_bands(
        &BandedRaster::new(vec![raster]).unwrap(),
        &meta,
        terrano_core::SampleFormat::F64,
        &mut bytes,
    )
    .unwrap();
    let path = dir.join("striped.tif");
    std::fs::write(&path, bytes).unwrap();
    path.display().to_string().replace('\\', "/")
}

#[cfg(feature = "gdal")]
#[tokio::test]
async fn a_gdal_layer_serves_the_tile_the_cog_layer_does() {
    let dir = tempfile::tempdir().unwrap();
    let text = GDAL_BESIDE_COG
        .replace("COG_PATH", &write_dem(dir.path()))
        .replace("GDAL_PATH", &write_striped_dem(dir.path()));
    let app = app(&text);
    let (x, y) = tile_of(12, 7.3, 46.8);

    let (status, cog) = get(&app, &format!("/tiles/dem/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, gdal) = get(&app, &format!("/tiles/dem-gdal/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(
        opaque_pixels(&gdal) > 0,
        "the tile rendered nothing but nodata"
    );
    assert_eq!(png_pixels(&gdal), png_pixels(&cog));

    let (_, body) = get(&app, "/layers").await;
    let layers: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(layers[1]["name"], "dem-gdal");
    assert_eq!(layers[1]["source"], "gdal");
}

/// the gray range the two-band layer stretches band 0 over, the dem's own
/// range across the rendered tile
const BANDED_GRAY: (f64, f64) = (300.0, 400.0);

fn read_tif(bytes: &[u8]) -> (BandedRaster, GeoTiffMetadata) {
    let reader = CogReader::open(bytes).expect("terrano reads back what it wrote");
    let base = &reader.levels()[0];
    let bands = reader
        .read_window_bands(0, 0, 0, base.width, base.height)
        .unwrap();
    (bands, reader.meta().clone())
}

#[tokio::test]
async fn the_same_tile_is_served_as_png_or_as_geotiff() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("bands.tif");
    std::fs::write(&path, banded_dem_cog()).unwrap();
    let text = format!(
        r#"
[[layer]]
name = "bands"
source = {{ kind = "cog", path = "{}" }}
gray = {{ min = {}, max = {} }}
"#,
        path.display().to_string().replace('\\', "/"),
        BANDED_GRAY.0,
        BANDED_GRAY.1,
    );
    let app = app(&text);
    let (x, y) = tile_of(12, 7.3, 46.8);

    // the suffixless path stays png, as does the one naming it
    let (status, bare_type, bare) = get_typed(&app, &format!("/tiles/bands/12/{x}/{y}")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(bare_type, "image/png");
    let (status, png_type, png) = get_typed(&app, &format!("/tiles/bands/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(png_type, "image/png");
    assert_eq!(bare, png);

    let (status, tif_type, tif) = get_typed(&app, &format!("/tiles/bands/12/{x}/{y}.tif")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(tif_type, "image/tiff");

    let (bands, meta) = read_tif(&tif);
    assert_eq!(bands.band_count(), 2, "the second band did not survive");
    assert_eq!(bands.width(), 256);
    assert_eq!(bands.height(), 256);

    // the tile is reprojected before it is encoded, so the geotiff carries
    // the web mercator corner and pixel size of the tile that was asked for
    let (min_x, _, max_x, max_y) = tile_bounds(12, x, y);
    let pixel = (max_x - min_x) / 256.0;
    assert_eq!(meta.epsg, 3857);
    assert!((meta.origin_x - min_x).abs() < pixel, "{}", meta.origin_x);
    assert!((meta.origin_y - max_y).abs() < pixel, "{}", meta.origin_y);
    assert!((meta.pixel_width - pixel).abs() < pixel / 100.0);
    assert!((meta.pixel_height - pixel).abs() < pixel / 100.0);

    // both bands warped together, so the lift between them is what it was
    // in the source file
    let base = bands.band(0).unwrap().data();
    let lifted = bands.band(1).unwrap().data();
    let mut checked = 0;
    for (i, (v, w)) in base.iter().zip(lifted).enumerate() {
        if !v.is_finite() {
            continue;
        }
        assert!(
            (w - v - BAND_LIFT).abs() < 1e-6,
            "pixel {i}: {v} against {w}"
        );
        checked += 1;
    }
    assert!(checked > 60_000, "only {checked} pixels held a value");

    // the png is the same chunk's band 0 stretched to gray, so the two
    // encodings agree pixel by pixel
    let pixels = png_pixels(&png);
    assert_eq!(pixels.len(), base.len());
    let span = BANDED_GRAY.1 - BANDED_GRAY.0;
    for (i, ((gray, alpha), v)) in pixels.iter().zip(base).enumerate() {
        if !v.is_finite() {
            assert_eq!(*alpha, 0, "pixel {i} is nodata in the geotiff");
            continue;
        }
        assert_eq!(*alpha, 255, "pixel {i}");
        let want = (((v - BANDED_GRAY.0) / span).clamp(0.0, 1.0) * 255.0).round() as u8;
        assert_eq!(*gray, want, "pixel {i} holds {v}");
    }
}

#[tokio::test]
async fn malformed_tile_requests_are_rejected_not_rendered() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&cog_layer_file(dir.path()));

    let cases = [
        ("/tiles/nope/12/2131/1443.png", StatusCode::NOT_FOUND),
        ("/tiles/dem/12/2131/north.png", StatusCode::BAD_REQUEST),
        ("/tiles/dem/12/2131/north.tif", StatusCode::BAD_REQUEST),
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

const NETCDF_AND_GRIB: &str = r#"
[[layer]]
name = "sst"
source = { kind = "netcdf", path = "/data/sst.nc", variable = "sst", datetime = "2024-01-01T00:00:00Z/2024-01-02T00:00:00Z", select = { zlev = 0 } }

[[layer]]
name = "t2m"
source = { kind = "grib", path = "/data/gsm.grib2", message = 5 }
"#;

#[test]
fn config_parses_netcdf_and_grib_layers() {
    let config = Config::parse(NETCDF_AND_GRIB).unwrap();
    match source_of(&config.layers[0]) {
        SourceConfig::Netcdf {
            variable, select, ..
        } => {
            assert_eq!(variable, "sst");
            assert_eq!(select.get("zlev"), Some(&0));
        }
        other => panic!("expected a netcdf source, got {other:?}"),
    }
    match source_of(&config.layers[1]) {
        SourceConfig::Grib { message, .. } => assert_eq!(*message, 5),
        other => panic!("expected a grib source, got {other:?}"),
    }
}

#[test]
fn netcdf_and_grib_layers_refuse_a_bad_datetime_and_message_zero() {
    let bad_datetime = NETCDF_AND_GRIB.replace("2024-01-02T00:00:00Z", "tomorrow");
    let message = Config::parse(&bad_datetime).unwrap_err();
    assert!(message.contains("layer sst"), "{message}");
    assert!(message.contains("datetime"), "{message}");
    let zero = NETCDF_AND_GRIB.replace("message = 5", "message = 0");
    let message = Config::parse(&zero).unwrap_err();
    assert!(message.contains("layer t2m"), "{message}");
    assert!(message.contains("count from 1"), "{message}");
}

const VECTOR_LAYERS: &str = r#"
[[layer]]
name = "parcel-features"
source = { kind = "geojson", path = "GEOJSON_PATH" }

[[layer]]
name = "shallow-features"
source = { kind = "geojson", path = "GEOJSON_PATH" }

[[layer.op]]
kind = "vec_filter"
field = "zone"
equals = "shallow"
"#;

fn vector_layer_file(dir: &Path) -> String {
    VECTOR_LAYERS.replace("GEOJSON_PATH", &write_parcels(dir))
}

const MVT_EXTENT: f64 = 4096.0;
const MVT_BUFFER: f64 = 64.0;

fn tile_units(z: u8, x: u32, y: u32, lon: f64, lat: f64) -> [f64; 2] {
    let radius = HALF_WORLD / std::f64::consts::PI;
    let mercator_x = lon.to_radians() * radius;
    let mercator_y = (std::f64::consts::FRAC_PI_4 + lat.to_radians() / 2.0)
        .tan()
        .ln()
        * radius;
    let (min_x, min_y, max_x, max_y) = tile_bounds(z, x, y);
    [
        (mercator_x - min_x) / (max_x - min_x) * MVT_EXTENT,
        (max_y - mercator_y) / (max_y - min_y) * MVT_EXTENT,
    ]
}

fn decoded_layer(body: &[u8], name: &str) -> VectorLayer {
    let tile = decode_tile(body).expect("an mvt tile");
    assert_eq!(tile.layers.len(), 1, "{:?}", tile.layers);
    tile.layer(name)
        .unwrap_or_else(|| panic!("no mvt layer {name}"))
        .clone()
}

fn zones(layer: &VectorLayer) -> Vec<String> {
    let mut zones: Vec<String> = layer
        .features
        .iter()
        .map(|feature| match feature.attributes.get("zone") {
            Some(AttributeValue::String(zone)) => zone.clone(),
            other => panic!("zone is {other:?}"),
        })
        .collect();
    zones.sort();
    zones
}

fn exterior_bounds(geometry: &TileGeometry) -> [f64; 4] {
    let TileGeometry::Polygons(polygons) = geometry else {
        panic!("not a polygon: {geometry:?}");
    };
    let mut bounds = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for vertex in polygons.iter().flat_map(|polygon| &polygon.exterior) {
        let (x, y) = (f64::from(vertex[0]), f64::from(vertex[1]));
        bounds = [
            bounds[0].min(x),
            bounds[1].min(y),
            bounds[2].max(x),
            bounds[3].max(y),
        ];
    }
    bounds
}

fn assert_near(got: f64, want: f64, what: &str) {
    // jung-mvt rounds coordinates to whole tile units
    assert!((got - want).abs() <= 1.0, "{what}: {got} against {want}");
}

#[tokio::test]
async fn a_vector_layer_serves_its_filtered_features_as_mvt() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&vector_layer_file(dir.path()));
    // the tile holding the shallow parcel's north east corner
    let (x, y) = tile_of(12, 7.3, 46.83);

    let (status, content_type, body) =
        get_typed(&app, &format!("/tiles/parcel-features/12/{x}/{y}.mvt")).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(content_type, "application/vnd.mapbox-vector-tile");
    let whole = decoded_layer(&body, "parcel-features");
    assert_eq!(zones(&whole), ["deep", "shallow"]);

    let uri = format!("/tiles/shallow-features/12/{x}/{y}.mvt");
    let (status, body) = get(&app, &uri).await;
    assert_eq!(status, StatusCode::OK);
    let filtered = decoded_layer(&body, "shallow-features");
    assert_eq!(zones(&filtered), ["shallow"]);
    let feature = &filtered.features[0];
    assert_eq!(
        feature.attributes.get("depth"),
        Some(&AttributeValue::Number(1.0))
    );

    // the parcel runs past the tile west and south, so those sides stop
    // at the buffer, and its east and north edges fall inside the tile
    let [min_x, min_y, max_x, max_y] = exterior_bounds(&feature.geometry);
    let [east, north] = tile_units(12, x, y, 7.35, 46.85);
    assert!(east > 0.0 && east < MVT_EXTENT && north > 0.0 && north < MVT_EXTENT);
    assert_near(min_x, -MVT_BUFFER, "west edge");
    assert_near(max_x, east, "east edge");
    assert_near(min_y, north, "north edge");
    assert_near(max_y, MVT_EXTENT + MVT_BUFFER, "south edge");

    let (status, timed) = get(
        &app,
        &format!("{uri}?t=2024-06-01T00:00:00Z/2024-07-01T00:00:00Z"),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(timed, body);
    let (status, _) = get(&app, &format!("{uri}?t=yesterday")).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn a_polygon_split_across_engine_chunks_comes_back_as_one_feature() {
    let dir = tempfile::tempdir().unwrap();
    let region = serde_json::json!({
        "type": "Feature",
        "properties": { "name": "region" },
        "geometry": {
            "type": "Polygon",
            "coordinates": [box_ring(6.9, 46.4, 7.7, 47.1)],
        },
    });
    let path = write_geojson(dir.path(), "region.geojson", region.to_string());
    let text = format!(
        r#"
[[layer]]
name = "region"
source = {{ kind = "geojson", path = "{path}" }}
"#
    );
    let app = app(&text);
    // at z10 the pull snaps to ladder level 1 over the 55 m vertex spacing, 28 km chunks under a 39 km tile
    let (x, y) = tile_of(10, 7.3, 46.75);

    let (status, body) = get(&app, &format!("/tiles/region/10/{x}/{y}.mvt")).await;
    assert_eq!(status, StatusCode::OK);
    let layer = decoded_layer(&body, "region");
    assert_eq!(layer.features.len(), 1, "{:?}", layer.features);
    let TileGeometry::Polygons(polygons) = &layer.features[0].geometry else {
        panic!("not a polygon");
    };
    assert_eq!(polygons.len(), 1, "{polygons:?}");
    assert!(polygons[0].holes.is_empty(), "{polygons:?}");
    // the region covers the whole buffered tile
    let [min_x, min_y, max_x, max_y] = exterior_bounds(&layer.features[0].geometry);
    assert_near(min_x, -MVT_BUFFER, "west edge");
    assert_near(min_y, -MVT_BUFFER, "north edge");
    assert_near(max_x, MVT_EXTENT + MVT_BUFFER, "east edge");
    assert_near(max_y, MVT_EXTENT + MVT_BUFFER, "south edge");
}

#[tokio::test]
async fn array_and_object_properties_go_out_as_json_text_and_nulls_are_left_out() {
    let dir = tempfile::tempdir().unwrap();
    let region = serde_json::json!({
        "type": "Feature",
        "properties": {
            "name": "region",
            "count": 3,
            "tags": ["a", "b"],
            "owner": { "id": 7 },
            "note": null,
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [box_ring(7.2, 46.7, 7.4, 46.9)],
        },
    });
    let path = write_geojson(dir.path(), "region.geojson", region.to_string());
    let text = format!(
        r#"
[[layer]]
name = "region"
source = {{ kind = "geojson", path = "{path}" }}
"#
    );
    let app = app(&text);
    let (x, y) = tile_of(10, 7.3, 46.8);

    let (status, body) = get(&app, &format!("/tiles/region/10/{x}/{y}.mvt")).await;
    assert_eq!(status, StatusCode::OK);
    let layer = decoded_layer(&body, "region");
    assert_eq!(layer.features.len(), 1, "{:?}", layer.features);
    let attributes = &layer.features[0].attributes;
    let text_of = |key: &str| match attributes.get(key) {
        Some(AttributeValue::String(text)) => text.clone(),
        other => panic!("{key} is {other:?}"),
    };
    assert_eq!(text_of("name"), "region");
    assert_eq!(attributes.get("count"), Some(&AttributeValue::Integer(3)));
    assert_eq!(text_of("tags"), r#"["a","b"]"#);
    assert_eq!(text_of("owner"), r#"{"id":7}"#);
    assert!(!attributes.contains_key("note"), "{attributes:?}");
}

#[tokio::test]
async fn a_layer_serves_only_its_own_kind_of_tile_and_zonal_takes_rasters() {
    let dir = tempfile::tempdir().unwrap();
    let text = format!(
        "{}{}",
        cog_layer_file(dir.path()),
        vector_layer_file(dir.path())
    );
    let app = app(&text);
    let (x, y) = tile_of(12, 7.3, 46.83);

    let (status, body) = get(&app, &format!("/tiles/dem/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(opaque_pixels(&body) > 0);
    let refused = [
        (format!("/tiles/dem/12/{x}/{y}.mvt"), "raster"),
        (format!("/tiles/parcel-features/12/{x}/{y}"), "vector"),
        (format!("/tiles/parcel-features/12/{x}/{y}.png"), "vector"),
        (format!("/tiles/parcel-features/12/{x}/{y}.tif"), "vector"),
    ];
    for (uri, kind) in refused {
        let (status, body) = get(&app, &uri).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{uri}");
        let message = String::from_utf8(body).unwrap();
        assert!(message.contains(kind), "{uri}: {message}");
    }

    let window = serde_json::json!({
        "type": "FeatureCollection",
        "features": [],
        "bbox": [7.30, 46.81, 7.32, 46.83],
        "resolution": 100.0,
    });
    let mut series = window.clone();
    series["steps"] = serde_json::json!(["2024-06-01T00:00:00Z/2024-07-01T00:00:00Z"]);
    for (uri, body) in [
        ("/zonal/parcel-features", &window),
        ("/zonal/parcel-features/series", &series),
    ] {
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header(axum::http::header::CONTENT_TYPE, "application/json")
                    .body(Body::from(serde_json::to_vec(body).unwrap()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST, "{uri}");
        let message = axum::body::to_bytes(response.into_body(), 1 << 20)
            .await
            .unwrap();
        let message = String::from_utf8(message.to_vec()).unwrap();
        assert!(message.contains("vector"), "{uri}: {message}");
    }

    let (status, body) = get(&app, "/layers").await;
    assert_eq!(status, StatusCode::OK);
    let layers: serde_json::Value = serde_json::from_slice(&body).unwrap();
    let kinds: HashMap<&str, &str> = layers
        .as_array()
        .unwrap()
        .iter()
        .map(|l| (l["name"].as_str().unwrap(), l["kind"].as_str().unwrap()))
        .collect();
    assert_eq!(kinds["dem"], "raster");
    assert_eq!(kinds["parcel-features"], "vector");
    assert_eq!(kinds["shallow-features"], "vector");
    assert_eq!(layers[1]["source"], "geojson", "{layers}");
}

#[test]
fn a_vector_layer_takes_no_gray_and_a_burned_one_still_needs_one() {
    let dir = tempfile::tempdir().unwrap();
    let vector = vector_layer_file(dir.path());
    let config = Config::parse(&vector).expect("a vector layer needs no gray");
    Layer::build_all(config, 64 << 20, None).expect("a vector layer builds without gray");

    let with_gray = vector.replacen(
        "name = \"parcel-features\"",
        "name = \"parcel-features\"\ngray = { min = 0.0, max = 4.0 }",
        1,
    );
    let config = Config::parse(&with_gray).unwrap();
    let error = match Layer::build_all(config, 64 << 20, None) {
        Ok(_) => panic!("a vector layer took a gray range"),
        Err(error) => error,
    };
    assert!(error.contains("parcel-features"), "{error}");
    assert!(error.contains("gray"), "{error}");

    let burned = GEOJSON
        .replace("GEOJSON_PATH", "/parcels.geojson")
        .replace("gray = { min = 0.0, max = 4.0 }\n", "");
    let error = Config::parse(&burned).expect_err("burned values carry no range");
    assert!(error.contains("gray"), "{error}");
}

const HYDROLOGY: &str = r#"
[[layer]]
name = "directions"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 0.0, max = 255.0 }

[[layer.op]]
kind = "flow_direction"
crs = "EPSG:32632"
region = REGION

[[layer]]
name = "accumulation"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 0.0, max = 1000000.0 }

[[layer.op]]
kind = "flow_accumulation"
crs = "EPSG:32632"
region = REGION

[[layer]]
name = "basins"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 0.0, max = 100.0 }

[[layer.op]]
kind = "watershed"
crs = "EPSG:32632"
region = REGION

[[layer]]
name = "streams"
source = { kind = "cog", path = "COG_PATH" }
gray = { min = 0.0, max = 5.0 }

[[layer.op]]
kind = "stream_order"
crs = "EPSG:32632"
region = REGION
threshold = 100000.0
"#;

const HYDROLOGY_LON: (f64, f64) = (7.31, 7.36);
const HYDROLOGY_LAT: (f64, f64) = (46.75, 46.79);

fn hydrology_region() -> [f64; 4] {
    let to_utm = projicio_core::Transform::new("EPSG:4326", "EPSG:32632").unwrap();
    let mut region = [
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    ];
    for lon in [HYDROLOGY_LON.0, HYDROLOGY_LON.1] {
        for lat in [HYDROLOGY_LAT.0, HYDROLOGY_LAT.1] {
            let (x, y) = to_utm.convert(lon, lat).unwrap();
            region = [
                region[0].min(x),
                region[1].min(y),
                region[2].max(x),
                region[3].max(y),
            ];
        }
    }
    region
}

fn hydrology_layer_file(cog_path: &str) -> String {
    let [min_x, min_y, max_x, max_y] = hydrology_region();
    HYDROLOGY.replace("COG_PATH", cog_path).replace(
        "REGION",
        &format!("[{min_x:?}, {min_y:?}, {max_x:?}, {max_y:?}]"),
    )
}

fn hydrology_tile() -> (u32, u32) {
    tile_of(
        12,
        (HYDROLOGY_LON.0 + HYDROLOGY_LON.1) / 2.0,
        (HYDROLOGY_LAT.0 + HYDROLOGY_LAT.1) / 2.0,
    )
}

async fn tile_band(app: &Router, uri: &str) -> Vec<f64> {
    let (status, body) = get(app, uri).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    let (bands, _) = read_tif(&body);
    bands.band(0).unwrap().data().to_vec()
}

#[test]
fn config_parses_the_hydrology_ops_and_each_needs_a_gray_range() {
    let text = hydrology_layer_file("/data/dem.tif");
    let config = Config::parse(&text).unwrap_or_else(|e| panic!("{e}"));
    let region = hydrology_region();
    let utm = geoplumb::Crs(32632);
    match &config.layers[0].ops[..] {
        [OpConfig::FlowDirection { crs, region: got }] => {
            assert_eq!((crs.0, *got), (utm, region));
        }
        other => panic!("unexpected ops: {other:?}"),
    }
    match &config.layers[1].ops[..] {
        [OpConfig::FlowAccumulation { crs, region: got }] => {
            assert_eq!((crs.0, *got), (utm, region));
        }
        other => panic!("unexpected ops: {other:?}"),
    }
    match &config.layers[2].ops[..] {
        [OpConfig::Watershed { crs, region: got }] => {
            assert_eq!((crs.0, *got), (utm, region));
        }
        other => panic!("unexpected ops: {other:?}"),
    }
    match &config.layers[3].ops[..] {
        [
            OpConfig::StreamOrder {
                crs,
                region: got,
                threshold,
            },
        ] => {
            assert_eq!((crs.0, *got, *threshold), (utm, region, 100000.0));
        }
        other => panic!("unexpected ops: {other:?}"),
    }

    for gray in [
        "gray = { min = 0.0, max = 255.0 }\n",
        "gray = { min = 0.0, max = 1000000.0 }\n",
        "gray = { min = 0.0, max = 100.0 }\n",
        "gray = { min = 0.0, max = 5.0 }\n",
    ] {
        let error = Config::parse(&text.replacen(gray, "", 1))
            .expect_err("a hydrology layer took a gray range from its source");
        assert!(error.contains("gray"), "{error}");
    }

    let error = Config::parse(&text.replace("EPSG:32632", "32632"))
        .expect_err("a crs without its authority parsed");
    assert!(error.contains("EPSG:<code>"), "{error}");
}

#[tokio::test]
async fn every_hydrology_op_serves_a_tile_cut_from_its_region() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&hydrology_layer_file(&write_dem(dir.path())));
    let (x, y) = hydrology_tile();
    let finite = |values: &[f64]| -> Vec<f64> {
        let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
        assert!(!finite.is_empty(), "the tile holds no region cell");
        assert!(
            finite.len() < values.len(),
            "the tile holds no cell outside the region"
        );
        finite
    };

    let directions = tile_band(&app, &format!("/tiles/directions/12/{x}/{y}.tif")).await;
    let codes = [0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0];
    let directions = finite(&directions);
    assert!(
        directions.iter().all(|code| codes.contains(code)),
        "{directions:?}"
    );
    assert!(
        directions.iter().any(|code| *code != 0.0),
        "every cell is a pit"
    );

    let accumulation = tile_band(&app, &format!("/tiles/accumulation/12/{x}/{y}.tif")).await;
    let accumulation = finite(&accumulation);
    let smallest = accumulation.iter().copied().fold(f64::INFINITY, f64::min);
    let largest = accumulation.iter().copied().fold(0.0, f64::max);
    assert!(smallest > 0.0, "{smallest}");
    assert!(
        largest > 10.0 * smallest,
        "no cell gathers the flow of many: {smallest} to {largest}"
    );

    let basins = tile_band(&app, &format!("/tiles/basins/12/{x}/{y}.tif")).await;
    let basins = finite(&basins);
    assert!(
        basins
            .iter()
            .all(|label| *label >= 1.0 && label.fract() == 0.0),
        "{basins:?}"
    );

    let (status, body) = get(&app, &format!("/tiles/streams/12/{x}/{y}.tif")).await;
    assert_eq!(status, StatusCode::OK);
    let (bands, _) = read_tif(&body);
    let orders: Vec<f64> = bands
        .band(0)
        .unwrap()
        .data()
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .collect();
    assert!(!orders.is_empty(), "no cell gathers 100000 square metres");
    assert!(
        orders
            .iter()
            .all(|order| *order >= 1.0 && order.fract() == 0.0),
        "{orders:?}"
    );

    let (status, png) = get(&app, &format!("/tiles/directions/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    assert!(opaque_pixels(&png) > 0);
}

#[tokio::test]
async fn a_region_over_the_pixel_cap_answers_400_on_tiles_and_exports() {
    let dir = tempfile::tempdir().unwrap();
    let config = Config::parse(&hydrology_layer_file(&write_dem(dir.path()))).unwrap();
    // about 3000 region cells at level 0 against a cap of a few hundred
    let layers = Layer::build_all(config, 16 << 10, None).expect("the layers must build");
    let app = router(layers, Some(dir.path().to_path_buf()));

    let (x, y) = hydrology_tile();
    let (status, body) = get(&app, &format!("/tiles/accumulation/12/{x}/{y}.png")).await;
    let message = String::from_utf8_lossy(&body);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{message}");
    assert!(message.contains("pixel cap"), "{message}");

    let [min_x, min_y, max_x, max_y] = hydrology_region();
    let (status, body) = get(
        &app,
        &format!(
            "/export/accumulation?bbox={min_x},{min_y},{max_x},{max_y}&crs=EPSG:32632&resolution=30"
        ),
    )
    .await;
    let message = String::from_utf8_lossy(&body);
    assert_eq!(status, StatusCode::BAD_REQUEST, "{message}");
    assert!(message.contains("pixel cap"), "{message}");
}

const CAST_STRIPES: [f64; 6] = [-5.0, 2.5, 300.0, 100.0, f64::NAN, 7.4];

fn write_cast_stripes(dir: &Path) -> String {
    let data = (0..W * H)
        .map(|index| CAST_STRIPES[(index % W) / (W / CAST_STRIPES.len())])
        .collect();
    let raster = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
    let params = CogParams {
        // terrano averages the overviews it writes
        overview_levels: 0,
        ..cog_params()
    };
    let path = dir.join("stripes.tif");
    let mut file = std::fs::File::create(&path).unwrap();
    write_cog(&raster, &params, &mut file).unwrap();
    path.display().to_string().replace('\\', "/")
}

const CAST_LAYER: &str = r#"
[[layer]]
name = "narrowed"
source = { kind = "cog", path = "STRIPES_PATH", value_kind = "categorical" }
gray = { min = 0.0, max = 255.0 }

[[layer.op]]
kind = "cast"
dtype = "u8"
NODATA
"#;

#[tokio::test]
async fn a_cast_op_narrows_rounds_and_saturates_into_its_dtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_cast_stripes(dir.path());
    let app = app(&CAST_LAYER
        .replace("STRIPES_PATH", &path)
        .replace("NODATA", "nodata = 0"));
    let (x, y) = tile_of(9, 7.3, 46.8);
    let (status, tif) = get(&app, &format!("/tiles/narrowed/9/{x}/{y}.tif")).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&tif));

    let reader = CogReader::open(tif.as_slice()).unwrap();
    let base = &reader.levels()[0];
    assert_eq!(base.sample_format(), terrano_core::SampleFormat::U8);
    assert_eq!(base.nodata(), Some(0.0));
    let read = reader
        .read_window_typed(0, 0, 0, base.width, base.height)
        .unwrap();
    let terrano_core::TypedBand::U8 { samples, nodata } = &read[0] else {
        panic!("the cast tile is not u8");
    };
    assert_eq!(*nodata, Some(0));
    let codes: std::collections::BTreeSet<u8> = samples.values().iter().copied().collect();
    // -5 saturates onto the sentinel 0 and steps up to 1, 2.5 rounds away from zero, 300 saturates
    let narrowed: std::collections::BTreeSet<u8> = [0, 1, 3, 7, 100, 255].into();
    assert_eq!(codes, narrowed);
}

#[test]
fn a_cast_op_refuses_a_dtype_it_cannot_build() {
    let cases = [
        ("dtype = \"u9\"", "not one of"),
        ("dtype = \"f32\"\nnodata = 0.0", "takes no nodata"),
        ("dtype = \"u8\"\nnodata = 300.0", "not a u8 value"),
        ("dtype = \"i16\"\nnodata = 0.5", "not a i16 value"),
    ];
    for (op, why) in cases {
        let text = format!(
            "[[layer]]\nname = \"a\"\nsource = {{ kind = \"cog\", path = \"/a.tif\" }}\n\
             [[layer.op]]\nkind = \"cast\"\n{op}\n"
        );
        let error = Config::parse(&text).expect_err(op);
        assert!(error.contains(why), "{op}: {error}");
    }
}

#[test]
fn a_cast_to_an_integer_without_nodata_refuses_a_band_with_absent_cells() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_cast_stripes(dir.path());
    let config = Config::parse(
        &CAST_LAYER
            .replace("STRIPES_PATH", &path)
            .replace("NODATA", ""),
    )
    .expect("the layer file parses");
    let Err(error) = Layer::build_all(config, 64 << 20, None) else {
        panic!("a cast with no nodata built over a band with NaN cells");
    };
    assert!(error.contains("no nodata to mark absent cells"), "{error}");
}

#[tokio::test]
async fn a_cog_layer_nodata_becomes_the_sentinel_of_a_file_without_one() {
    let dir = tempfile::tempdir().unwrap();
    let data = (0..W * H)
        .map(|index| 1.0 + ((index % W) / 7 % 2) as f64)
        .collect();
    let raster = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
    let params = CogParams {
        overview_levels: 0,
        format: terrano_core::SampleFormat::U8,
        nodata: None,
        ..cog_params()
    };
    let path = dir.path().join("codes.tif");
    let mut file = std::fs::File::create(&path).unwrap();
    write_cog(&raster, &params, &mut file).unwrap();
    let app = app(&format!(
        r#"
[[layer]]
name = "codes"
source = {{ kind = "cog", path = "{}", value_kind = "categorical", nodata = 2 }}
gray = {{ min = 0.0, max = 3.0 }}
"#,
        path.display().to_string().replace('\\', "/")
    ));
    let (x, y) = tile_of(12, 7.3, 46.8);
    let (status, tif) = get(&app, &format!("/tiles/codes/12/{x}/{y}.tif")).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&tif));
    let reader = CogReader::open(tif.as_slice()).unwrap();
    let base = &reader.levels()[0];
    assert_eq!(base.sample_format(), terrano_core::SampleFormat::U8);
    assert_eq!(base.nodata(), Some(2.0));
    let read = reader
        .read_window_typed(0, 0, 0, base.width, base.height)
        .unwrap();
    let codes: std::collections::BTreeSet<u8> = match &read[0] {
        terrano_core::TypedBand::U8 { samples, .. } => samples.values().iter().copied().collect(),
        other => panic!("the tile is {}", other.format().name()),
    };
    let expected: std::collections::BTreeSet<u8> = [1, 2].into();
    assert_eq!(codes, expected);
}

#[tokio::test]
async fn pyramid_writes_every_covering_tile_as_the_server_serves_it() {
    let dir = tempfile::tempdir().unwrap();
    let text = cog_layer_file(dir.path());
    let layers = dir.path().join("layers.toml");
    std::fs::write(&layers, &text).unwrap();
    let out = dir.path().join("tiles");
    let (west, south, east, north) = (7.05, 46.7, 7.5, 46.95);
    let query = geoplumb_server::cli::PyramidQuery {
        bbox: format!("{west},{south},{east},{north}"),
        min_zoom: 8,
        max_zoom: 10,
        t: None,
    };

    let written = geoplumb_server::cli::write_pyramid(&layers, "dem", query, &out)
        .await
        .unwrap();

    let served = app(&text);
    let mut expected = 0;
    for z in 8..=10 {
        let (x0, y0) = tile_of(z, west, north);
        let (x1, y1) = tile_of(z, east, south);
        for x in x0..=x1 {
            for y in y0..=y1 {
                expected += 1;
                let file = std::fs::read(out.join(format!("{z}/{x}/{y}.png"))).unwrap();
                let (status, body) = get(&served, &format!("/tiles/dem/{z}/{x}/{y}.png")).await;
                assert_eq!(status, StatusCode::OK);
                assert_eq!(file, body, "tile {z}/{x}/{y}");
            }
        }
    }
    assert_eq!(written, expected);
}

const CLIP_CUTLINE: &str = r#"cutline = "boundary""#;

// the dem plain and clipped to the boundary layer, both over one gray range
fn clip_layer_file(dir: &Path) -> String {
    format!(
        r#"
[[layer]]
name = "boundary"
source = {{ kind = "geojson", path = "{boundary}" }}

[[layer]]
name = "dem"
source = {{ kind = "cog", path = "{dem}" }}
gray = {{ min = 300.0, max = 700.0 }}

[[layer]]
name = "clipped"
source = {{ kind = "cog", path = "{dem}" }}
gray = {{ min = 300.0, max = 700.0 }}

[[layer.op]]
kind = "clip_raster"
{CLIP_CUTLINE}
"#,
        boundary = write_geojson(dir, "boundary.geojson", boundary_geojson()),
        dem = write_dem(dir),
    )
}

#[test]
fn config_parses_a_clip_raster_op_and_checks_its_cutline_layer() {
    let dir = tempfile::tempdir().unwrap();
    let text = clip_layer_file(dir.path());
    let ops_of = |config: &Config| {
        config
            .layers
            .iter()
            .find(|l| l.name == "clipped")
            .unwrap()
            .ops
            .clone()
    };
    match &ops_of(&Config::parse(&text).unwrap())[..] {
        [OpConfig::ClipRaster { cutline, nodata }] => {
            assert_eq!(cutline, "boundary");
            assert_eq!(*nodata, None);
        }
        other => panic!("unexpected ops: {other:?}"),
    }
    let with_nodata = text.replace(CLIP_CUTLINE, &format!("{CLIP_CUTLINE}\nnodata = 255"));
    match &ops_of(&Config::parse(&with_nodata).unwrap())[..] {
        [OpConfig::ClipRaster { nodata, .. }] => assert_eq!(*nodata, Some(255.0)),
        other => panic!("unexpected ops: {other:?}"),
    }

    let missing = text.replace(CLIP_CUTLINE, r#"cutline = "parcels""#);
    let error = Config::parse(&missing).unwrap_err();
    assert!(error.contains("cutline layer parcels"), "{error}");
    let itself = text.replace(CLIP_CUTLINE, r#"cutline = "clipped""#);
    let error = Config::parse(&itself).unwrap_err();
    assert!(error.contains("clip_raster of its own"), "{error}");
}

#[tokio::test]
async fn a_clip_raster_layer_blanks_the_tile_past_its_cutline() {
    let dir = tempfile::tempdir().unwrap();
    let app = app(&clip_layer_file(dir.path()));
    let (x, y) = tile_of(12, 7.3, 46.8);
    let (status, dem) = get(&app, &format!("/tiles/dem/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    let (status, clipped) = get(&app, &format!("/tiles/clipped/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK);
    let (dem, clipped) = (png_pixels(&dem), png_pixels(&clipped));

    let tile_lon = |tile_x: f64| tile_x / f64::from(1u32 << 12) * 360.0 - 180.0;
    // three dem cells, past where the bilinear warp to web mercator reaches across the cut
    let margin = 0.003;
    let (mut kept, mut blanked) = (0, 0);
    for (index, (dem_pixel, clipped_pixel)) in dem.iter().zip(&clipped).enumerate() {
        let col = index % 256;
        let lon = tile_lon(f64::from(x) + (col as f64 + 0.5) / 256.0);
        assert_eq!(dem_pixel.1, 255, "the dem covers the whole tile");
        if lon < BOUNDARY_MAX_LON - margin {
            kept += 1;
            assert_eq!(clipped_pixel, dem_pixel, "pixel {index} at {lon} is inside");
        } else if lon > BOUNDARY_MAX_LON + margin {
            blanked += 1;
            assert_eq!(clipped_pixel.1, 0, "pixel {index} at {lon} is outside");
        }
    }
    assert!(kept > 0 && blanked > 0, "{kept} kept, {blanked} blanked");
}

// west of this longitude the relief fixture holds RELIEF_VALUE, east of it nodata
const RELIEF_EDGE_LON: f64 = 7.34;
const RELIEF_VALUE: f64 = 200.0;
const RELIEF_COLORS: &str = "100 0 0 0\n300 200 100 50\n";
// RELIEF_VALUE lies halfway between the two stops
const RELIEF_MIDPOINT: [u8; 4] = [100, 50, 25, 255];

fn write_relief_fixture(dir: &Path) -> (String, String) {
    let mut data = Vec::with_capacity(W * H);
    for _ in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            data.push(if lon < RELIEF_EDGE_LON {
                RELIEF_VALUE
            } else {
                f64::NAN
            });
        }
    }
    let raster = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
    let mut buf = std::io::Cursor::new(Vec::new());
    write_cog(&raster, &cog_params(), &mut buf).unwrap();
    let cog = dir.join("relief.tif");
    std::fs::write(&cog, buf.into_inner()).unwrap();
    let colors = dir.join("relief.txt");
    std::fs::write(&colors, RELIEF_COLORS).unwrap();
    let toml_path = |path: &Path| path.display().to_string().replace('\\', "/");
    (toml_path(&cog), toml_path(&colors))
}

fn rgba_pixels(bytes: &[u8]) -> Vec<[u8; 4]> {
    let mut reader = png::Decoder::new(bytes).read_info().expect("a png");
    let mut buf = vec![0; reader.output_buffer_size()];
    let info = reader.next_frame(&mut buf).unwrap();
    assert_eq!(info.color_type, png::ColorType::Rgba);
    buf[..info.buffer_size()].as_chunks::<4>().0.to_vec()
}

async fn relief_tile(layer_file: &str, layer: &str) -> Vec<[u8; 4]> {
    let app = app(layer_file);
    let (x, y) = tile_of(12, 7.3, 46.8);
    let (status, content_type, body) =
        get_typed(&app, &format!("/tiles/{layer}/12/{x}/{y}.png")).await;
    assert_eq!(status, StatusCode::OK, "{}", String::from_utf8_lossy(&body));
    assert_eq!(content_type, "image/png");
    rgba_pixels(&body)
}

// a z12 tile holds about 33000 pixels on each side of RELIEF_EDGE_LON
const RELIEF_SIDE_PIXELS: usize = 20_000;

#[tokio::test]
async fn a_layer_ending_in_color_relief_serves_rgba_png() {
    let dir = tempfile::tempdir().unwrap();
    let (cog, colors) = write_relief_fixture(dir.path());
    let pixels = relief_tile(
        &format!(
            r#"
[[layer]]
name = "relief"
source = {{ kind = "cog", path = "{cog}" }}

[[layer.op]]
kind = "color_relief"
color_file = "{colors}"
"#
        ),
        "relief",
    )
    .await;
    let opaque: Vec<&[u8; 4]> = pixels.iter().filter(|pixel| pixel[3] == 255).collect();
    assert!(opaque.len() > RELIEF_SIDE_PIXELS, "{} opaque", opaque.len());
    // the warp to web mercator blends colour bands only beside transparent pixels
    assert!(opaque.iter().all(|pixel| **pixel == RELIEF_MIDPOINT));
    let transparent = pixels
        .iter()
        .filter(|pixel| **pixel == [0, 0, 0, 0])
        .count();
    assert!(
        transparent > RELIEF_SIDE_PIXELS,
        "{transparent} transparent"
    );
}

#[tokio::test]
async fn a_ramp_colours_a_single_band_layer_as_rgba_png() {
    let dir = tempfile::tempdir().unwrap();
    let (cog, colors) = write_relief_fixture(dir.path());
    let pixels = relief_tile(
        &format!(
            r#"
[[layer]]
name = "ramped"
source = {{ kind = "cog", path = "{cog}" }}
ramp = "{colors}"
"#
        ),
        "ramped",
    )
    .await;
    let midpoint = pixels
        .iter()
        .filter(|pixel| **pixel == RELIEF_MIDPOINT)
        .count();
    let transparent = pixels
        .iter()
        .filter(|pixel| **pixel == [0, 0, 0, 0])
        .count();
    assert!(midpoint > RELIEF_SIDE_PIXELS, "{midpoint} coloured");
    assert!(
        transparent > RELIEF_SIDE_PIXELS,
        "{transparent} transparent"
    );
    assert_eq!(midpoint + transparent, pixels.len());
}

#[test]
fn color_relief_and_ramp_layers_refuse_what_they_cannot_colour() {
    let dir = tempfile::tempdir().unwrap();
    let (cog, colors) = write_relief_fixture(dir.path());
    let percent = dir.path().join("percent.txt");
    std::fs::write(&percent, "0% black\n100% white\n").unwrap();
    let percent = percent.display().to_string().replace('\\', "/");
    let mauve = dir.path().join("mauve.txt");
    std::fs::write(&mauve, "0 mauve\n").unwrap();
    let mauve = mauve.display().to_string().replace('\\', "/");
    let layer = |tail: String| {
        format!(
            "[[layer]]\nname = \"relief\"\nsource = {{ kind = \"cog\", path = \"{cog}\" }}\n{tail}"
        )
    };
    let relief_op =
        |file: &str| format!("\n[[layer.op]]\nkind = \"color_relief\"\ncolor_file = \"{file}\"\n");
    let error = |tail: String| Config::parse(&layer(tail)).unwrap_err();

    let followed = format!("{}\n[[layer.op]]\nkind = \"slope\"\n", relief_op(&colors));
    assert!(error(followed).contains("last op"));
    let with_gray = format!("gray = {{ min = 0.0, max = 1.0 }}\n{}", relief_op(&colors));
    assert!(error(with_gray).contains("no gray or ramp"));
    let both = format!("ramp = \"{colors}\"\ngray = {{ min = 0.0, max = 1.0 }}\n");
    assert!(error(both).contains("both gray and ramp"));
    assert!(error(format!("ramp = \"{percent}\"\n")).contains("percent stops"));
    assert!(error(relief_op(&percent)).contains("min and max"));
    assert!(error(relief_op(&mauve)).contains("unknown colour name"));

    let ranged = format!(
        "{}selection = \"nearest\"\nrange = {{ min = 0.0, max = 1.0 }}\n",
        relief_op(&percent)
    );
    let config = Config::parse(&layer(ranged)).unwrap();
    assert!(matches!(
        config.layers[0].ops[0],
        OpConfig::ColorRelief {
            selection: ColorSelectionConfig::Nearest,
            range: Some(_),
            ..
        }
    ));
}
