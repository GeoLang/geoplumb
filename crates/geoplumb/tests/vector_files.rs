use std::collections::HashMap;
use std::path::{Path, PathBuf};

use geoplumb::element::Source;
use geoplumb::elements::{GpkgSrc, VecSrc, write_geopackage};
use geoplumb::{Bbox, Crs, VectorChunk, VectorFeature, WindowReq};
use rusqlite::Connection;
use serde_json::{Value, json};
use shapefile::dbase::{FieldName, FieldValue, Record, TableWriterBuilder};
use shapefile::{PolygonRing, Writer};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{Coord, MultiPolygon, Point, Polygon, Ring};

async fn every_feature(src: &impl Source) -> VectorChunk {
    let grid = src.grid();
    // reads at the base resolution are never simplified
    let reach = 1e7;
    let bbox = Bbox::new(
        grid.origin_x - reach,
        grid.origin_y - reach,
        grid.origin_x + reach,
        grid.origin_y + reach,
    );
    let request = WindowReq {
        bbox,
        resolution: grid.base_resolution,
        time: None,
    };
    let mut chunk = src.read(&request).await.unwrap().into_vector().unwrap();
    chunk.features.sort_by_key(|f| f.id);
    chunk
}

fn square_points(min_x: f64, min_y: f64, side: f64) -> Vec<shapefile::Point> {
    [
        (min_x, min_y),
        (min_x, min_y + side),
        (min_x + side, min_y + side),
        (min_x + side, min_y),
        (min_x, min_y),
    ]
    .into_iter()
    .map(|(x, y)| shapefile::Point::new(x, y))
    .collect()
}

fn field(name: &str) -> FieldName {
    FieldName::try_from(name).unwrap()
}

fn record(name: &str, count: f64, depth: f64, open: bool) -> Record {
    let mut record = Record::default();
    record.insert("name".into(), FieldValue::Character(Some(name.to_string())));
    record.insert("count".into(), FieldValue::Numeric(Some(count)));
    record.insert("depth".into(), FieldValue::Numeric(Some(depth)));
    record.insert("open".into(), FieldValue::Logical(Some(open)));
    record
}

const PIER_X: f64 = 500_000.0;
const DOCK_X: f64 = 500_200.0;
const NORTHING: f64 = 4_000_000.0;

fn write_shapefile(dir: &Path) -> PathBuf {
    let path = dir.join("parcels.shp");
    let table = TableWriterBuilder::new()
        .add_character_field(field("name"), 16)
        .add_numeric_field(field("count"), 6, 0)
        .add_numeric_field(field("depth"), 8, 2)
        .add_logical_field(field("open"));
    let mut writer = Writer::from_path(&path, table).unwrap();
    let pier = shapefile::Polygon::with_rings(vec![
        PolygonRing::Outer(square_points(PIER_X, NORTHING, 100.0)),
        PolygonRing::Inner(square_points(PIER_X + 20.0, NORTHING + 20.0, 20.0)),
    ]);
    let dock = shapefile::Polygon::with_rings(vec![
        PolygonRing::Outer(square_points(DOCK_X, NORTHING, 50.0)),
        PolygonRing::Outer(square_points(DOCK_X + 100.0, NORTHING, 20.0)),
    ]);
    writer
        .write_shape_and_record(&pier, &record("pier", 3.0, 1.5, true))
        .unwrap();
    writer
        .write_shape_and_record(&dock, &record("dock", 7.0, 2.0, false))
        .unwrap();
    path
}

#[tokio::test]
async fn a_shapefile_reads_its_prj_crs_holes_parts_and_typed_attributes() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_shapefile(dir.path());
    std::fs::write(path.with_extension("prj"), crs_definitions::EPSG_32633.wkt).unwrap();

    let chunk = every_feature(&VecSrc::from_shapefile(&path, None).unwrap()).await;
    assert_eq!(chunk.crs, Crs(32633));
    let [pier, dock] = &chunk.features[..] else {
        panic!("{} features", chunk.features.len());
    };

    let FeatureGeometry::Polygon(holed) = &pier.geometry else {
        panic!("the pier is {:?}", pier.geometry);
    };
    assert_eq!(holed.interiors().len(), 1);
    assert!((holed.area() - (100.0 * 100.0 - 20.0 * 20.0)).abs() < 1e-6);
    let FeatureGeometry::MultiPolygon(parts) = &dock.geometry else {
        panic!("the dock is {:?}", dock.geometry);
    };
    assert_eq!(parts.polygons().len(), 2);
    assert!((parts.area() - (50.0 * 50.0 + 20.0 * 20.0)).abs() < 1e-6);

    assert_eq!(pier.properties["name"], json!("pier"));
    assert_eq!(pier.properties["open"], json!(true));
    assert_eq!(dock.properties["open"], json!(false));
    // every count is whole, so the column reads as integers a vec_filter matches exactly
    assert!(pier.properties["count"].is_i64(), "{:?}", pier.properties);
    assert_eq!(pier.properties["count"], json!(3));
    assert_eq!(dock.properties["count"], json!(7));
    // 1.5 makes depth a float column, so the whole 2.0 stays a float
    assert_eq!(pier.properties["depth"], json!(1.5));
    assert!(dock.properties["depth"].is_f64(), "{:?}", dock.properties);
    assert_eq!(dock.properties["depth"].as_f64(), Some(2.0));
}

#[tokio::test]
async fn a_shapefile_without_an_epsg_prj_takes_the_crs_it_is_given() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_shapefile(dir.path());
    let missing = VecSrc::from_shapefile(&path, None)
        .err()
        .unwrap()
        .to_string();
    assert!(missing.contains("pass a crs"), "{missing}");

    // esri writes .prj files with no epsg authority
    std::fs::write(
        path.with_extension("prj"),
        r#"PROJCS["WGS_1984_UTM_Zone_33N",GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",SPHEROID["WGS_1984",6378137.0,298.257223563]],PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]],PROJECTION["Transverse_Mercator"],PARAMETER["False_Easting",500000.0],PARAMETER["False_Northing",0.0],PARAMETER["Central_Meridian",15.0],PARAMETER["Scale_Factor",0.9996],PARAMETER["Latitude_Of_Origin",0.0],UNIT["Meter",1.0]]"#,
    )
    .unwrap();
    let unnamed = VecSrc::from_shapefile(&path, None)
        .err()
        .unwrap()
        .to_string();
    assert!(unnamed.contains("names no epsg code"), "{unnamed}");

    let chunk = every_feature(&VecSrc::from_shapefile(&path, Some(Crs(32633))).unwrap()).await;
    assert_eq!(chunk.crs, Crs(32633));
    assert_eq!(chunk.features.len(), 2);
}

#[tokio::test]
async fn a_csv_reads_points_with_typed_properties() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sites.csv");
    std::fs::write(
        &path,
        "name,lon,lat,count,depth,note\n\
         pier,13.4,52.5,3,1.5,\n\
         dock,13.5,52.6,-7,2,\"Berlin, Mitte\"\n",
    )
    .unwrap();

    let chunk = every_feature(&VecSrc::from_csv(&path, "lon", "lat", Crs::WGS84).unwrap()).await;
    assert_eq!(chunk.crs, Crs::WGS84);
    let [pier, dock] = &chunk.features[..] else {
        panic!("{} features", chunk.features.len());
    };
    assert_eq!(
        pier.geometry,
        FeatureGeometry::Point(Point::new(13.4, 52.5))
    );
    assert_eq!(
        dock.geometry,
        FeatureGeometry::Point(Point::new(13.5, 52.6))
    );
    let expected = |pairs: &[(&str, Value)]| -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(name, value)| (name.to_string(), value.clone()))
            .collect()
    };
    assert_eq!(
        pier.properties,
        expected(&[
            ("name", json!("pier")),
            ("count", json!(3)),
            ("depth", json!(1.5)),
            ("note", json!("")),
        ])
    );
    assert_eq!(
        dock.properties,
        expected(&[
            ("name", json!("dock")),
            ("count", json!(-7)),
            ("depth", json!(2)),
            ("note", json!("Berlin, Mitte")),
        ])
    );
}

#[test]
fn a_csv_with_a_bad_coordinate_or_a_missing_column_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sites.csv");
    std::fs::write(&path, "name,lon,lat\npier,13.4,52.5\ndock,east,52.6\n").unwrap();

    let bad = VecSrc::from_csv(&path, "lon", "lat", Crs::WGS84)
        .err()
        .unwrap()
        .to_string();
    assert!(bad.contains("line 3") && bad.contains("\"east\""), "{bad}");
    let missing = VecSrc::from_csv(&path, "x", "lat", Crs::WGS84)
        .err()
        .unwrap()
        .to_string();
    assert!(missing.contains("no column is named x"), "{missing}");
}

fn square(min_x: f64, min_y: f64, side: f64) -> Ring {
    Ring::new(vec![
        Coord::new(min_x, min_y),
        Coord::new(min_x + side, min_y),
        Coord::new(min_x + side, min_y + side),
        Coord::new(min_x, min_y + side),
        Coord::new(min_x, min_y),
    ])
}

fn written_features() -> Vec<VectorFeature> {
    let holed = Polygon::new(
        square(PIER_X, NORTHING, 100.0),
        vec![square(PIER_X + 20.0, NORTHING + 20.0, 20.0)],
    );
    let parts = MultiPolygon::new(vec![
        Polygon::new(square(DOCK_X, NORTHING, 50.0), vec![]),
        Polygon::new(square(DOCK_X + 100.0, NORTHING, 20.0), vec![]),
    ]);
    let feature = |id: u64, geometry: FeatureGeometry, properties: Value| VectorFeature {
        id,
        geometry,
        properties: serde_json::from_value(properties).unwrap(),
    };
    vec![
        feature(
            0,
            FeatureGeometry::Polygon(holed),
            json!({ "name": "pier", "count": 3, "depth": 1.5, "open": true }),
        ),
        feature(
            1,
            FeatureGeometry::MultiPolygon(parts),
            json!({ "name": "dock", "count": 7, "open": false }),
        ),
        feature(
            2,
            FeatureGeometry::Point(Point::new(PIER_X + 50.0, NORTHING + 500.0)),
            json!({ "name": "buoy", "count": 1, "depth": 4 }),
        ),
    ]
}

#[tokio::test]
async fn a_written_geopackage_reads_back_through_gpkg_src() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("sites.gpkg");
    write_geopackage(&path, "sites", &written_features(), Crs(32633)).unwrap();

    let src = GpkgSrc::open(&path, None, None).unwrap();
    let chunk = every_feature(&src).await;
    assert_eq!(chunk.crs, Crs(32633));
    let [pier, dock, buoy] = &chunk.features[..] else {
        panic!("{} features", chunk.features.len());
    };
    let FeatureGeometry::Polygon(holed) = &pier.geometry else {
        panic!("the pier is {:?}", pier.geometry);
    };
    assert_eq!(holed.interiors().len(), 1);
    assert!((holed.area() - 9600.0).abs() < 1e-6);
    let FeatureGeometry::MultiPolygon(parts) = &dock.geometry else {
        panic!("the dock is {:?}", dock.geometry);
    };
    assert_eq!(parts.polygons().len(), 2);
    assert_eq!(
        buoy.geometry,
        FeatureGeometry::Point(Point::new(PIER_X + 50.0, NORTHING + 500.0))
    );

    assert_eq!(pier.properties["name"], json!("pier"));
    assert_eq!(pier.properties["count"], json!(3));
    assert_eq!(pier.properties["depth"], json!(1.5));
    // geopackage booleans are integers 0 and 1, which GpkgSrc reads as numbers
    assert_eq!(pier.properties["open"], json!(1));
    assert_eq!(dock.properties["open"], json!(0));
    assert_eq!(dock.properties["depth"], Value::Null);
    // 1.5 and 4 share a REAL column
    assert_eq!(buoy.properties["depth"].as_f64(), Some(4.0));
    assert_eq!(buoy.properties["open"], Value::Null);

    let connection = Connection::open(&path).unwrap();
    let pragma = |name: &str| -> i64 {
        connection
            .query_row(&format!("PRAGMA {name}"), [], |row| row.get(0))
            .unwrap()
    };
    assert_eq!(pragma("application_id"), 0x4750_4B47);
    assert_eq!(pragma("user_version"), 10400);
    let extent: (f64, f64, f64, f64, i64) = connection
        .query_row(
            "SELECT min_x, min_y, max_x, max_y, srs_id FROM gpkg_contents WHERE table_name = 'sites'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
        )
        .unwrap();
    assert_eq!(
        extent,
        (PIER_X, NORTHING, DOCK_X + 120.0, NORTHING + 500.0, 32633)
    );
    let geometry_type: String = connection
        .query_row(
            "SELECT geometry_type_name FROM gpkg_geometry_columns WHERE table_name = 'sites'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(geometry_type, "GEOMETRY");
    let srs_ids: Vec<i64> = connection
        .prepare("SELECT srs_id FROM gpkg_spatial_ref_sys ORDER BY srs_id")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(srs_ids, [-1, 0, 4326, 32633]);
    let triggers: i64 = connection
        .query_row(
            "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' \
             AND name LIKE 'rtree_sites_geom_%'",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(triggers, 7);
    let blob: Vec<u8> = connection
        .query_row("SELECT geom FROM sites WHERE fid = 3", [], |row| row.get(0))
        .unwrap();
    // little endian header with an xy envelope, then the epsg code as srs id
    assert_eq!(blob[..4], [b'G', b'P', 0, 0b0000_0011]);
    assert_eq!(blob[4..8], 32633i32.to_le_bytes());
}

#[test]
fn a_property_named_like_the_key_column_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clash.gpkg");
    let features = vec![VectorFeature {
        id: 0,
        geometry: FeatureGeometry::Point(Point::new(0.0, 0.0)),
        properties: HashMap::from([("FID".to_string(), json!(4))]),
    }];
    let message = write_geopackage(&path, "clash", &features, Crs::WGS84)
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("FID") && message.contains("fid"),
        "{message}"
    );
}
