use std::collections::HashMap;
use std::path::{Path, PathBuf};

use geoplumb::element::Source;
use geoplumb::elements::{GpkgSrc, VecSrc, read_csv, read_shapefile, write_geopackage};
use geoplumb::{Bbox, Crs, VectorChunk, VectorFeature, WindowReq};
use rusqlite::Connection;
use serde_json::{Value, json};
use shapefile::dbase::{FieldName, FieldValue, Record, TableWriterBuilder};
use shapefile::{PolygonRing, Writer};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{Coord, LineString, MultiPolygon, Point, Polygon, Ring};

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

fn write_named_point(dir: &Path, name: &str) -> PathBuf {
    let path = dir.join("countries.shp");
    let table = TableWriterBuilder::new().add_character_field(field("name"), 32);
    let mut writer = Writer::from_path(&path, table).unwrap();
    let mut record = Record::default();
    record.insert("name".into(), FieldValue::Character(Some(name.to_string())));
    writer
        .write_shape_and_record(&shapefile::Point::new(-5.5, 7.5), &record)
        .unwrap();
    path
}

fn set_language_driver(shp: &Path, language_driver: u8) -> Vec<u8> {
    let dbf = shp.with_extension("dbf");
    let mut bytes = std::fs::read(&dbf).unwrap();
    // dbf header byte 29 is the language driver id, 0x57 is gdal's default ISO-8859-1
    bytes[29] = language_driver;
    std::fs::write(&dbf, &bytes).unwrap();
    bytes
}

async fn read_name(shp: &Path) -> Value {
    let chunk = every_feature(&VecSrc::from_shapefile(shp, Some(Crs::WGS84)).unwrap()).await;
    chunk.features[0].properties["name"].clone()
}

#[tokio::test]
async fn a_dbf_whose_language_driver_names_latin1_reads_as_latin1() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_named_point(dir.path(), "Cote d'Ivoire");
    let mut bytes = set_language_driver(&path, 0x57);
    let at = bytes.windows(4).position(|w| w == b"Cote").unwrap();
    // 0xF4 is ô in ISO-8859-1 and not valid UTF-8 on its own
    bytes[at + 1] = 0xF4;
    std::fs::write(path.with_extension("dbf"), &bytes).unwrap();

    assert_eq!(read_name(&path).await, json!("Côte d'Ivoire"));
}

#[tokio::test]
async fn a_cpg_naming_utf8_wins_over_the_language_driver() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_named_point(dir.path(), "Côte d'Ivoire");
    set_language_driver(&path, 0x57);
    std::fs::write(path.with_extension("cpg"), "UTF-8\r\n").unwrap();

    // read as ISO-8859-1 the two utf-8 bytes of ô would come back as Ã´
    assert_eq!(read_name(&path).await, json!("Côte d'Ivoire"));
}

#[test]
fn a_declared_encoding_geoplumb_cannot_decode_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_named_point(dir.path(), "Abidjan");
    std::fs::write(path.with_extension("cpg"), "437").unwrap();

    let message = VecSrc::from_shapefile(&path, Some(Crs::WGS84))
        .err()
        .unwrap()
        .to_string();
    assert!(message.contains("CP437"), "{message}");
}

fn write_ferry(dir: &Path, stored: &[(f64, f64)]) -> PathBuf {
    let path = dir.join("ferry.shp");
    let table = TableWriterBuilder::new().add_character_field(field("name"), 16);
    let mut writer = Writer::from_path(&path, table).unwrap();
    let line = shapefile::Polyline::new(
        stored
            .iter()
            .map(|&(x, y)| shapefile::Point::new(x, y))
            .collect(),
    );
    let mut record = Record::default();
    record.insert("name".into(), FieldValue::Character(Some("ferry".into())));
    writer.write_shape_and_record(&line, &record).unwrap();
    path
}

#[test]
fn read_shapefile_returns_every_vertex_as_stored_and_the_prj_crs() {
    let dir = tempfile::tempdir().unwrap();
    // VecSrc's window clip returns 0.3 as 0.30000000000000004 and drops the repeated vertex
    let stored = [(-0.1, 0.0), (0.3, 1.0), (0.3, 1.0), (0.7, 2.0)];
    let path = write_ferry(dir.path(), &stored);
    std::fs::write(path.with_extension("prj"), crs_definitions::EPSG_32633.wkt).unwrap();

    let (collection, crs) = read_shapefile(&path, None).unwrap();
    assert_eq!(crs, Crs(32633));
    let [ferry] = &collection.features[..] else {
        panic!("{} features", collection.features.len());
    };
    let coords = stored.iter().map(|&(x, y)| Coord::new(x, y)).collect();
    assert_eq!(
        ferry.geometry,
        Some(FeatureGeometry::LineString(LineString::new(coords)))
    );
    assert_eq!(ferry.properties["name"], json!("ferry"));
}

fn write_empty_point_shapefile(dir: &Path) -> PathBuf {
    let path = dir.join("empty.shp");
    // esri shapefile header: big-endian file code 9994 and length in 16-bit words, little-endian version 1000 and shape type 1, point
    let mut header = [0u8; 100];
    header[0..4].copy_from_slice(&9994i32.to_be_bytes());
    header[24..28].copy_from_slice(&50i32.to_be_bytes());
    header[28..32].copy_from_slice(&1000i32.to_le_bytes());
    header[32..36].copy_from_slice(&1i32.to_le_bytes());
    std::fs::write(&path, header).unwrap();
    std::fs::write(path.with_extension("shx"), header).unwrap();
    // dbase writes the table header when the writer is dropped
    drop(
        TableWriterBuilder::new()
            .add_character_field(field("name"), 16)
            .build_with_file_dest(path.with_extension("dbf"))
            .unwrap(),
    );
    path
}

#[test]
fn an_empty_shapefile_or_csv_reads_as_an_empty_collection() {
    let dir = tempfile::tempdir().unwrap();
    let shp = write_empty_point_shapefile(dir.path());
    let (collection, crs) = read_shapefile(&shp, Some(Crs::WGS84)).unwrap();
    assert!(collection.features.is_empty());
    assert_eq!(crs, Crs::WGS84);

    let csv = dir.path().join("empty.csv");
    std::fs::write(&csv, "name,lon,lat\n").unwrap();
    assert!(read_csv(&csv, "lon", "lat").unwrap().features.is_empty());
}

#[cfg(unix)]
#[test]
fn a_symlinked_shapefile_file_is_refused_with_or_without_a_target() {
    for extension in ["shp", "shx", "dbf", "prj", "cpg"] {
        let dir = tempfile::tempdir().unwrap();
        let path = write_named_point(dir.path(), "Abidjan");
        std::fs::write(path.with_extension("prj"), crs_definitions::EPSG_4326.wkt).unwrap();
        std::fs::write(path.with_extension("cpg"), "UTF-8").unwrap();
        let linked = path.with_extension(extension);
        let moved = dir.path().join(format!("moved.{extension}"));
        std::fs::rename(&linked, &moved).unwrap();

        let absent = dir.path().join(format!("absent.{extension}"));
        for target in [&moved, &absent] {
            std::os::unix::fs::symlink(target, &linked).unwrap();
            let message = read_shapefile(&path, None).err().unwrap().to_string();
            assert!(
                message.contains(&linked.display().to_string())
                    && message.contains("without following symlinks"),
                "{extension} linked to {}: {message}",
                target.display()
            );
            std::fs::remove_file(&linked).unwrap();
        }
        std::fs::rename(&moved, &linked).unwrap();
        assert_eq!(read_shapefile(&path, None).unwrap().0.features.len(), 1);
    }

    let dir = tempfile::tempdir().unwrap();
    let path = write_named_point(dir.path(), "Abidjan");
    for extension in ["shp", "shx", "dbf"] {
        std::fs::rename(
            path.with_extension(extension),
            path.with_extension(extension.to_ascii_uppercase()),
        )
        .unwrap();
    }
    let shp = path.with_extension("SHP");
    let moved = dir.path().join("moved.DBF");
    std::fs::rename(shp.with_extension("DBF"), &moved).unwrap();
    std::os::unix::fs::symlink(&moved, shp.with_extension("DBF")).unwrap();
    let message = read_shapefile(&shp, Some(Crs::WGS84))
        .err()
        .unwrap()
        .to_string();
    assert!(
        message.contains("countries.DBF") && message.contains("without following symlinks"),
        "{message}"
    );
}

#[test]
fn read_csv_returns_every_coordinate_as_written() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("moorings.csv");
    std::fs::write(&path, "name,lon,lat\neast,0.3,-0.1\nwest,0.3,-0.1\n").unwrap();

    let collection = read_csv(&path, "lon", "lat").unwrap();
    let geometries: Vec<_> = collection
        .features
        .iter()
        .map(|feature| feature.geometry.clone())
        .collect();
    assert_eq!(
        geometries,
        vec![Some(FeatureGeometry::Point(Point::new(0.3, -0.1))); 2]
    );
    assert_eq!(collection.features[1].properties["name"], json!("west"));
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

fn mooring(id: u64, lon: f64, lat: f64, name: &str) -> VectorFeature {
    VectorFeature {
        id,
        geometry: FeatureGeometry::Point(Point::new(lon, lat)),
        properties: HashMap::from([("name".to_string(), json!(name))]),
    }
}

fn names(chunk: &VectorChunk) -> Vec<&str> {
    chunk
        .features
        .iter()
        .map(|feature| feature.properties["name"].as_str().unwrap())
        .collect()
}

#[tokio::test]
async fn writing_a_layer_into_a_geopackage_replaces_that_layer_and_keeps_the_others() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("harbour.gpkg");
    write_geopackage(&path, "sites", &written_features(), Crs(32633)).unwrap();
    let old = [
        mooring(0, PIER_X, NORTHING, "old east"),
        mooring(1, DOCK_X, NORTHING, "old west"),
    ];
    write_geopackage(&path, "moorings", &old, Crs(32633)).unwrap();
    let mut new = [
        mooring(0, 13.4, 52.5, "east"),
        mooring(1, 13.5, 52.6, "west"),
        mooring(2, 13.6, 52.7, "north"),
    ];
    new[2].properties.insert("depth".to_string(), json!(4.5));
    write_geopackage(&path, "moorings", &new, Crs::WGS84).unwrap();

    let sites = every_feature(&GpkgSrc::open(&path, Some("sites"), None).unwrap()).await;
    assert_eq!(sites.crs, Crs(32633));
    assert_eq!(names(&sites), ["pier", "dock", "buoy"]);
    let moorings = every_feature(&GpkgSrc::open(&path, Some("moorings"), None).unwrap()).await;
    assert_eq!(moorings.crs, Crs::WGS84);
    assert_eq!(names(&moorings), ["east", "west", "north"]);
    assert_eq!(moorings.features[2].properties["depth"], json!(4.5));

    let connection = Connection::open(&path).unwrap();
    let count = |sql: &str| -> i64 { connection.query_row(sql, [], |row| row.get(0)).unwrap() };
    for metadata in ["gpkg_contents", "gpkg_geometry_columns", "gpkg_extensions"] {
        assert_eq!(
            count(&format!("SELECT count(*) FROM {metadata}")),
            2,
            "{metadata}"
        );
    }
    assert_eq!(count("SELECT count(*) FROM rtree_moorings_geom"), 3);
    assert_eq!(
        count(
            "SELECT count(*) FROM sqlite_master WHERE type = 'trigger' \
             AND name LIKE 'rtree_moorings_geom_%'"
        ),
        7
    );
}

#[test]
fn writing_a_geopackage_over_a_file_that_is_not_one_is_refused() {
    let dir = tempfile::tempdir().unwrap();
    let text = dir.path().join("notes.gpkg");
    std::fs::write(&text, "harbour notes, not a database").unwrap();
    let message = write_geopackage(&text, "moorings", &written_features(), Crs(32633))
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("notes.gpkg") && message.contains("not a geopackage"),
        "{message}"
    );
    assert_eq!(
        std::fs::read_to_string(&text).unwrap(),
        "harbour notes, not a database"
    );

    let database = dir.path().join("ledger.gpkg");
    let connection = Connection::open(&database).unwrap();
    connection
        .execute_batch("CREATE TABLE entries (amount INTEGER)")
        .unwrap();
    drop(connection);
    let message = write_geopackage(&database, "moorings", &written_features(), Crs(32633))
        .unwrap_err()
        .to_string();
    assert!(
        message.contains("ledger.gpkg") && message.contains("not a geopackage"),
        "{message}"
    );
    let tables: Vec<String> = Connection::open(&database)
        .unwrap()
        .prepare("SELECT name FROM sqlite_master")
        .unwrap()
        .query_map([], |row| row.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();
    assert_eq!(tables, ["entries"]);
}

#[test]
fn an_empty_or_nul_holding_name_is_refused_before_any_file_is_made() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("names.gpkg");
    let mut unnamed_property = written_features();
    unnamed_property[0]
        .properties
        .insert(String::new(), json!("blank"));
    let cases = [
        ("", written_features(), "cannot be empty"),
        ("moor\0ings", written_features(), "NUL"),
        ("moorings", unnamed_property, "cannot be empty"),
    ];
    for (table, features, problem) in cases {
        let message = write_geopackage(&path, table, &features, Crs(32633))
            .unwrap_err()
            .to_string();
        assert!(message.contains(problem), "{table:?}: {message}");
        assert!(!path.exists(), "{table:?} made a file");
    }
}
