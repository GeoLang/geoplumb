use std::path::Path;

use rusqlite::Connection;
use rusqlite::types::Value;
use topoi_core::Coord;
use topoi_core::geojson::FeatureGeometry;

pub const TABLE: &str = "parcels";
const GEOMETRY_COLUMN: &str = "geom";
const GPKG_APPLICATION_ID: i32 = 0x4750_4B47;

pub struct Srs {
    pub id: i64,
    pub organization: &'static str,
    pub organization_coordsys_id: i64,
}

pub const WGS84: Srs = Srs {
    id: 4326,
    organization: "EPSG",
    organization_coordsys_id: 4326,
};

pub struct FixtureFeature {
    pub id: i64,
    pub geometry: FeatureGeometry,
    // one value per entry of `Layer::columns`, in that order
    pub properties: Vec<Value>,
}

pub struct Layer<'a> {
    pub srs: Srs,
    // property column name beside its sql type
    pub columns: &'a [(&'a str, &'a str)],
    pub features: &'a [FixtureFeature],
    pub rtree: bool,
}

// min x, max x, min y, max y: the order the gpkg envelope and rtree use
pub fn envelope(geometry: &FeatureGeometry) -> [f64; 4] {
    let coords: Vec<Coord> = match geometry {
        FeatureGeometry::Polygon(p) => p.exterior().coords().to_vec(),
        FeatureGeometry::LineString(l) => l.coords().to_vec(),
        other => panic!("the fixture writes polygons and lines, got {other:?}"),
    };
    coords.iter().fold(
        [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ],
        |[min_x, max_x, min_y, max_y], c| {
            [
                min_x.min(c.x),
                max_x.max(c.x),
                min_y.min(c.y),
                max_y.max(c.y),
            ]
        },
    )
}

fn push_coords(coords: &[Coord], out: &mut Vec<u8>) {
    out.extend_from_slice(&(coords.len() as u32).to_le_bytes());
    for c in coords {
        out.extend_from_slice(&c.x.to_le_bytes());
        out.extend_from_slice(&c.y.to_le_bytes());
    }
}

fn geometry_blob(geometry: &FeatureGeometry, srs_id: i64) -> Vec<u8> {
    let envelope_type_xy = 1u8 << 1;
    let little_endian_header = 1u8;
    let mut out = vec![b'G', b'P', 0, envelope_type_xy | little_endian_header];
    out.extend_from_slice(&(srs_id as i32).to_le_bytes());
    for bound in envelope(geometry) {
        out.extend_from_slice(&bound.to_le_bytes());
    }
    out.push(1);
    match geometry {
        FeatureGeometry::Polygon(p) => {
            out.extend_from_slice(&3u32.to_le_bytes());
            out.extend_from_slice(&(1 + p.interiors().len() as u32).to_le_bytes());
            push_coords(p.exterior().coords(), &mut out);
            for hole in p.interiors() {
                push_coords(hole.coords(), &mut out);
            }
        }
        FeatureGeometry::LineString(l) => {
            out.extend_from_slice(&2u32.to_le_bytes());
            push_coords(l.coords(), &mut out);
        }
        other => panic!("the fixture writes polygons and lines, got {other:?}"),
    }
    out
}

pub fn write_geopackage(path: &Path, layer: &Layer) {
    let mut connection = Connection::open(path).unwrap();
    let property_columns: String = layer
        .columns
        .iter()
        .map(|(name, kind)| format!(", \"{name}\" {kind}"))
        .collect();
    connection
        .execute_batch(&format!(
            "PRAGMA application_id = {GPKG_APPLICATION_ID};
             CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL,
                 srs_id INTEGER PRIMARY KEY, organization TEXT NOT NULL,
                 organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL,
                 description TEXT);
             CREATE TABLE gpkg_contents (table_name TEXT PRIMARY KEY,
                 data_type TEXT NOT NULL, identifier TEXT, description TEXT DEFAULT '',
                 last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                 min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE, srs_id INTEGER);
             CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL,
                 column_name TEXT NOT NULL, geometry_type_name TEXT NOT NULL,
                 srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL);
             CREATE TABLE gpkg_extensions (table_name TEXT, column_name TEXT,
                 extension_name TEXT NOT NULL, definition TEXT NOT NULL, scope TEXT NOT NULL);
             CREATE TABLE \"{TABLE}\" (fid INTEGER PRIMARY KEY, \"{GEOMETRY_COLUMN}\" BLOB
                 {property_columns});"
        ))
        .unwrap();
    let transaction = connection.transaction().unwrap();
    transaction
        .execute(
            "INSERT INTO gpkg_spatial_ref_sys VALUES ('fixture', ?1, ?2, ?3, 'undefined', NULL)",
            rusqlite::params![
                layer.srs.id,
                layer.srs.organization,
                layer.srs.organization_coordsys_id
            ],
        )
        .unwrap();
    let placeholders: String = (0..layer.columns.len())
        .map(|i| format!(", ?{}", i + 3))
        .collect();
    let insert = format!("INSERT INTO \"{TABLE}\" VALUES (?1, ?2{placeholders})");
    let mut extent = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    for feature in layer.features {
        let mut values = vec![
            Value::Integer(feature.id),
            Value::Blob(geometry_blob(&feature.geometry, layer.srs.id)),
        ];
        values.extend(feature.properties.iter().cloned());
        transaction
            .execute(&insert, rusqlite::params_from_iter(values))
            .unwrap();
        let [min_x, max_x, min_y, max_y] = envelope(&feature.geometry);
        extent = [
            extent[0].min(min_x),
            extent[1].max(max_x),
            extent[2].min(min_y),
            extent[3].max(max_y),
        ];
    }
    transaction
        .execute(
            "INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y,
                 max_x, max_y, srs_id) VALUES (?1, 'features', ?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                TABLE,
                extent[0],
                extent[2],
                extent[1],
                extent[3],
                layer.srs.id
            ],
        )
        .unwrap();
    transaction
        .execute(
            "INSERT INTO gpkg_geometry_columns VALUES (?1, ?2, 'GEOMETRY', ?3, 0, 0)",
            rusqlite::params![TABLE, GEOMETRY_COLUMN, layer.srs.id],
        )
        .unwrap();
    if layer.rtree {
        let rtree = format!("rtree_{TABLE}_{GEOMETRY_COLUMN}");
        transaction
            .execute_batch(&format!(
                "CREATE VIRTUAL TABLE \"{rtree}\" USING rtree(id, minx, maxx, miny, maxy)"
            ))
            .unwrap();
        transaction
            .execute(
                "INSERT INTO gpkg_extensions VALUES (?1, ?2, 'gpkg_rtree_index',
                     'http://www.geopackage.org/spec120/#extension_rtree', 'write-only')",
                rusqlite::params![TABLE, GEOMETRY_COLUMN],
            )
            .unwrap();
        let insert = format!("INSERT INTO \"{rtree}\" VALUES (?1, ?2, ?3, ?4, ?5)");
        for feature in layer.features {
            let [min_x, max_x, min_y, max_y] = envelope(&feature.geometry);
            transaction
                .execute(
                    &insert,
                    rusqlite::params![feature.id, min_x, max_x, min_y, max_y],
                )
                .unwrap();
        }
    }
    transaction.commit().unwrap();
}
