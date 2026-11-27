use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension};
use serde_json::Value;
use topoi_core::geojson::FeatureGeometry;

use super::file_stamp;
use super::vector::{geometry_envelope, simplify_geometry};
use crate::caps::{CapsPattern, CapsSet, Constraint, Crs, ResRange, SetField, VectorPattern};
use crate::chunk::{Chunk, VectorChunk, VectorFeature, clip_geometry};
use crate::element::Source;
use crate::error::{Error, Result};
use crate::window::{GridSpec, WindowReq};

pub struct GpkgSrc {
    table: Arc<FeatureTable>,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    crs: Crs,
}

struct FeatureTable {
    path: PathBuf,
    name: String,
    window_query: String,
    property_columns: Vec<String>,
    // a rusqlite connection is Send but not Sync
    connections: Mutex<Vec<Connection>>,
    rows_read: AtomicU64,
}

const MAX_POOLED_CONNECTIONS: usize = 4;

impl GpkgSrc {
    pub fn open(path: impl AsRef<Path>, table: Option<&str>, crs: Option<Crs>) -> Result<Self> {
        let path = path.as_ref();
        let sql_error = |e: rusqlite::Error| Error::Source(format!("{}: {e}", path.display()));
        let source_error =
            |message: String| Error::Source(format!("{}: {message}", path.display()));
        let connection = connect(path)?;

        let table = match table {
            Some(name) => {
                let registered: bool = connection
                    .query_row(
                        "SELECT EXISTS(SELECT 1 FROM gpkg_contents \
                         WHERE data_type = 'features' AND table_name = ?1)",
                        [name],
                        |row| row.get(0),
                    )
                    .map_err(sql_error)?;
                if !registered {
                    return Err(source_error(format!(
                        "table {name} is not a feature table in gpkg_contents"
                    )));
                }
                name.to_string()
            }
            None => connection
                .query_row(
                    "SELECT table_name FROM gpkg_contents WHERE data_type = 'features' \
                     ORDER BY table_name LIMIT 1",
                    [],
                    |row| row.get(0),
                )
                .optional()
                .map_err(sql_error)?
                .ok_or_else(|| source_error("no feature table in gpkg_contents".into()))?,
        };

        let (geometry_column, srs_id): (String, i64) = connection
            .query_row(
                "SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name = ?1",
                [&table],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .optional()
            .map_err(sql_error)?
            .ok_or_else(|| {
                source_error(format!("table {table} has no row in gpkg_geometry_columns"))
            })?;

        let columns: Vec<(String, i64)> = connection
            .prepare("SELECT name, pk FROM pragma_table_info(?1) ORDER BY cid")
            .and_then(|mut statement| {
                statement
                    .query_map([&table], |row| Ok((row.get(0)?, row.get(1)?)))?
                    .collect()
            })
            .map_err(sql_error)?;
        let key_columns: Vec<&String> = columns
            .iter()
            .filter(|(_, pk)| *pk > 0)
            .map(|(name, _)| name)
            .collect();
        let [primary_key] = key_columns[..] else {
            return Err(source_error(format!(
                "table {table} needs one primary key column, it has {}",
                key_columns.len()
            )));
        };
        let property_columns: Vec<String> = columns
            .iter()
            .map(|(name, _)| name)
            .filter(|name| *name != primary_key && **name != geometry_column)
            .cloned()
            .collect();

        let rtree = format!("rtree_{table}_{geometry_column}");
        let has_rtree: bool = connection
            .query_row(
                "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1)",
                [&rtree],
                |row| row.get(0),
            )
            .map_err(sql_error)?;
        if !has_rtree {
            return Err(source_error(format!(
                "table {table} has no rtree index {rtree}, read it through VecSrc"
            )));
        }

        let extent: [Option<f64>; 4] = connection
            .query_row(
                "SELECT min_x, min_y, max_x, max_y FROM gpkg_contents WHERE table_name = ?1",
                [&table],
                |row| Ok([row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?]),
            )
            .map_err(sql_error)?;
        let [Some(min_x), Some(min_y), Some(max_x), Some(max_y)] = extent else {
            return Err(source_error(format!(
                "table {table} has no extent in gpkg_contents"
            )));
        };
        let longer_side = (max_x - min_x).max(max_y - min_y);
        if longer_side <= 0.0 {
            return Err(source_error(format!(
                "table {table} has a zero-size extent in gpkg_contents"
            )));
        }

        let crs = match crs {
            Some(crs) => crs,
            None => epsg_crs(&connection, srs_id)
                .map_err(sql_error)?
                .ok_or_else(|| {
                    source_error(format!(
                        "table {table} has srs_id {srs_id}, which is not an epsg code, pass a crs"
                    ))
                })?,
        };

        let selected: Vec<String> = [primary_key, &geometry_column]
            .into_iter()
            .chain(&property_columns)
            .map(|c| quote_identifier(c).map_err(source_error))
            .collect::<Result<_>>()?;
        let primary_key = quote_identifier(primary_key).map_err(source_error)?;
        let window_query = format!(
            "SELECT {} FROM {} WHERE {primary_key} IN (SELECT id FROM {} \
             WHERE maxx >= ?1 AND minx <= ?2 AND maxy >= ?3 AND miny <= ?4) \
             ORDER BY {primary_key}",
            selected.join(", "),
            quote_identifier(&table).map_err(source_error)?,
            quote_identifier(&rtree).map_err(source_error)?,
        );

        Ok(GpkgSrc {
            table: Arc::new(FeatureTable {
                path: path.to_path_buf(),
                name: table,
                window_query,
                property_columns,
                connections: Mutex::new(vec![connection]),
                rows_read: AtomicU64::new(0),
            }),
            origin_x: min_x,
            origin_y: max_y,
            // no segment lengths without reading every row, this is VecSrc's
            // own base when it has no segments
            base_resolution: longer_side / 1024.0,
            crs,
        })
    }

    pub fn rows_read(&self) -> u64 {
        self.table.rows_read.load(Ordering::Relaxed)
    }
}

fn connect(path: &Path) -> Result<Connection> {
    Connection::open_with_flags(
        path,
        OpenFlags::SQLITE_OPEN_READ_ONLY | OpenFlags::SQLITE_OPEN_NO_MUTEX,
    )
    .map_err(|e| Error::Source(format!("{}: {e}", path.display())))
}

fn quote_identifier(identifier: &str) -> std::result::Result<String, String> {
    if identifier.contains('"') {
        return Err(format!(
            "geopackage identifier {identifier} contains a double quote"
        ));
    }
    Ok(format!("\"{identifier}\""))
}

fn epsg_crs(connection: &Connection, srs_id: i64) -> rusqlite::Result<Option<Crs>> {
    let row: Option<(String, i64)> = connection
        .query_row(
            "SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys \
             WHERE srs_id = ?1",
            [srs_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()?;
    let Some((organization, code)) = row else {
        return Ok(None);
    };
    if !organization.eq_ignore_ascii_case("EPSG") || code != srs_id {
        return Ok(None);
    }
    Ok(u32::try_from(srs_id).ok().map(Crs))
}

impl FeatureTable {
    fn read_window(&self, req: &WindowReq, base_resolution: f64, crs: Crs) -> Result<VectorChunk> {
        let pooled = self.connections.lock().unwrap().pop();
        let connection = match pooled {
            Some(connection) => connection,
            None => connect(&self.path)?,
        };
        let features = self.query_window(&connection, req);
        let mut pool = self.connections.lock().unwrap();
        if pool.len() < MAX_POOLED_CONNECTIONS {
            pool.push(connection);
        }
        drop(pool);

        let coarse = req.resolution > base_resolution;
        let mut fragments = Vec::new();
        for f in features? {
            let geometry = if coarse {
                match simplify_geometry(&f.geometry, req.resolution) {
                    Some(g) => g,
                    None => continue,
                }
            } else {
                f.geometry
            };
            for geometry in clip_geometry(&geometry, &req.bbox) {
                fragments.push(VectorFeature {
                    id: f.id,
                    geometry,
                    properties: f.properties.clone(),
                });
            }
        }
        Ok(VectorChunk::new(fragments, req.bbox, req.resolution, crs))
    }

    fn query_window(&self, connection: &Connection, req: &WindowReq) -> Result<Vec<VectorFeature>> {
        let sql_error = |e: rusqlite::Error| {
            Error::Source(format!("{} table {}: {e}", self.path.display(), self.name))
        };
        let mut statement = connection
            .prepare_cached(&self.window_query)
            .map_err(sql_error)?;
        let mut rows = statement
            .query([
                req.bbox.min_x,
                req.bbox.max_x,
                req.bbox.min_y,
                req.bbox.max_y,
            ])
            .map_err(sql_error)?;
        let mut features = Vec::new();
        while let Some(row) = rows.next().map_err(sql_error)? {
            self.rows_read.fetch_add(1, Ordering::Relaxed);
            let pk: i64 = row.get(0).map_err(sql_error)?;
            let row_error = |message: String| {
                Error::Source(format!(
                    "{} table {} row {pk}: {message}",
                    self.path.display(),
                    self.name
                ))
            };
            let id = u64::try_from(pk).map_err(|_| row_error("negative primary key".into()))?;
            // VecSrc drops a feature with no geometry the same way
            let Some(blob) = row
                .get_ref(1)
                .map_err(sql_error)?
                .as_blob_or_null()
                .map_err(|e| row_error(format!("geometry: {e}")))?
            else {
                continue;
            };
            let geometry = topoi_core::read_geopackage_geometry(blob)
                .map_err(|e| row_error(format!("geometry: {e}")))?;
            let mut properties = HashMap::new();
            for (column, name) in self.property_columns.iter().enumerate() {
                let value = match row.get_ref(column + 2).map_err(sql_error)? {
                    ValueRef::Null => serde_json::Value::Null,
                    ValueRef::Integer(i) => serde_json::Value::from(i),
                    ValueRef::Real(r) => serde_json::Value::from(r),
                    ValueRef::Text(bytes) => std::str::from_utf8(bytes)
                        .map_err(|e| row_error(format!("column {name}: {e}")))?
                        .into(),
                    // a blob has no json form
                    ValueRef::Blob(_) => continue,
                };
                properties.insert(name.clone(), value);
            }
            features.push(VectorFeature {
                id,
                geometry,
                properties,
            });
        }
        Ok(features)
    }
}

impl Source for GpkgSrc {
    fn identity(&self) -> String {
        let table = &self.table;
        format!(
            "GpkgSrc path={:?} table={:?} property_columns={:?} origin_x={:?} origin_y={:?} base_resolution={:?} crs={} {}",
            table.path,
            table.name,
            table.property_columns,
            self.origin_x,
            self.origin_y,
            self.base_resolution,
            self.crs,
            file_stamp(&table.path)
        )
    }

    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Vector(VectorPattern {
            crs: SetField::one(self.crs),
            resolution: ResRange::at_least(self.base_resolution),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            base_resolution: self.base_resolution,
            chunk_px: 256,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let table = self.table.clone();
        let req = *req;
        let (base_resolution, crs) = (self.base_resolution, self.crs);
        Box::pin(async move {
            crate::engine::offload(move || {
                table
                    .read_window(&req, base_resolution, crs)
                    .map(Chunk::Vector)
            })
            .await
        })
    }
}

// "GPKG" in the sqlite header's application id field
const GPKG_APPLICATION_ID: i32 = 0x4750_4B47;
// geopackage 1.4, the spec version times ten thousand
const GPKG_USER_VERSION: i32 = 10400;
const GEOMETRY_COLUMN: &str = "geom";
const PRIMARY_KEY: &str = "fid";

// the tables the geopackage spec requires, with the columns and constraints gdal creates
const METADATA_TABLES: &str = "
    CREATE TABLE gpkg_spatial_ref_sys (
        srs_name TEXT NOT NULL,
        srs_id INTEGER NOT NULL PRIMARY KEY,
        organization TEXT NOT NULL,
        organization_coordsys_id INTEGER NOT NULL,
        definition TEXT NOT NULL,
        description TEXT);
    CREATE TABLE gpkg_contents (
        table_name TEXT NOT NULL PRIMARY KEY,
        data_type TEXT NOT NULL,
        identifier TEXT UNIQUE,
        description TEXT DEFAULT '',
        last_change DATETIME NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
        min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
        srs_id INTEGER,
        CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
    CREATE TABLE gpkg_geometry_columns (
        table_name TEXT NOT NULL,
        column_name TEXT NOT NULL,
        geometry_type_name TEXT NOT NULL,
        srs_id INTEGER NOT NULL,
        z TINYINT NOT NULL,
        m TINYINT NOT NULL,
        CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
        CONSTRAINT uk_gc_table_name UNIQUE (table_name),
        CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
        CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id));
    CREATE TABLE gpkg_extensions (
        table_name TEXT,
        column_name TEXT,
        extension_name TEXT NOT NULL,
        definition TEXT NOT NULL,
        scope TEXT NOT NULL,
        CONSTRAINT ge_tce UNIQUE (table_name, column_name, extension_name));
    INSERT INTO gpkg_spatial_ref_sys VALUES
        ('Undefined Cartesian SRS', -1, 'NONE', -1, 'undefined',
            'undefined Cartesian coordinate reference system'),
        ('Undefined geographic SRS', 0, 'NONE', 0, 'undefined',
            'undefined geographic coordinate reference system');";

#[derive(Debug, Clone, Copy, PartialEq)]
enum ColumnType {
    Integer,
    Real,
    Text,
    Boolean,
}

impl ColumnType {
    fn of(value: &Value) -> Option<ColumnType> {
        match value {
            Value::Null => None,
            Value::Bool(_) => Some(ColumnType::Boolean),
            Value::Number(number) if number.is_i64() => Some(ColumnType::Integer),
            Value::Number(_) => Some(ColumnType::Real),
            Value::String(_) | Value::Array(_) | Value::Object(_) => Some(ColumnType::Text),
        }
    }

    fn merge(self, other: ColumnType) -> ColumnType {
        match (self, other) {
            (a, b) if a == b => a,
            (ColumnType::Integer, ColumnType::Real) | (ColumnType::Real, ColumnType::Integer) => {
                ColumnType::Real
            }
            _ => ColumnType::Text,
        }
    }

    fn sql_name(self) -> &'static str {
        match self {
            ColumnType::Integer => "INTEGER",
            ColumnType::Real => "REAL",
            ColumnType::Text => "TEXT",
            ColumnType::Boolean => "BOOLEAN",
        }
    }

    fn sql_value(self, value: Option<&Value>) -> rusqlite::types::Value {
        use rusqlite::types::Value as Sql;
        let Some(value) = value.filter(|v| !v.is_null()) else {
            return Sql::Null;
        };
        match (self, value) {
            (ColumnType::Text, Value::String(text)) => Sql::Text(text.clone()),
            (ColumnType::Text, other) => Sql::Text(other.to_string()),
            (ColumnType::Boolean, value) => Sql::Integer(i64::from(
                value
                    .as_bool()
                    .expect("a boolean column holds only booleans"),
            )),
            (ColumnType::Integer, value) => Sql::Integer(
                value
                    .as_i64()
                    .expect("an integer column holds only i64 numbers"),
            ),
            (ColumnType::Real, value) => {
                Sql::Real(value.as_f64().expect("a real column holds only numbers"))
            }
        }
    }
}

fn property_columns(
    features: &[VectorFeature],
) -> std::result::Result<Vec<(String, ColumnType)>, String> {
    let mut columns: BTreeMap<&str, Option<ColumnType>> = BTreeMap::new();
    for feature in features {
        for (name, value) in &feature.properties {
            let seen = columns.entry(name.as_str()).or_default();
            *seen = match (*seen, ColumnType::of(value)) {
                (Some(a), Some(b)) => Some(a.merge(b)),
                (a, b) => a.or(b),
            };
        }
    }
    // sqlite matches column names case-insensitively
    let mut folded: HashMap<String, &str> = HashMap::from([
        (PRIMARY_KEY.to_string(), PRIMARY_KEY),
        (GEOMETRY_COLUMN.to_string(), GEOMETRY_COLUMN),
    ]);
    for name in columns.keys().copied() {
        if let Some(other) = folded.insert(name.to_ascii_lowercase(), name) {
            return Err(format!(
                "property {name} has the name of column {other} once case is ignored, \
                 as geopackage column names are, rename the property"
            ));
        }
    }
    Ok(columns
        .into_iter()
        .map(|(name, kind)| (name.to_string(), kind.unwrap_or(ColumnType::Text)))
        .collect())
}

fn geometry_type_name(geometry: &FeatureGeometry) -> &'static str {
    match geometry {
        FeatureGeometry::Point(_) => "POINT",
        FeatureGeometry::LineString(_) => "LINESTRING",
        FeatureGeometry::Polygon(_) => "POLYGON",
        FeatureGeometry::MultiPoint(_) => "MULTIPOINT",
        FeatureGeometry::MultiLineString(_) => "MULTILINESTRING",
        FeatureGeometry::MultiPolygon(_) => "MULTIPOLYGON",
        FeatureGeometry::GeometryCollection(_) => "GEOMETRYCOLLECTION",
    }
}

fn rtree_triggers(table: &str, rtree: &str) -> std::result::Result<String, String> {
    let trigger = |suffix: &str| quote_identifier(&format!("{rtree}_{suffix}"));
    let (insert, update6, update7, update2, update5, update4, delete) = (
        trigger("insert")?,
        trigger("update6")?,
        trigger("update7")?,
        trigger("update2")?,
        trigger("update5")?,
        trigger("update4")?,
        trigger("delete")?,
    );
    let quoted_table = quote_identifier(table)?;
    let quoted_rtree = quote_identifier(rtree)?;
    let geometry = format!("\"{GEOMETRY_COLUMN}\"");
    let key = format!("\"{PRIMARY_KEY}\"");
    let bounds = format!(
        "ST_MinX(NEW.{geometry}), ST_MaxX(NEW.{geometry}), ST_MinY(NEW.{geometry}), ST_MaxY(NEW.{geometry})"
    );
    // gdal's geopackage 1.4 rtree triggers, calling ST_ functions plain sqlite lacks
    Ok(format!(
        "CREATE TRIGGER {insert} AFTER INSERT ON {quoted_table} \
         WHEN (new.{geometry} NOT NULL AND NOT ST_IsEmpty(NEW.{geometry})) \
         BEGIN INSERT OR REPLACE INTO {quoted_rtree} VALUES (NEW.{key}, {bounds}); END;
         CREATE TRIGGER {update6} AFTER UPDATE OF {geometry} ON {quoted_table} \
         WHEN OLD.{key} = NEW.{key} AND (NEW.{geometry} NOTNULL AND NOT ST_IsEmpty(NEW.{geometry})) \
         AND (OLD.{geometry} NOTNULL AND NOT ST_IsEmpty(OLD.{geometry})) \
         BEGIN UPDATE {quoted_rtree} SET minx = ST_MinX(NEW.{geometry}), maxx = ST_MaxX(NEW.{geometry}), \
         miny = ST_MinY(NEW.{geometry}), maxy = ST_MaxY(NEW.{geometry}) WHERE id = NEW.{key}; END;
         CREATE TRIGGER {update7} AFTER UPDATE OF {geometry} ON {quoted_table} \
         WHEN OLD.{key} = NEW.{key} AND (NEW.{geometry} NOTNULL AND NOT ST_IsEmpty(NEW.{geometry})) \
         AND (OLD.{geometry} ISNULL OR ST_IsEmpty(OLD.{geometry})) \
         BEGIN INSERT INTO {quoted_rtree} VALUES (NEW.{key}, {bounds}); END;
         CREATE TRIGGER {update2} AFTER UPDATE OF {geometry} ON {quoted_table} \
         WHEN OLD.{key} = NEW.{key} AND (NEW.{geometry} ISNULL OR ST_IsEmpty(NEW.{geometry})) \
         BEGIN DELETE FROM {quoted_rtree} WHERE id = OLD.{key}; END;
         CREATE TRIGGER {update5} AFTER UPDATE ON {quoted_table} \
         WHEN OLD.{key} != NEW.{key} AND (NEW.{geometry} NOTNULL AND NOT ST_IsEmpty(NEW.{geometry})) \
         BEGIN DELETE FROM {quoted_rtree} WHERE id = OLD.{key}; \
         INSERT OR REPLACE INTO {quoted_rtree} VALUES (NEW.{key}, {bounds}); END;
         CREATE TRIGGER {update4} AFTER UPDATE ON {quoted_table} \
         WHEN OLD.{key} != NEW.{key} AND (NEW.{geometry} ISNULL OR ST_IsEmpty(NEW.{geometry})) \
         BEGIN DELETE FROM {quoted_rtree} WHERE id IN (OLD.{key}, NEW.{key}); END;
         CREATE TRIGGER {delete} AFTER DELETE ON {quoted_table} WHEN old.{geometry} NOT NULL \
         BEGIN DELETE FROM {quoted_rtree} WHERE id = OLD.{key}; END;"
    ))
}

pub fn write_geopackage(
    path: &Path,
    table: &str,
    features: &[VectorFeature],
    crs: Crs,
) -> Result<()> {
    let write_error = |message: String| Error::Write(format!("{}: {message}", path.display()));
    let sql_error = |e: rusqlite::Error| write_error(e.to_string());
    let columns = property_columns(features).map_err(write_error)?;
    let srs_id = i32::try_from(crs.0)
        .map_err(|_| write_error(format!("{crs} does not fit a geopackage srs_id")))?;
    let definition = u16::try_from(crs.0)
        .ok()
        .and_then(crs_definitions::from_code)
        .ok_or_else(|| {
            write_error(format!(
                "no well-known text for {crs}, which gpkg_spatial_ref_sys needs"
            ))
        })?
        .wkt;
    let mut kinds = features.iter().map(|f| geometry_type_name(&f.geometry));
    let geometry_type = match kinds.next() {
        Some(first) if kinds.all(|kind| kind == first) => first,
        _ => "GEOMETRY",
    };
    let envelopes: Vec<_> = features
        .iter()
        .map(|f| geometry_envelope(&f.geometry))
        .collect();
    // geometry_envelope gives an empty geometry an infinite envelope
    let extent = envelopes
        .iter()
        .filter(|e| e.min_x.is_finite())
        .copied()
        .reduce(|a, b| a.union(&b));

    let rtree = format!("rtree_{table}_{GEOMETRY_COLUMN}");
    let quoted_table = quote_identifier(table).map_err(write_error)?;
    let quoted_rtree = quote_identifier(&rtree).map_err(write_error)?;
    let column_names: Vec<String> = columns
        .iter()
        .map(|(name, _)| quote_identifier(name))
        .collect::<std::result::Result<_, _>>()
        .map_err(write_error)?;
    let column_definitions: String = column_names
        .iter()
        .zip(&columns)
        .map(|(name, (_, kind))| format!(", {name} {}", kind.sql_name()))
        .collect();
    let insert_feature = format!(
        "INSERT INTO {quoted_table} (\"{GEOMETRY_COLUMN}\"{}) VALUES (?1{})",
        column_names
            .iter()
            .map(|name| format!(", {name}"))
            .collect::<String>(),
        (0..columns.len())
            .map(|i| format!(", ?{}", i + 2))
            .collect::<String>(),
    );
    let triggers = rtree_triggers(table, &rtree).map_err(write_error)?;

    let mut connection = Connection::open(path).map_err(sql_error)?;
    connection
        .pragma_update(None, "application_id", GPKG_APPLICATION_ID)
        .map_err(sql_error)?;
    connection
        .pragma_update(None, "user_version", GPKG_USER_VERSION)
        .map_err(sql_error)?;
    let transaction = connection.transaction().map_err(sql_error)?;
    transaction
        .execute_batch(METADATA_TABLES)
        .map_err(sql_error)?;
    transaction
        .execute(
            "INSERT INTO gpkg_spatial_ref_sys VALUES ('WGS 84 geodetic', 4326, 'EPSG', 4326, ?1, \
             'longitude/latitude coordinates in decimal degrees on the WGS 84 spheroid')",
            [crs_definitions::EPSG_4326.wkt],
        )
        .map_err(sql_error)?;
    if crs != Crs::WGS84 {
        transaction
            .execute(
                "INSERT INTO gpkg_spatial_ref_sys VALUES (?1, ?2, 'EPSG', ?2, ?3, NULL)",
                rusqlite::params![crs.to_string(), srs_id, definition],
            )
            .map_err(sql_error)?;
    }
    transaction
        .execute_batch(&format!(
            "CREATE TABLE {quoted_table} (\"{PRIMARY_KEY}\" INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, \
             \"{GEOMETRY_COLUMN}\" {geometry_type}{column_definitions});
             CREATE VIRTUAL TABLE {quoted_rtree} USING rtree(id, minx, maxx, miny, maxy);"
        ))
        .map_err(sql_error)?;
    transaction
        .execute(
            "INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, \
             max_x, max_y, srs_id) VALUES (?1, 'features', ?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![
                table,
                extent.map(|e| e.min_x),
                extent.map(|e| e.min_y),
                extent.map(|e| e.max_x),
                extent.map(|e| e.max_y),
                srs_id
            ],
        )
        .map_err(sql_error)?;
    transaction
        .execute(
            "INSERT INTO gpkg_geometry_columns VALUES (?1, ?2, ?3, ?4, 0, 0)",
            rusqlite::params![table, GEOMETRY_COLUMN, geometry_type, srs_id],
        )
        .map_err(sql_error)?;
    transaction
        .execute(
            "INSERT INTO gpkg_extensions VALUES (?1, ?2, 'gpkg_rtree_index', \
             'http://www.geopackage.org/spec120/#extension_rtree', 'write-only')",
            rusqlite::params![table, GEOMETRY_COLUMN],
        )
        .map_err(sql_error)?;
    {
        let mut insert = transaction.prepare(&insert_feature).map_err(sql_error)?;
        let mut index = transaction
            .prepare(&format!(
                "INSERT INTO {quoted_rtree} VALUES (?1, ?2, ?3, ?4, ?5)"
            ))
            .map_err(sql_error)?;
        for (feature, envelope) in features.iter().zip(&envelopes) {
            let mut values = vec![rusqlite::types::Value::Blob(
                topoi_core::write_geopackage_geometry(&feature.geometry, srs_id),
            )];
            values.extend(
                columns
                    .iter()
                    .map(|(name, kind)| kind.sql_value(feature.properties.get(name))),
            );
            insert
                .execute(rusqlite::params_from_iter(values))
                .map_err(sql_error)?;
            if envelope.min_x.is_finite() {
                index
                    .execute(rusqlite::params![
                        transaction.last_insert_rowid(),
                        envelope.min_x,
                        envelope.max_x,
                        envelope.min_y,
                        envelope.max_y
                    ])
                    .map_err(sql_error)?;
            }
        }
    }
    transaction.execute_batch(&triggers).map_err(sql_error)?;
    transaction.commit().map_err(sql_error)?;
    connection.close().map_err(|(_, e)| sql_error(e))
}
