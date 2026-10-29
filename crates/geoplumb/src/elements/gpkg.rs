use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, OptionalExtension};

use super::vector::simplify_geometry;
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
            .map(|c| quote_identifier(c))
            .collect::<Result<_>>()?;
        let primary_key = quote_identifier(primary_key)?;
        let window_query = format!(
            "SELECT {} FROM {} WHERE {primary_key} IN (SELECT id FROM {} \
             WHERE maxx >= ?1 AND minx <= ?2 AND maxy >= ?3 AND miny <= ?4) \
             ORDER BY {primary_key}",
            selected.join(", "),
            quote_identifier(&table)?,
            quote_identifier(&rtree)?,
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

fn quote_identifier(identifier: &str) -> Result<String> {
    if identifier.contains('"') {
        return Err(Error::Source(format!(
            "geopackage identifier {identifier} contains a double quote"
        )));
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
