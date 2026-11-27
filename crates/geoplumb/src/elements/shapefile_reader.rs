use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{BufReader, ErrorKind};
use std::path::{Path, PathBuf};

use serde_json::Value;
use shapefile::dbase::{self, FieldValue};
use shapefile::record::traits::HasXY;
use shapefile::{PolygonRing, Shape, ShapeReader};
use topoi_core::geojson::{Feature, FeatureCollection, FeatureGeometry};
use topoi_core::{
    Coord, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, Ring,
};

use super::vector::VecSrc;
use crate::caps::Crs;
use crate::error::{Error, Result};

// f64 holds every integer up to 2^53 exactly
const MAX_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;

impl VecSrc {
    pub fn from_shapefile(path: &Path, crs: Option<Crs>) -> Result<Self> {
        let named = |message: String| Error::Source(format!("{}: {message}", path.display()));
        let crs = match crs {
            Some(crs) => crs,
            None => prj_crs(path)?,
        };
        let shapes = ShapeReader::with_shx(open(path)?, open(&sidecar(path, "shx"))?)
            .map_err(|e| named(e.to_string()))?;
        let dbf = sidecar(path, "dbf");
        let table = dbase::Reader::new(open(&dbf)?)
            .map_err(|e| Error::Source(format!("{}: {e}", dbf.display())))?;
        let mut reader = shapefile::Reader::new(shapes, table);

        let mut rows = Vec::new();
        for (index, row) in reader.iter_shapes_and_records().enumerate() {
            let (shape, record) = row.map_err(|e| named(format!("record {index}: {e}")))?;
            let geometry =
                shape_geometry(shape).map_err(|e| named(format!("record {index}: {e}")))?;
            rows.push((geometry, record.into_iter().collect::<Vec<_>>()));
        }

        // dbase keeps a numeric field's decimal count private
        let mut numeric_columns: HashMap<String, bool> = HashMap::new();
        for (_, fields) in &rows {
            for (name, value) in fields {
                if let FieldValue::Numeric(Some(number)) = value {
                    let whole = numeric_columns.entry(name.clone()).or_insert(true);
                    *whole &= number.fract() == 0.0 && number.abs() <= MAX_EXACT_INTEGER;
                }
            }
        }
        let whole_columns: HashSet<String> = numeric_columns
            .into_iter()
            .filter(|(_, whole)| *whole)
            .map(|(name, _)| name)
            .collect();
        let features = rows
            .into_iter()
            .map(|(geometry, fields)| Feature {
                geometry,
                properties: fields
                    .into_iter()
                    .map(|(name, value)| {
                        let whole = whole_columns.contains(&name);
                        (name, property(value, whole))
                    })
                    .collect(),
            })
            .collect();
        VecSrc::new(FeatureCollection { features }, crs)
    }
}

fn open(path: &Path) -> Result<BufReader<File>> {
    File::open(path)
        .map(BufReader::new)
        .map_err(|e| Error::Source(format!("{}: {e}", path.display())))
}

fn sidecar(shp: &Path, lowercase_extension: &str) -> PathBuf {
    if shp.extension().and_then(|e| e.to_str()) == Some("SHP") {
        return shp.with_extension(lowercase_extension.to_ascii_uppercase());
    }
    shp.with_extension(lowercase_extension)
}

fn prj_crs(shp: &Path) -> Result<Crs> {
    let prj = sidecar(shp, "prj");
    let wkt = match std::fs::read_to_string(&prj) {
        Ok(wkt) => wkt,
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return Err(Error::Source(format!(
                "{}: no {} beside it to give the crs, pass a crs",
                shp.display(),
                prj.display()
            )));
        }
        Err(e) => return Err(Error::Source(format!("{}: {e}", prj.display()))),
    };
    projicio_core::epsg::parse_wkt_epsg(&wkt)
        .map(Crs)
        .ok_or_else(|| {
            Error::Source(format!(
                "{}: the well-known text names no epsg code, pass a crs",
                prj.display()
            ))
        })
}

fn coord<P: HasXY>(point: &P) -> Coord {
    Coord::new(point.x(), point.y())
}

fn line_geometry<P: HasXY>(parts: &[Vec<P>]) -> FeatureGeometry {
    let mut lines: Vec<LineString> = parts
        .iter()
        .map(|part| LineString::new(part.iter().map(coord).collect()))
        .collect();
    match lines.len() {
        1 => FeatureGeometry::LineString(lines.remove(0)),
        _ => FeatureGeometry::MultiLineString(MultiLineString::new(lines)),
    }
}

fn polygon_geometry<P: HasXY>(
    rings: &[PolygonRing<P>],
) -> std::result::Result<FeatureGeometry, String> {
    let mut polygons: Vec<(Ring, Vec<Ring>)> = Vec::new();
    for ring in rings {
        let coords = Ring::new(ring.points().iter().map(coord).collect());
        match ring {
            PolygonRing::Outer(_) => polygons.push((coords, Vec::new())),
            // shapefile's own geo-types conversion attaches each hole to the outer ring before it
            PolygonRing::Inner(_) => polygons
                .last_mut()
                .ok_or_else(|| "a hole comes before any outer ring".to_string())?
                .1
                .push(coords),
        }
    }
    let mut polygons: Vec<Polygon> = polygons
        .into_iter()
        .map(|(exterior, holes)| Polygon::new(exterior, holes))
        .collect();
    match polygons.len() {
        0 => Err("a polygon with no rings".into()),
        1 => Ok(FeatureGeometry::Polygon(polygons.remove(0))),
        _ => Ok(FeatureGeometry::MultiPolygon(MultiPolygon::new(polygons))),
    }
}

fn multipoint_geometry<P: HasXY>(points: &[P]) -> FeatureGeometry {
    FeatureGeometry::MultiPoint(MultiPoint::new(
        points.iter().map(|p| Point(coord(p))).collect(),
    ))
}

fn shape_geometry(shape: Shape) -> std::result::Result<Option<FeatureGeometry>, String> {
    let geometry = match shape {
        Shape::NullShape => return Ok(None),
        Shape::Point(p) => FeatureGeometry::Point(Point(coord(&p))),
        Shape::PointM(p) => FeatureGeometry::Point(Point(coord(&p))),
        Shape::PointZ(p) => FeatureGeometry::Point(Point(coord(&p))),
        Shape::Polyline(l) => line_geometry(l.parts()),
        Shape::PolylineM(l) => line_geometry(l.parts()),
        Shape::PolylineZ(l) => line_geometry(l.parts()),
        Shape::Polygon(p) => polygon_geometry(p.rings())?,
        Shape::PolygonM(p) => polygon_geometry(p.rings())?,
        Shape::PolygonZ(p) => polygon_geometry(p.rings())?,
        Shape::Multipoint(m) => multipoint_geometry(m.points()),
        Shape::MultipointM(m) => multipoint_geometry(m.points()),
        Shape::MultipointZ(m) => multipoint_geometry(m.points()),
        Shape::Multipatch(_) => return Err("a multipatch has no 2d feature form".into()),
    };
    Ok(Some(geometry))
}

fn property(value: FieldValue, whole_numeric: bool) -> Value {
    match value {
        FieldValue::Character(Some(text)) | FieldValue::Memo(text) => Value::from(text),
        FieldValue::Numeric(Some(number)) if whole_numeric => Value::from(number as i64),
        FieldValue::Numeric(Some(number))
        | FieldValue::Double(number)
        | FieldValue::Currency(number) => Value::from(number),
        FieldValue::Float(Some(number)) => Value::from(f64::from(number)),
        FieldValue::Integer(number) => Value::from(number),
        FieldValue::Logical(Some(flag)) => Value::from(flag),
        FieldValue::Date(Some(date)) => Value::from(format!(
            "{:04}-{:02}-{:02}",
            date.year(),
            date.month(),
            date.day()
        )),
        FieldValue::DateTime(moment) => {
            let (date, time) = (moment.date(), moment.time());
            Value::from(format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                date.year(),
                date.month(),
                date.day(),
                time.hours(),
                time.minutes(),
                time.seconds()
            ))
        }
        FieldValue::Character(None)
        | FieldValue::Numeric(None)
        | FieldValue::Float(None)
        | FieldValue::Logical(None)
        | FieldValue::Date(None) => Value::Null,
    }
}
