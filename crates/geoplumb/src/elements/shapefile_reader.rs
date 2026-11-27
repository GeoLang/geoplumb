use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};

use serde_json::Value;
use shapefile::dbase::encoding::DynEncoding;
use shapefile::dbase::{self, FieldValue, Unicode};
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
        let (collection, crs) = read_shapefile(path, crs)?;
        VecSrc::new(collection, crs)
    }
}

pub fn read_shapefile(path: &Path, crs: Option<Crs>) -> Result<(FeatureCollection, Crs)> {
    let named = |message: String| Error::Source(format!("{}: {message}", path.display()));
    let crs = match crs {
        Some(crs) => crs,
        None => prj_crs(path)?,
    };
    let shapes = ShapeReader::with_shx(open_buffered(path)?, open_buffered(&sidecar(path, "shx"))?)
        .map_err(|e| named(e.to_string()))?;
    let dbf = sidecar(path, "dbf");
    let table = dbase::ReaderBuilder::new()
        .with_encoding(dbf_encoding(path, &dbf)?)
        .build(open_buffered(&dbf)?)
        .map_err(|e| Error::Source(format!("{}: {e}", dbf.display())))?;
    let mut reader = shapefile::Reader::new(shapes, table);

    let mut rows = Vec::new();
    for (index, row) in reader.iter_shapes_and_records().enumerate() {
        let (shape, record) = row.map_err(|e| named(format!("record {index}: {e}")))?;
        let geometry = shape_geometry(shape).map_err(|e| named(format!("record {index}: {e}")))?;
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
    Ok((FeatureCollection { features }, crs))
}

#[cfg(unix)]
fn open_without_following_symlinks(path: &Path) -> std::io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;
    // O_NOFOLLOW fails the open with ELOOP when the last path component is a symlink
    OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW)
        .open(path)
        .map_err(|e| {
            if e.raw_os_error() != Some(libc::ELOOP) {
                return e;
            }
            std::io::Error::other(format!(
                "{e}, shapefile files are opened without following symlinks"
            ))
        })
}

#[cfg(windows)]
fn open_without_following_symlinks(path: &Path) -> std::io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;
    // windows FILE_FLAG_OPEN_REPARSE_POINT opens a symlink itself instead of its target
    const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OPEN_REPARSE_POINT)
        .open(path)?;
    if file.metadata()?.file_type().is_symlink() {
        return Err(std::io::Error::other(
            "a symbolic link, shapefile files are opened without following symlinks",
        ));
    }
    Ok(file)
}

fn open_buffered(path: &Path) -> Result<BufReader<File>> {
    open_without_following_symlinks(path)
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
    let mut wkt = String::new();
    let read =
        open_without_following_symlinks(&prj).and_then(|mut file| file.read_to_string(&mut wkt));
    match read {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {
            return Err(Error::Source(format!(
                "{}: no {} beside it to give the crs, pass a crs",
                shp.display(),
                prj.display()
            )));
        }
        Err(e) => return Err(Error::Source(format!("{}: {e}", prj.display()))),
    }
    projicio_core::epsg::parse_wkt_epsg(&wkt)
        .map(Crs)
        .ok_or_else(|| {
            Error::Source(format!(
                "{}: the well-known text names no epsg code, pass a crs",
                prj.display()
            ))
        })
}

// gdal's GetEncodingFromLDIDNumber, the code page each dbf language driver id names
const LANGUAGE_DRIVER_CODE_PAGES: &[(u8, u16)] = &[
    (1, 437),
    (2, 850),
    (3, 1252),
    (4, 10000),
    (8, 865),
    (10, 850),
    (11, 437),
    (13, 437),
    (14, 850),
    (15, 437),
    (16, 850),
    (17, 437),
    (18, 850),
    (19, 932),
    (20, 850),
    (21, 437),
    (22, 850),
    (23, 865),
    (24, 437),
    (25, 437),
    (26, 850),
    (27, 437),
    (28, 863),
    (29, 850),
    (31, 852),
    (34, 852),
    (35, 852),
    (36, 860),
    (37, 850),
    (38, 866),
    (55, 850),
    (64, 852),
    (77, 936),
    (78, 949),
    (79, 950),
    (80, 874),
    (88, 1252),
    (89, 1252),
    (100, 852),
    (101, 866),
    (102, 865),
    (103, 861),
    (104, 895),
    (105, 620),
    (106, 737),
    (107, 857),
    (108, 863),
    (120, 950),
    (121, 949),
    (122, 936),
    (123, 932),
    (124, 874),
    (134, 737),
    (135, 852),
    (136, 857),
    (150, 10007),
    (151, 10029),
    (200, 1250),
    (201, 1251),
    (202, 1254),
    (203, 1253),
    (204, 1257),
];

fn cpg_encoding_name(line: &str) -> String {
    // gdal's GetEncodingFromCPG
    let digits: String = line
        .trim_start()
        .chars()
        .take_while(char::is_ascii_digit)
        .collect();
    match digits.parse::<u32>() {
        Ok(code) if (437..=950).contains(&code) || (1250..=1258).contains(&code) => {
            return format!("CP{code}");
        }
        _ => {}
    }
    let upper = line.to_ascii_uppercase();
    if upper.starts_with("8859") {
        let rest = &line[4..];
        return format!("ISO-8859-{}", rest.strip_prefix('-').unwrap_or(rest));
    }
    if upper.starts_with("UTF-8") || upper.starts_with("UTF8") {
        return "UTF-8".to_string();
    }
    if upper.starts_with("ANSI 1251") {
        return "CP1251".to_string();
    }
    line.to_string()
}

fn declared_encoding(shp: &Path, dbf: &Path) -> Result<Option<String>> {
    // gdal's DBFOpenLL reads the .cpg's first line, then the language driver id at header byte 29
    let cpg = sidecar(shp, "cpg");
    let mut bytes = Vec::new();
    let read =
        open_without_following_symlinks(&cpg).and_then(|mut file| file.read_to_end(&mut bytes));
    match read {
        Ok(_) => {
            let text = String::from_utf8_lossy(&bytes);
            let line = &text[..text.find(['\r', '\n']).unwrap_or(text.len())];
            if !line.is_empty() {
                return Ok(Some(cpg_encoding_name(line)));
            }
        }
        Err(e) if e.kind() == ErrorKind::NotFound => {}
        Err(e) => return Err(Error::Source(format!("{}: {e}", cpg.display()))),
    }
    let mut header = [0u8; 30];
    open_without_following_symlinks(dbf)
        .and_then(|mut file| file.read_exact(&mut header))
        .map_err(|e| Error::Source(format!("{}: {e}", dbf.display())))?;
    let language_driver = header[29];
    // gdal names 87, the id its own shapefile writer defaults to, iso-8859-1 rather than a code page
    if language_driver == 87 {
        return Ok(Some("ISO-8859-1".to_string()));
    }
    Ok(LANGUAGE_DRIVER_CODE_PAGES
        .iter()
        .find(|(id, _)| *id == language_driver)
        .map(|(_, code_page)| format!("CP{code_page}")))
}

fn dbf_encoding(shp: &Path, dbf: &Path) -> Result<DynEncoding> {
    // gdal recodes nothing when neither the .cpg nor the language driver id names an encoding
    let Some(name) = declared_encoding(shp, dbf)? else {
        return Ok(Unicode.into());
    };
    if name.eq_ignore_ascii_case("UTF-8") {
        return Ok(Unicode.into());
    }
    DynEncoding::from_name(&name).ok_or_else(|| {
        Error::Source(format!(
            "{}: the dbf text is in {name}, which geoplumb cannot decode",
            dbf.display()
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
