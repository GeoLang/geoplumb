use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use geoplumb::caps::ValueKind;
use geoplumb::element::Source;
use geoplumb::elements::stac::parse_rfc3339;
use geoplumb::elements::{NetcdfSrc, Reproject};
use geoplumb::tile::{XyzTile, render_tile_at};
use geoplumb::{Bbox, Crs, Engine, Graph, RasterChunk, TimeInterval, WindowReq};

enum Values {
    Short(Vec<i16>),
    Float(Vec<f32>),
    Double(Vec<f64>),
}

impl Values {
    fn nc_type(&self) -> u32 {
        match self {
            Values::Short(_) => 3,
            Values::Float(_) => 5,
            Values::Double(_) => 6,
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Values::Short(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            Values::Float(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
            Values::Double(v) => v.iter().flat_map(|x| x.to_be_bytes()).collect(),
        }
    }
}

enum Attribute {
    Text(&'static str),
    Short(i16),
    Float(f32),
}

struct Variable {
    name: &'static str,
    dimensions: Vec<u32>,
    attributes: Vec<(&'static str, Attribute)>,
    values: Values,
}

fn put_name(out: &mut Vec<u8>, name: &str) {
    out.extend((name.len() as u32).to_be_bytes());
    out.extend(name.as_bytes());
    pad(out);
}

fn pad(out: &mut Vec<u8>) {
    while !out.len().is_multiple_of(4) {
        out.push(0);
    }
}

fn classic_netcdf(
    dimensions: &[(&str, usize)],
    record_dimension: Option<u32>,
    variables: &[Variable],
) -> Vec<u8> {
    // the unlimited dimension's size in `dimensions` is the record count
    let record_count = record_dimension.map_or(0, |d| dimensions[d as usize].1);
    let is_record = |v: &Variable| {
        record_dimension.is_some() && v.dimensions.first() == record_dimension.as_ref()
    };
    let mut out = b"CDF\x01".to_vec();
    out.extend((record_count as u32).to_be_bytes());
    out.extend(10u32.to_be_bytes());
    out.extend((dimensions.len() as u32).to_be_bytes());
    for (index, (name, size)) in dimensions.iter().enumerate() {
        put_name(&mut out, name);
        let stored = if Some(index as u32) == record_dimension {
            0
        } else {
            *size
        };
        out.extend((stored as u32).to_be_bytes());
    }
    out.extend([0u8; 8]);
    out.extend(11u32.to_be_bytes());
    out.extend((variables.len() as u32).to_be_bytes());
    let mut begin_positions = Vec::new();
    for variable in variables {
        put_name(&mut out, variable.name);
        out.extend((variable.dimensions.len() as u32).to_be_bytes());
        for dimension in &variable.dimensions {
            out.extend(dimension.to_be_bytes());
        }
        match variable.attributes.len() {
            0 => out.extend([0u8; 8]),
            count => {
                out.extend(12u32.to_be_bytes());
                out.extend((count as u32).to_be_bytes());
            }
        }
        for (name, attribute) in &variable.attributes {
            put_name(&mut out, name);
            let (nc_type, count, bytes) = match attribute {
                Attribute::Text(text) => (2u32, text.len(), text.as_bytes().to_vec()),
                Attribute::Short(v) => (3, 1, v.to_be_bytes().to_vec()),
                Attribute::Float(v) => (5, 1, v.to_be_bytes().to_vec()),
            };
            out.extend(nc_type.to_be_bytes());
            out.extend((count as u32).to_be_bytes());
            out.extend(bytes);
            pad(&mut out);
        }
        out.extend(variable.values.nc_type().to_be_bytes());
        out.extend(0u32.to_be_bytes());
        begin_positions.push(out.len());
        out.extend(0u32.to_be_bytes());
    }
    let mut begins = vec![0usize; variables.len()];
    for (index, variable) in variables.iter().enumerate() {
        if !is_record(variable) {
            begins[index] = out.len();
            out.extend(variable.values.bytes());
            pad(&mut out);
        }
    }
    let records: Vec<usize> = (0..variables.len())
        .filter(|i| is_record(&variables[*i]))
        .collect();
    let single = records.len() == 1;
    let record_start = out.len();
    let mut within_record = 0;
    for &index in &records {
        begins[index] = record_start + within_record;
        let slab = variables[index].values.bytes().len() / record_count;
        within_record += if single {
            slab
        } else {
            slab.next_multiple_of(4)
        };
    }
    for record in 0..record_count {
        for &index in &records {
            let bytes = variables[index].values.bytes();
            let slab = bytes.len() / record_count;
            out.extend(&bytes[record * slab..(record + 1) * slab]);
            if !single {
                pad(&mut out);
            }
        }
    }
    for (position, begin) in begin_positions.into_iter().zip(begins) {
        out[position..position + 4].copy_from_slice(&(begin as u32).to_be_bytes());
    }
    out
}

fn write(dir: &Path, bytes: &[u8]) -> PathBuf {
    let path = dir.join("fixture.nc");
    std::fs::write(&path, bytes).unwrap();
    path
}

fn coordinate(
    name: &'static str,
    dimension: u32,
    units: &'static str,
    values: Vec<f32>,
) -> Variable {
    Variable {
        name,
        dimensions: vec![dimension],
        attributes: vec![("units", Attribute::Text(units))],
        values: Values::Float(values),
    }
}

fn request(bbox: Bbox, resolution: f64, time: Option<TimeInterval>) -> WindowReq {
    WindowReq {
        bbox,
        resolution,
        time,
    }
}

async fn read(src: &NetcdfSrc, req: &WindowReq) -> RasterChunk {
    src.read(req).await.unwrap().into_raster().unwrap()
}

fn interval(start: &str, end: &str) -> TimeInterval {
    TimeInterval::new(parse_rfc3339(start).unwrap(), parse_rfc3339(end).unwrap())
}

fn assert_values(chunk: &RasterChunk, expected: &[f64]) {
    let actual = chunk.bands.bands()[0].data();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.is_nan() && e.is_nan()) || (a - e).abs() < 1e-9,
            "pixel {i}: read {a}, expected {e}"
        );
    }
}

#[tokio::test]
async fn south_to_north_rows_flip_and_longitudes_past_180_rotate() {
    // stored rows run south to north, columns 45, 135, 225 and 315 east
    let stored: Vec<f32> = (0..2)
        .flat_map(|row| (0..4).map(move |column| (10 * row + column) as f32))
        .collect();
    let bytes = classic_netcdf(
        &[("lat", 2), ("lon", 4)],
        None,
        &[
            coordinate("lat", 0, "degrees_north", vec![-45.0, 45.0]),
            coordinate("lon", 1, "degrees_east", vec![45.0, 135.0, 225.0, 315.0]),
            Variable {
                name: "sst",
                dimensions: vec![0, 1],
                attributes: vec![],
                values: Values::Float(stored),
            },
        ],
    );
    let dir = tempfile::tempdir().unwrap();
    let src = NetcdfSrc::open(
        write(dir.path(), &bytes),
        "sst",
        &BTreeMap::new(),
        None,
        None,
        ValueKind::Continuous,
    )
    .unwrap();
    let grid = src.grid();
    assert_eq!(
        (grid.origin_x, grid.origin_y, grid.base_resolution),
        (-180.0, 90.0, 90.0)
    );
    assert!(!src.time_varying());
    let chunk = read(
        &src,
        &request(Bbox::new(-180.0, -90.0, 180.0, 90.0), 90.0, None),
    )
    .await;
    assert_eq!(chunk.crs, Crs::WGS84);
    // north row first, columns from -135 east: stored 225, 315, 45, 135
    assert_values(&chunk, &[12.0, 13.0, 10.0, 11.0, 2.0, 3.0, 0.0, 1.0]);
}

const PACKED_FILL: i16 = -32767;
const PACKED_MISSING: i16 = -1;

fn packed_time_series() -> Vec<u8> {
    // t2m's nine shorts per record pad to twenty bytes beside the time record variable
    let mut packed = Vec::new();
    for step in 0..3i16 {
        for cell in 0..9i16 {
            packed.push(match (step, cell) {
                (1, 4) => PACKED_FILL,
                (2, 8) => PACKED_MISSING,
                _ => step * 100 + cell,
            });
        }
    }
    classic_netcdf(
        &[("time", 3), ("lat", 3), ("lon", 3)],
        Some(0),
        &[
            coordinate("lat", 1, "degrees_north", vec![50.0, 40.0, 30.0]),
            coordinate("lon", 2, "degrees_east", vec![-10.0, 0.0, 10.0]),
            Variable {
                name: "time",
                dimensions: vec![0],
                attributes: vec![
                    ("units", Attribute::Text("hours since 2024-01-01 00:00:00")),
                    ("calendar", Attribute::Text("standard")),
                ],
                values: Values::Double(vec![0.0, 6.0, 12.0]),
            },
            Variable {
                name: "t2m",
                dimensions: vec![0, 1, 2],
                attributes: vec![
                    ("scale_factor", Attribute::Float(0.5)),
                    ("add_offset", Attribute::Float(100.0)),
                    ("_FillValue", Attribute::Short(PACKED_FILL)),
                    ("missing_value", Attribute::Short(PACKED_MISSING)),
                ],
                values: Values::Short(packed),
            },
        ],
    )
}

fn unpacked(step: i16) -> Vec<f64> {
    (0..9i16)
        .map(|cell| match (step, cell) {
            (1, 4) | (2, 8) => f64::NAN,
            _ => f64::from(step * 100 + cell) * 0.5 + 100.0,
        })
        .collect()
}

#[tokio::test]
async fn packed_time_steps_resolve_per_pull_interval() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), &packed_time_series());
    let day = interval("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z");
    let src = NetcdfSrc::open(
        &path,
        "t2m",
        &BTreeMap::new(),
        None,
        Some(day),
        ValueKind::Continuous,
    )
    .unwrap();
    assert!(src.time_varying());
    let hour = 3_600_000;
    let start = parse_rfc3339("2024-01-01T00:00:00Z").unwrap();
    assert_eq!(
        src.step_instants_ms(),
        &[start, start + 6 * hour, start + 12 * hour]
    );
    let bbox = Bbox::new(-15.0, 25.0, 15.0, 55.0);

    // the layer's own interval holds every step, the last one wins
    assert_values(&read(&src, &request(bbox, 10.0, None)).await, &unpacked(2));
    let around_six = interval("2024-01-01T05:00:00Z", "2024-01-01T07:00:00Z");
    assert_values(
        &read(&src, &request(bbox, 10.0, Some(around_six))).await,
        &unpacked(1),
    );
    let first_two = interval("2024-01-01T00:00:00Z", "2024-01-01T07:00:00Z");
    assert_values(
        &read(&src, &request(bbox, 10.0, Some(first_two))).await,
        &unpacked(1),
    );
    let first = interval("2024-01-01T00:00:00Z", "2024-01-01T06:00:00Z");
    assert_values(
        &read(&src, &request(bbox, 10.0, Some(first))).await,
        &unpacked(0),
    );
    let february = interval("2024-02-01T00:00:00Z", "2024-02-02T00:00:00Z");
    let empty = read(&src, &request(bbox, 10.0, Some(february))).await;
    assert_values(&empty, &[f64::NAN; 9]);
}

#[test]
fn a_time_dimension_needs_a_datetime() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), &packed_time_series());
    let error = NetcdfSrc::open(
        &path,
        "t2m",
        &BTreeMap::new(),
        None,
        None,
        ValueKind::Continuous,
    )
    .err()
    .expect("open without datetime fails");
    assert!(error.to_string().contains("datetime"), "{error}");
}

fn levelled() -> Vec<u8> {
    let values: Vec<f32> = (0..2)
        .flat_map(|level| (0..4).map(move |cell| (level * 10 + cell) as f32))
        .collect();
    classic_netcdf(
        &[("level", 2), ("lat", 2), ("lon", 2)],
        None,
        &[
            coordinate("level", 0, "hPa", vec![850.0, 500.0]),
            coordinate("lat", 1, "degrees_north", vec![1.0, 0.0]),
            coordinate("lon", 2, "degrees_east", vec![0.0, 1.0]),
            Variable {
                name: "u",
                dimensions: vec![0, 1, 2],
                attributes: vec![],
                values: Values::Float(values),
            },
        ],
    )
}

#[tokio::test]
async fn an_extra_dimension_needs_a_select_index() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), &levelled());
    let error = NetcdfSrc::open(
        &path,
        "u",
        &BTreeMap::new(),
        None,
        None,
        ValueKind::Continuous,
    )
    .err()
    .expect("open without select fails");
    assert!(error.to_string().contains("level"), "{error}");

    let select = BTreeMap::from([("level".to_string(), 1)]);
    let src = NetcdfSrc::open(&path, "u", &select, None, None, ValueKind::Continuous).unwrap();
    let chunk = read(&src, &request(Bbox::new(-0.5, -0.5, 1.5, 1.5), 1.0, None)).await;
    assert_values(&chunk, &[10.0, 11.0, 12.0, 13.0]);
}

fn global_grid(first_lon: f32, marked_column: usize) -> Vec<u8> {
    let lons: Vec<f32> = (0..360).map(|i| first_lon + i as f32).collect();
    let lats: Vec<f32> = (0..21).map(|i| 10.0 - i as f32).collect();
    let values: Vec<f32> = lats
        .iter()
        .flat_map(|_| (0..360).map(|column| if column == marked_column { 100.0 } else { 0.0 }))
        .collect();
    classic_netcdf(
        &[("lat", 21), ("lon", 360)],
        None,
        &[
            coordinate("lat", 0, "degrees_north", lats),
            coordinate("lon", 1, "degrees_east", lons),
            Variable {
                name: "u",
                dimensions: vec![0, 1],
                attributes: vec![],
                values: Values::Float(values),
            },
        ],
    )
}

#[tokio::test]
async fn a_cell_centred_on_180_serves_both_sides_of_the_antimeridian() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), &global_grid(0.0, 180));
    let src = NetcdfSrc::open(
        &path,
        "u",
        &BTreeMap::new(),
        None,
        None,
        ValueKind::Continuous,
    )
    .unwrap();
    // the cell centred on 180 sits at both ends, so the raster spans -180.5 to 180.5
    assert_eq!(src.grid().origin_x, -180.5);
    let row = read(
        &src,
        &request(Bbox::new(-180.5, -0.5, 180.5, 0.5), 1.0, None),
    )
    .await;
    let values = row.bands.bands()[0].data();
    assert_eq!(values.len(), 361);
    assert_eq!((values[0], values[360]), (100.0, 100.0));

    let mut graph = Graph::new();
    let source = graph.add_source(Box::new(src));
    let mercator = graph.add_transform(source, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
    let engine = Engine::new(graph, 64 << 20).unwrap();
    // zoom 8 tiles 255 and 0 touch 180 from the west and -180 from the east
    let pixel_size = 360.0 / 256.0 / 256.0;
    for (x, lon) in [(255u32, 179.98f64), (0, -179.98)] {
        let tile = render_tile_at(&engine, mercator, XyzTile { z: 8, x, y: 127 }, None)
            .await
            .unwrap();
        let tile_west = -180.0 + f64::from(x) * 256.0 * pixel_size;
        let column = ((lon - tile_west) / pixel_size) as usize;
        let value = tile.bands.bands()[0].data()[128 * 256 + column];
        assert!(
            value > 90.0,
            "lon {lon}: served {value}, the 180 cell holds 100"
        );
    }
}

#[tokio::test]
async fn a_global_grid_with_edges_on_180_keeps_its_columns() {
    let dir = tempfile::tempdir().unwrap();
    let path = write(dir.path(), &global_grid(0.5, 179));
    let src = NetcdfSrc::open(
        &path,
        "u",
        &BTreeMap::new(),
        None,
        None,
        ValueKind::Continuous,
    )
    .unwrap();
    assert_eq!(src.grid().origin_x, -180.0);
    let row = read(
        &src,
        &request(Bbox::new(-180.0, -0.5, 180.0, 0.5), 1.0, None),
    )
    .await;
    let values = row.bands.bands()[0].data();
    assert_eq!(values.len(), 360);
    assert_eq!((values[0], values[359]), (0.0, 100.0));
}
