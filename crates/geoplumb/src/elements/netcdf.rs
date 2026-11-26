use std::collections::{BTreeMap, VecDeque};
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use terrano_core::Raster;
use terrano_core::netcdf::{
    AttributeValue, NetCdfMetadata, Variable, read_netcdf_metadata, read_netcdf_values,
};

use crate::bands::RasterBands;
use crate::caps::{Constraint, Crs, ValueKind};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::elements::RasterSrc;
use crate::elements::file_stamp;
use crate::elements::stac::{format_rfc3339, parse_rfc3339};
use crate::elements::stored_grid::{CELL_TOLERANCE, StoredGrid, north_up_source};
use crate::error::{Error, Result};
use crate::window::{GridSpec, TimeInterval, WindowReq};

const DECODED_STEP_CAPACITY: usize = 8;

const GREGORIAN_START: &str = "1582-10-15T00:00:00Z";

type DecodedStep = Arc<Mutex<Option<Arc<RasterSrc>>>>;

pub struct NetcdfSrc {
    grid: Arc<NetcdfGrid>,
    time: Option<TimeSteps>,
    resident_step: usize,
    resident: Arc<RasterSrc>,
    decoded_steps: Mutex<VecDeque<(usize, DecodedStep)>>,
}

struct TimeSteps {
    datetime: TimeInterval,
    instants_ms: Vec<i64>,
}

struct NetcdfGrid {
    path: PathBuf,
    variable: String,
    select: BTreeMap<String, usize>,
    crs: Crs,
    value_kind: ValueKind,
    stored: StoredGrid,
    metadata: NetCdfMetadata,
    select_indices: Vec<usize>,
    time_position: Option<usize>,
}

enum Axis {
    Latitude,
    Longitude,
    ProjectionX,
    ProjectionY,
    Time,
    Other,
}

impl NetcdfSrc {
    pub fn open(
        path: impl AsRef<Path>,
        variable: &str,
        select: &BTreeMap<String, usize>,
        crs: Option<Crs>,
        datetime: Option<TimeInterval>,
        value_kind: ValueKind,
    ) -> Result<NetcdfSrc> {
        let path = path.as_ref();
        let named = |detail: String| {
            Error::Source(format!("{} variable {variable}: {detail}", path.display()))
        };
        let file = File::open(path).map_err(|e| named(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let metadata = read_netcdf_metadata(&mut reader).map_err(|e| named(e.to_string()))?;
        let target = metadata
            .variables
            .iter()
            .find(|v| v.name == variable)
            .ok_or_else(|| named("the file has no such variable".into()))?;
        let sizes = &target.shape;
        let dimension_count = target.dimensions.len();
        if dimension_count < 2 {
            return Err(named(format!(
                "has {dimension_count} dimensions, a grid needs y and x"
            )));
        }
        let y_name = &target.dimensions[dimension_count - 2];
        let x_name = &target.dimensions[dimension_count - 1];
        let coordinate_of = |dimension: &str| {
            coordinate_variable(&metadata, dimension)
                .ok_or_else(|| named(format!("dimension {dimension} has no coordinate variable")))
        };
        let x_coordinate = coordinate_of(x_name.as_str())?;
        let y_coordinate = coordinate_of(y_name.as_str())?;
        let (crs, x_is_longitude) = match (
            axis_of(x_coordinate).map_err(named)?,
            axis_of(y_coordinate).map_err(named)?,
        ) {
            (Axis::Longitude, Axis::Latitude) => match crs {
                None | Some(Crs::WGS84) => (Crs::WGS84, true),
                Some(other) => {
                    return Err(named(format!(
                        "lat/lon coordinates are {}, the layer names {other}",
                        Crs::WGS84
                    )));
                }
            },
            (Axis::ProjectionX, Axis::ProjectionY) => (
                crs.ok_or_else(|| {
                    named("x/y projection coordinates need a crs in the layer".into())
                })?,
                false,
            ),
            _ => {
                return Err(named(format!(
                    "the last two dimensions {y_name} and {x_name} are neither latitude and \
                     longitude nor projection y and x"
                )));
            }
        };
        let mut values_of = |coordinate: &Variable| {
            read_netcdf_values(&mut reader, &metadata, &coordinate.name, &[])
                .map_err(|e| named(e.to_string()))
        };
        let x_values = values_of(x_coordinate)?;
        let y_values = values_of(y_coordinate)?;
        let (first_column_x, column_step) = regular_axis(x_name, &x_values).map_err(named)?;
        let (first_row_y, row_step) = regular_axis(y_name, &y_values).map_err(named)?;
        let cell_size = column_step.abs();
        let longest_side = x_values.len().max(y_values.len()) as f64;
        if (cell_size - row_step.abs()).abs() * longest_side > CELL_TOLERANCE * cell_size {
            return Err(named(format!(
                "cells are {cell_size} by {}, the engine takes square cells",
                row_step.abs()
            )));
        }
        let stored = StoredGrid {
            columns: x_values.len(),
            rows: y_values.len(),
            cell_size,
            first_column_x,
            first_row_y,
            columns_run_west: column_step < 0.0,
            rows_run_north: row_step > 0.0,
            x_is_longitude,
        };

        let leading = &target.dimensions[..dimension_count - 2];
        let mut time_position = None;
        let mut fixed_indices = vec![0usize; leading.len()];
        for (position, dimension) in leading.iter().enumerate() {
            let coordinate = coordinate_variable(&metadata, dimension);
            let is_time = match coordinate {
                Some(coordinate) => matches!(axis_of(coordinate).map_err(named)?, Axis::Time),
                None => false,
            };
            if is_time {
                if time_position.is_some() {
                    return Err(named("has two time dimensions".into()));
                }
                time_position = Some(position);
                continue;
            }
            let index = *select.get(dimension).ok_or_else(|| {
                named(format!(
                    "has a {dimension} dimension of size {}, name an index for it in select",
                    sizes[position]
                ))
            })?;
            if index >= sizes[position] {
                return Err(named(format!(
                    "select {dimension} = {index} is past the dimension's size {}",
                    sizes[position]
                )));
            }
            fixed_indices[position] = index;
        }
        for name in select.keys() {
            let selectable = leading
                .iter()
                .enumerate()
                .any(|(position, dimension)| dimension == name && time_position != Some(position));
            if !selectable {
                return Err(named(format!(
                    "select names {name}, which is not a dimension besides time, y and x"
                )));
            }
        }

        let time = match (time_position, datetime) {
            (None, None) => None,
            (None, Some(_)) => {
                return Err(named(
                    "has no time dimension for the layer's datetime".into(),
                ));
            }
            (Some(_), None) => {
                return Err(named(
                    "has a time dimension, the layer needs datetime = \"start/end\"".into(),
                ));
            }
            (Some(position), Some(datetime)) => {
                let coordinate = coordinate_of(leading[position].as_str())?;
                let values = values_of(coordinate)?;
                let instants_ms = time_instants(coordinate, &values).map_err(named)?;
                Some(TimeSteps {
                    datetime,
                    instants_ms,
                })
            }
        };
        let step_count = time.as_ref().map_or(1, |time| time.instants_ms.len());
        if step_count == 0 {
            return Err(named("has no time steps".into()));
        }
        let resident_step = match &time {
            None => 0,
            Some(time) => step_in(&time.instants_ms, time.datetime).ok_or_else(|| {
                named(format!(
                    "datetime {} holds no time step, the file runs {} to {}",
                    time.datetime.to_stac_datetime(),
                    format_rfc3339(time.instants_ms[0]),
                    format_rfc3339(time.instants_ms[time.instants_ms.len() - 1])
                ))
            })?,
        };
        let grid = Arc::new(NetcdfGrid {
            path: path.to_path_buf(),
            variable: variable.to_string(),
            select: select.clone(),
            crs,
            value_kind,
            stored,
            metadata: metadata.clone(),
            select_indices: fixed_indices,
            time_position,
        });
        let resident = Arc::new(grid.decode(resident_step)?);
        Ok(NetcdfSrc {
            grid,
            time,
            resident_step,
            resident,
            decoded_steps: Mutex::new(VecDeque::new()),
        })
    }

    pub fn step_instants_ms(&self) -> &[i64] {
        match &self.time {
            Some(time) => &time.instants_ms,
            None => &[],
        }
    }

    fn decoded_step(&self, step: usize) -> DecodedStep {
        let mut decoded = self.decoded_steps.lock().unwrap();
        if let Some(position) = decoded.iter().position(|(s, _)| *s == step) {
            let entry = decoded.remove(position).expect("position is in range");
            decoded.push_back(entry.clone());
            return entry.1;
        }
        let slot = DecodedStep::default();
        decoded.push_back((step, slot.clone()));
        if decoded.len() > DECODED_STEP_CAPACITY {
            decoded.pop_front();
        }
        slot
    }

    async fn step_source(&self, step: usize) -> Result<Arc<RasterSrc>> {
        if step == self.resident_step {
            return Ok(self.resident.clone());
        }
        let slot = self.decoded_step(step);
        let grid = self.grid.clone();
        crate::engine::offload(move || -> Result<Arc<RasterSrc>> {
            let mut decoded = slot.lock().unwrap();
            if let Some(source) = decoded.as_ref() {
                return Ok(source.clone());
            }
            let source = Arc::new(grid.decode(step)?);
            *decoded = Some(source.clone());
            Ok(source)
        })
        .await
    }
}

fn step_in(instants_ms: &[i64], interval: TimeInterval) -> Option<usize> {
    instants_ms
        .iter()
        .enumerate()
        .filter(|(_, instant)| (interval.start_ms..interval.end_ms).contains(*instant))
        .max_by_key(|(step, instant)| (**instant, *step))
        .map(|(step, _)| step)
}

fn empty_window(req: &WindowReq, crs: Crs) -> Result<RasterChunk> {
    let columns = (req.bbox.width() / req.resolution).round() as usize;
    let rows = (req.bbox.height() / req.resolution).round() as usize;
    let band = Raster::from_vec(
        columns,
        rows,
        vec![f64::NAN; columns * rows],
        req.resolution,
        f64::NAN,
    )?;
    Ok(RasterChunk {
        bands: RasterBands::from_rasters(vec![band])?,
        bbox: req.bbox,
        resolution: req.resolution,
        crs,
    })
}

impl Source for NetcdfSrc {
    fn identity(&self) -> String {
        let grid = &self.grid;
        format!(
            "NetcdfSrc path={:?} variable={:?} select={:?} crs={} value_kind={:?} datetime={:?} {}",
            grid.path,
            grid.variable,
            grid.select,
            grid.crs,
            grid.value_kind,
            self.time.as_ref().map(|time| time.datetime),
            file_stamp(&grid.path)
        )
    }

    fn constraint(&self) -> Constraint {
        self.resident.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.resident.grid()
    }

    fn time_varying(&self) -> bool {
        self.time.is_some()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        Box::pin(async move {
            let step = match &self.time {
                None => Some(self.resident_step),
                Some(time) => step_in(&time.instants_ms, req.time.unwrap_or(time.datetime)),
            };
            let Some(step) = step else {
                return Ok(Chunk::Raster(empty_window(req, self.grid.crs)?));
            };
            let source = self.step_source(step).await?;
            source.read(req).await
        })
    }
}

impl NetcdfGrid {
    fn decode(&self, step: usize) -> Result<RasterSrc> {
        let named = |detail: String| {
            Error::Source(format!(
                "{} variable {} step {step}: {detail}",
                self.path.display(),
                self.variable
            ))
        };
        let file = File::open(&self.path).map_err(|e| named(e.to_string()))?;
        let mut indices = self.select_indices.clone();
        if let Some(position) = self.time_position {
            indices[position] = step;
        }
        let values = read_netcdf_values(
            &mut BufReader::new(file),
            &self.metadata,
            &self.variable,
            &indices,
        )
        .map_err(|e| named(e.to_string()))?;
        north_up_source(&values, &self.stored, self.crs, self.value_kind)
    }
}

fn coordinate_variable<'a>(metadata: &'a NetCdfMetadata, dimension: &str) -> Option<&'a Variable> {
    metadata
        .variables
        .iter()
        .find(|v| v.name == dimension && v.dimensions.len() == 1 && v.dimensions[0] == dimension)
}

fn axis_of(coordinate: &Variable) -> core::result::Result<Axis, String> {
    let standard_name = text_attribute(coordinate, "standard_name")?;
    let units = text_attribute(coordinate, "units")?;
    if let Some(axis) = standard_name.and_then(|name| match name {
        "latitude" => Some(Axis::Latitude),
        "longitude" => Some(Axis::Longitude),
        "projection_x_coordinate" => Some(Axis::ProjectionX),
        "projection_y_coordinate" => Some(Axis::ProjectionY),
        "time" => Some(Axis::Time),
        _ => None,
    }) {
        return Ok(axis);
    }
    if let Some(units) = units {
        // cf conventions section 4.1 lists these spellings
        match units {
            "degrees_north" | "degree_north" | "degree_N" | "degrees_N" | "degreeN"
            | "degreesN" => return Ok(Axis::Latitude),
            "degrees_east" | "degree_east" | "degree_E" | "degrees_E" | "degreeE" | "degreesE" => {
                return Ok(Axis::Longitude);
            }
            _ if units.contains(" since ") => return Ok(Axis::Time),
            _ => {}
        }
    }
    Ok(match coordinate.name.to_ascii_lowercase().as_str() {
        "lat" | "latitude" => Axis::Latitude,
        "lon" | "longitude" => Axis::Longitude,
        "x" => Axis::ProjectionX,
        "y" => Axis::ProjectionY,
        _ => Axis::Other,
    })
}

fn regular_axis(name: &str, values: &[f64]) -> core::result::Result<(f64, f64), String> {
    if values.len() < 2 {
        return Err(format!("coordinate {name} has {} values", values.len()));
    }
    let first = values[0];
    let step = (values[values.len() - 1] - first) / (values.len() - 1) as f64;
    if !step.is_finite() || step == 0.0 {
        return Err(format!("coordinate {name} has no spacing"));
    }
    let irregular = values
        .iter()
        .enumerate()
        .any(|(i, v)| (v - (first + i as f64 * step)).abs() > CELL_TOLERANCE * step.abs());
    if irregular {
        return Err(format!("coordinate {name} is not regularly spaced"));
    }
    Ok((first, step))
}

fn time_instants(coordinate: &Variable, values: &[f64]) -> core::result::Result<Vec<i64>, String> {
    let units = text_attribute(coordinate, "units")?
        .ok_or_else(|| format!("time coordinate {} has no units", coordinate.name))?;
    let (unit_ms, reference_ms) = parse_time_units(units)?;
    let calendar = text_attribute(coordinate, "calendar")?.map(str::to_ascii_lowercase);
    let proleptic = match calendar.as_deref() {
        None | Some("standard") | Some("gregorian") => false,
        Some("proleptic_gregorian") => true,
        Some(other) => {
            return Err(format!(
                "time calendar {other} is not read, only standard and proleptic_gregorian"
            ));
        }
    };
    if let Some(bad) = values.iter().find(|v| !v.is_finite()) {
        return Err(format!("time coordinate holds {bad}"));
    }
    let instants: Vec<i64> = values
        .iter()
        .map(|v| reference_ms + (v * unit_ms).round() as i64)
        .collect();
    let gregorian_start = parse_rfc3339(GREGORIAN_START).map_err(|e| e.to_string())?;
    let julian = reference_ms < gregorian_start || instants.iter().any(|t| *t < gregorian_start);
    if !proleptic && julian {
        return Err(format!(
            "time units {units:?} reach before {GREGORIAN_START}, where the standard calendar is julian"
        ));
    }
    Ok(instants)
}

fn parse_time_units(units: &str) -> core::result::Result<(f64, i64), String> {
    let invalid = || format!("time units {units:?} are not \"<unit> since <date> <time>\"");
    let (unit, reference) = units.split_once(" since ").ok_or_else(invalid)?;
    let unit_ms = match unit.trim().to_ascii_lowercase().as_str() {
        "days" | "day" | "d" => 86_400_000.0,
        "hours" | "hour" | "hr" | "h" => 3_600_000.0,
        "minutes" | "minute" | "min" => 60_000.0,
        "seconds" | "second" | "sec" | "s" => 1_000.0,
        other => {
            return Err(format!(
                "time unit {other:?} is not days, hours, minutes or seconds"
            ));
        }
    };
    let mut parts = reference.split([' ', 'T']).filter(|part| !part.is_empty());
    let date = parts.next().ok_or_else(invalid)?;
    let time = parts.next().unwrap_or("00:00:00");
    let zone = parts.next();
    if parts.next().is_some()
        || !matches!(zone, None | Some("Z" | "UTC" | "GMT" | "+00:00" | "+0:00"))
    {
        return Err(invalid());
    }
    let time = time.strip_suffix('Z').unwrap_or(time);
    let number = |text: &str| match !text.is_empty() && text.bytes().all(|b| b.is_ascii_digit()) {
        true => text.parse::<u32>().ok(),
        false => None,
    };
    let mut date_parts = date.split('-').map(number);
    let mut time_parts = time.split(':');
    let (year, month, day) = match (
        date_parts.next(),
        date_parts.next(),
        date_parts.next(),
        date_parts.next(),
    ) {
        (Some(Some(y)), Some(Some(m)), Some(Some(d)), None) => (y, m, d),
        _ => return Err(invalid()),
    };
    let hour = time_parts.next().and_then(number).ok_or_else(invalid)?;
    let minute = time_parts.next().and_then(number).ok_or_else(invalid)?;
    let seconds = time_parts.next().unwrap_or("0");
    if time_parts.next().is_some() {
        return Err(invalid());
    }
    let (whole_seconds, fraction) = seconds.split_once('.').unwrap_or((seconds, ""));
    let whole_seconds = number(whole_seconds).ok_or_else(invalid)?;
    let fraction = match fraction {
        "" => String::new(),
        digits if number(digits).is_some() => format!(".{digits}"),
        _ => return Err(invalid()),
    };
    let rfc3339 = format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{whole_seconds:02}{fraction}Z"
    );
    let reference_ms = parse_rfc3339(&rfc3339).map_err(|_| invalid())?;
    Ok((unit_ms, reference_ms))
}

fn text_attribute<'a>(
    variable: &'a Variable,
    name: &str,
) -> core::result::Result<Option<&'a str>, String> {
    match variable.attributes.get(name) {
        None => Ok(None),
        Some(AttributeValue::Text(text)) => Ok(Some(text.as_str())),
        Some(_) => Err(format!("{} attribute {name} is not text", variable.name)),
    }
}
