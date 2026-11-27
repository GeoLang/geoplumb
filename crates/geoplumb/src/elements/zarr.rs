use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io::Read;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;

use futures::future::BoxFuture;
use ruzstd::decoding::StreamingDecoder;
use serde_json::Value;
use terrano_core::{RangeRead, SampleFormat};
use zarrs::array::codec::api::{
    CodecPluginV2, CodecPluginV3, CodecTraitsV2, CodecTraitsV3, PartialDecoderCapability,
    PartialEncoderCapability,
};
use zarrs::array::{
    Array, ArrayBytesRaw, ArrayCreateError, ArrayError, ArrayMetadata, BytesRepresentation,
    BytesToBytesCodecTraits, Codec, CodecError, CodecMetadataOptions, CodecOptions, CodecTraits,
    ElementOwned, FillValueMetadata, RecommendedConcurrency,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::metadata::Configuration;
use zarrs::metadata::v2::MetadataV2;
use zarrs::metadata::v3::MetadataV3;
use zarrs::plugin::{ExtensionName, PluginCreateError, ZarrVersion};
use zarrs::storage::byte_range::{ByteRange, ByteRangeIterator};
use zarrs::storage::{
    Bytes, MaybeBytes, MaybeBytesIterator, ReadableStorage, ReadableStorageTraits, StorageError,
    StoreKey,
};

use crate::bands::{RasterBands, narrowed};
use crate::caps::{
    BandDtypes, BandKinds, CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange,
    SetField, ValueKind,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::elements::cog::{HttpRange, decimate_band};
use crate::elements::netcdf::{
    Axis, LeadingIndices, TimeSteps, coordinate_axis, leading_indices, regular_grid, step_in,
    time_instants, time_pairing, time_steps,
};
use crate::elements::stored_grid::{CELL_TOLERANCE, StoredGrid};
use crate::error::{Error, Result};
use crate::window::{GridSpec, TimeInterval, WindowReq};

type ZarrArray = Array<dyn ReadableStorageTraits>;

pub enum ZarrStore {
    Path(PathBuf),
    Url(String),
}

pub struct ZarrLayer {
    pub array: String,
    pub x_coordinate: Option<String>,
    pub y_coordinate: Option<String>,
    pub select: BTreeMap<String, usize>,
    pub crs: Option<Crs>,
    pub datetime: Option<TimeInterval>,
    pub value_kind: ValueKind,
}

pub struct ZarrSrc {
    dataset: Arc<ZarrDataset>,
    time: Option<TimeSteps>,
    identity: String,
}

struct ZarrDataset {
    label: String,
    array: ZarrArray,
    stored_type: StoredType,
    grid: StoredGrid,
    origin_x: f64,
    origin_y: f64,
    crs: Crs,
    value_kind: ValueKind,
    dtype: Dtype,
    packing: Packing,
    fixed_indices: Vec<usize>,
    time_position: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum StoredType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
}

struct Packing {
    scale_factor: Option<f64>,
    add_offset: Option<f64>,
    missing_values: Vec<f64>,
}

impl ZarrSrc {
    // blocking, an http store fetches metadata and coordinates here
    pub fn open(store: &ZarrStore, layer: &ZarrLayer) -> Result<ZarrSrc> {
        let array_path = format!("/{}", layer.array.trim_start_matches('/'));
        let location = match store {
            ZarrStore::Path(path) => path.display().to_string(),
            ZarrStore::Url(url) => url.clone(),
        };
        let label = format!("{location} array {array_path}");
        let named = |detail: String| Error::Source(format!("{label}: {detail}"));
        let storage: ReadableStorage = match store {
            ZarrStore::Path(path) => {
                Arc::new(FilesystemStore::new(path).map_err(|e| named(e.to_string()))?)
            }
            ZarrStore::Url(url) => Arc::new(ZarrHttpStore {
                base_url: url.trim_end_matches('/').to_string(),
            }),
        };
        let array =
            ZarrArray::open(storage.clone(), &array_path).map_err(|e| named(e.to_string()))?;
        let stored_type = StoredType::of(&array).map_err(named)?;
        let shape: Vec<usize> = array.shape().iter().map(|&size| size as usize).collect();
        let dimension_count = shape.len();
        if dimension_count < 2 {
            return Err(named(format!(
                "has {dimension_count} dimensions, a grid needs y and x"
            )));
        }
        let names = dimension_names(&array).map_err(named)?;
        let (group, _) = array_path
            .rsplit_once('/')
            .expect("the array path starts with a slash");
        let y_name = layer
            .y_coordinate
            .clone()
            .or_else(|| names[dimension_count - 2].clone())
            .ok_or_else(|| {
                named("the y dimension has no name, name its coordinate in y_coordinate".into())
            })?;
        let x_name = layer
            .x_coordinate
            .clone()
            .or_else(|| names[dimension_count - 1].clone())
            .ok_or_else(|| {
                named("the x dimension has no name, name its coordinate in x_coordinate".into())
            })?;
        let required_coordinate = |name: &str| {
            coordinate_array(&storage, group, name)
                .and_then(|found| {
                    found.ok_or_else(|| format!("coordinate array {name} is not in the store"))
                })
                .map_err(named)
        };
        let x_array = required_coordinate(&x_name)?;
        let y_array = required_coordinate(&y_name)?;
        let x_axis = array_axis(&x_array, &x_name).map_err(named)?;
        let y_axis = array_axis(&y_array, &y_name).map_err(named)?;
        if matches!(x_axis, Axis::Latitude) || matches!(y_axis, Axis::Longitude) {
            return Err(named(format!(
                "the last two dimensions {y_name} and {x_name} hold longitude before latitude, \
                 only latitude then longitude is read"
            )));
        }
        let geographic = matches!((&x_axis, &y_axis), (Axis::Longitude, Axis::Latitude));
        let crs = match (layer.crs, geographic) {
            (Some(crs), _) => crs,
            (None, true) => Crs::WGS84,
            (None, false) => {
                return Err(named(format!(
                    "coordinates {y_name} and {x_name} are not latitude and longitude, \
                     name the crs in the layer"
                )));
            }
        };
        let x_values =
            coordinate_values(&x_array, &x_name, shape[dimension_count - 1]).map_err(named)?;
        let y_values =
            coordinate_values(&y_array, &y_name, shape[dimension_count - 2]).map_err(named)?;
        let x_is_longitude = matches!(x_axis, Axis::Longitude);
        let grid =
            regular_grid(&x_name, &x_values, &y_name, &y_values, x_is_longitude).map_err(named)?;

        let leading_names = names[..dimension_count - 2]
            .iter()
            .enumerate()
            .map(|(position, name)| {
                name.clone()
                    .ok_or_else(|| format!("dimension {position} has no name to select it by"))
            })
            .collect::<core::result::Result<Vec<String>, String>>()
            .map_err(named)?;
        let leading_coordinates = leading_names
            .iter()
            .map(|name| coordinate_array(&storage, group, name))
            .collect::<core::result::Result<Vec<Option<ZarrArray>>, String>>()
            .map_err(named)?;
        let time_flags = leading_coordinates
            .iter()
            .zip(&leading_names)
            .map(|(coordinate, name)| match coordinate {
                Some(coordinate) => Ok(matches!(array_axis(coordinate, name)?, Axis::Time)),
                None => Ok(false),
            })
            .collect::<core::result::Result<Vec<bool>, String>>()
            .map_err(named)?;
        let LeadingIndices {
            time_position,
            fixed_indices,
        } = leading_indices(
            &leading_names,
            &shape[..dimension_count - 2],
            &time_flags,
            &layer.select,
        )
        .map_err(named)?;
        let time = match time_pairing(time_position, layer.datetime).map_err(named)? {
            None => None,
            Some((position, datetime)) => {
                let name = &leading_names[position];
                let coordinate = leading_coordinates[position]
                    .as_ref()
                    .expect("a time dimension has a coordinate array");
                let values = coordinate_values(coordinate, name, shape[position]).map_err(named)?;
                let instants_ms = time_instants(
                    name,
                    text_attribute(coordinate, name, "units").map_err(named)?,
                    text_attribute(coordinate, name, "calendar").map_err(named)?,
                    &values,
                )
                .map_err(named)?;
                Some(time_steps(datetime, instants_ms).map_err(named)?)
            }
        };

        let packing = Packing::of(&array).map_err(named)?;
        let scaled = packing.scale_factor.is_some() || packing.add_offset.is_some();
        let dtype = match (
            stored_type.format(),
            scaled,
            packing.missing_values.as_slice(),
        ) {
            (Some(format), false, _) if format.integer_range().is_none() => {
                Dtype::from_format(format, None)?
            }
            (Some(format), false, []) => Dtype::from_format(format, None)?,
            (Some(format), false, [missing]) => {
                Dtype::from_format(format, Some(*missing)).map_err(|e| named(e.to_string()))?
            }
            _ => Dtype::F64,
        };

        let cell_size = grid.cell_size;
        let west_center = match grid.columns_run_west {
            true => grid.first_column_x - (grid.columns - 1) as f64 * cell_size,
            false => grid.first_column_x,
        };
        let north_center = match grid.rows_run_north {
            true => grid.first_row_y + (grid.rows - 1) as f64 * cell_size,
            false => grid.first_row_y,
        };
        let west_edge = west_center - cell_size / 2.0;
        // pulls in EPSG:4326 name longitudes from -180 to 180
        let origin_x = match x_is_longitude && west_edge >= 180.0 - CELL_TOLERANCE * cell_size {
            true => west_edge - 360.0,
            false => west_edge,
        };

        let identity = format!(
            "ZarrSrc store={location:?} array={array_path:?} x_coordinate={x_name:?} \
             y_coordinate={y_name:?} select={:?} crs={crs} value_kind={:?} datetime={:?} metadata={}",
            layer.select,
            layer.value_kind,
            layer.datetime,
            array.metadata()
        );
        Ok(ZarrSrc {
            dataset: Arc::new(ZarrDataset {
                label,
                array,
                stored_type,
                grid,
                origin_x,
                origin_y: north_center + cell_size / 2.0,
                crs,
                value_kind: layer.value_kind,
                dtype,
                packing,
                fixed_indices,
                time_position,
            }),
            time,
            identity,
        })
    }

    pub fn step_instants_ms(&self) -> &[i64] {
        match &self.time {
            Some(time) => &time.instants_ms,
            None => &[],
        }
    }
}

impl Source for ZarrSrc {
    fn identity(&self) -> String {
        self.identity.clone()
    }

    fn constraint(&self) -> Constraint {
        let dataset = &self.dataset;
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtypes: SetField::one(BandDtypes::from(dataset.dtype)),
            bands: SetField::one(1),
            crs: SetField::one(dataset.crs),
            resolution: ResRange::at_least(dataset.grid.cell_size),
            chunk_px: SetField::Any,
            value_kinds: SetField::one(BandKinds::from(dataset.value_kind)),
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.dataset.origin_x,
            origin_y: self.dataset.origin_y,
            base_resolution: self.dataset.grid.cell_size,
            chunk_px: 256,
        }
    }

    fn time_varying(&self) -> bool {
        self.time.is_some()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let dataset = self.dataset.clone();
        let req = *req;
        let step = match &self.time {
            None => Some(0),
            Some(time) => step_in(&time.instants_ms, req.time.unwrap_or(time.datetime)),
        };
        Box::pin(async move {
            crate::engine::offload(move || dataset.read_window(&req, step).map(Chunk::Raster)).await
        })
    }
}

impl ZarrDataset {
    fn read_window(&self, req: &WindowReq, step: Option<usize>) -> Result<RasterChunk> {
        let factor = (req.resolution / self.grid.cell_size).round().max(1.0) as usize;
        let columns = (req.bbox.width() / req.resolution).round() as usize;
        let rows = (req.bbox.height() / req.resolution).round() as usize;
        let (fine_columns, fine_rows) = (columns * factor, rows * factor);
        let fine_cell = req.resolution / factor as f64;
        let fine = match step {
            None => vec![f64::NAN; fine_columns * fine_rows],
            Some(step) => self.window_values(req, step, (fine_columns, fine_rows), fine_cell)?,
        };
        let band = narrowed(self.dtype, fine_columns, fine_rows, fine_cell, fine);
        let band = match factor {
            1 => band,
            _ => decimate_band(&band, factor, self.value_kind, req.resolution),
        };
        Ok(RasterChunk {
            bands: RasterBands::new(vec![band])?,
            bbox: req.bbox,
            resolution: req.resolution,
            crs: self.crs,
        })
    }

    fn window_values(
        &self,
        req: &WindowReq,
        step: usize,
        (fine_columns, fine_rows): (usize, usize),
        fine_cell: f64,
    ) -> Result<Vec<f64>> {
        let stored_columns: Vec<Option<usize>> = (0..fine_columns)
            .map(|column| {
                stored_column(
                    &self.grid,
                    req.bbox.min_x + (column as f64 + 0.5) * fine_cell,
                )
            })
            .collect();
        let stored_rows: Vec<Option<usize>> = (0..fine_rows)
            .map(|row| stored_row(&self.grid, req.bbox.max_y - (row as f64 + 0.5) * fine_cell))
            .collect();
        let first_row = stored_rows.iter().flatten().min();
        let last_row = stored_rows.iter().flatten().max();
        let column_runs = contiguous_runs(&stored_columns);
        let (Some(&first_row), Some(&last_row)) = (first_row, last_row) else {
            return Ok(vec![f64::NAN; fine_columns * fine_rows]);
        };
        let row_span = first_row..last_row + 1;
        let blocks = column_runs
            .iter()
            .map(|run| self.read_block(step, &row_span, run))
            .collect::<Result<Vec<Vec<f64>>>>()?;
        let column_slots: Vec<Option<(usize, usize)>> = stored_columns
            .iter()
            .map(|column| {
                column.map(|column| {
                    let run = column_runs
                        .iter()
                        .position(|run| run.contains(&column))
                        .expect("every stored column is in a run");
                    (run, column - column_runs[run].start)
                })
            })
            .collect();
        let mut values = Vec::with_capacity(fine_columns * fine_rows);
        for row in &stored_rows {
            for slot in &column_slots {
                let value = match (row, slot) {
                    (Some(row), Some((run, offset))) => {
                        let width = column_runs[*run].len();
                        self.packing
                            .unpack(blocks[*run][(row - row_span.start) * width + offset])
                    }
                    _ => f64::NAN,
                };
                values.push(value);
            }
        }
        Ok(values)
    }

    fn read_block(
        &self,
        step: usize,
        rows: &Range<usize>,
        columns: &Range<usize>,
    ) -> Result<Vec<f64>> {
        let mut subset: Vec<Range<u64>> = self
            .fixed_indices
            .iter()
            .map(|&index| index as u64..index as u64 + 1)
            .collect();
        if let Some(position) = self.time_position {
            subset[position] = step as u64..step as u64 + 1;
        }
        subset.push(rows.start as u64..rows.end as u64);
        subset.push(columns.start as u64..columns.end as u64);
        self.stored_type
            .read(&self.array, &subset)
            .map_err(|e| Error::Source(format!("{}: {e}", self.label)))
    }
}

fn stored_index(first: f64, step: f64, count: usize, coordinate: f64) -> Option<usize> {
    let index = ((coordinate - first) / step).round();
    (index >= 0.0 && index < count as f64).then_some(index as usize)
}

fn stored_column(grid: &StoredGrid, x: f64) -> Option<usize> {
    let step = match grid.columns_run_west {
        true => -grid.cell_size,
        false => grid.cell_size,
    };
    let at = |x: f64| stored_index(grid.first_column_x, step, grid.columns, x);
    if !grid.x_is_longitude {
        return at(x);
    }
    at(x).or_else(|| at(x + 360.0)).or_else(|| at(x - 360.0))
}

fn stored_row(grid: &StoredGrid, y: f64) -> Option<usize> {
    let step = match grid.rows_run_north {
        true => grid.cell_size,
        false => -grid.cell_size,
    };
    stored_index(grid.first_row_y, step, grid.rows, y)
}

fn contiguous_runs(indices: &[Option<usize>]) -> Vec<Range<usize>> {
    let mut sorted: Vec<usize> = indices.iter().flatten().copied().collect();
    sorted.sort_unstable();
    sorted.dedup();
    let mut runs: Vec<Range<usize>> = Vec::new();
    for index in sorted {
        match runs.last_mut() {
            Some(run) if run.end == index => run.end += 1,
            _ => runs.push(index..index + 1),
        }
    }
    runs
}

impl StoredType {
    fn of(array: &ZarrArray) -> core::result::Result<StoredType, String> {
        Ok(match array.data_type().name_v3().as_deref() {
            Some("uint8") => StoredType::U8,
            Some("int8") => StoredType::I8,
            Some("uint16") => StoredType::U16,
            Some("int16") => StoredType::I16,
            Some("uint32") => StoredType::U32,
            Some("int32") => StoredType::I32,
            Some("uint64") => StoredType::U64,
            Some("int64") => StoredType::I64,
            Some("float32") => StoredType::F32,
            Some("float64") => StoredType::F64,
            _ => {
                return Err(format!(
                    "data type {} is not read, only 8 to 64 bit integers, float32 and float64",
                    array.data_type()
                ));
            }
        })
    }

    fn format(self) -> Option<SampleFormat> {
        match self {
            StoredType::U8 => Some(SampleFormat::U8),
            StoredType::I8 => Some(SampleFormat::I8),
            StoredType::U16 => Some(SampleFormat::U16),
            StoredType::I16 => Some(SampleFormat::I16),
            StoredType::U32 => Some(SampleFormat::U32),
            StoredType::I32 => Some(SampleFormat::I32),
            StoredType::F32 => Some(SampleFormat::F32),
            StoredType::F64 => Some(SampleFormat::F64),
            StoredType::U64 | StoredType::I64 => None,
        }
    }

    fn read(
        self,
        array: &ZarrArray,
        subset: &[Range<u64>],
    ) -> core::result::Result<Vec<f64>, ArrayError> {
        match self {
            StoredType::U8 => widened::<u8>(array, subset, f64::from),
            StoredType::I8 => widened::<i8>(array, subset, f64::from),
            StoredType::U16 => widened::<u16>(array, subset, f64::from),
            StoredType::I16 => widened::<i16>(array, subset, f64::from),
            StoredType::U32 => widened::<u32>(array, subset, f64::from),
            StoredType::I32 => widened::<i32>(array, subset, f64::from),
            StoredType::U64 => widened::<u64>(array, subset, |value| value as f64),
            StoredType::I64 => widened::<i64>(array, subset, |value| value as f64),
            StoredType::F32 => widened::<f32>(array, subset, f64::from),
            StoredType::F64 => widened::<f64>(array, subset, |value| value),
        }
    }
}

fn widened<T: ElementOwned>(
    array: &ZarrArray,
    subset: &[Range<u64>],
    widen: fn(T) -> f64,
) -> core::result::Result<Vec<f64>, ArrayError> {
    Ok(array
        .retrieve_array_subset::<Vec<T>>(&subset)?
        .into_iter()
        .map(widen)
        .collect())
}

impl Packing {
    fn of(array: &ZarrArray) -> core::result::Result<Packing, String> {
        let attributes = array.attributes();
        let mut missing_values = attribute_numbers(array, "missing_value")?;
        missing_values.extend(single_number(array, "_FillValue")?);
        // xarray masks a zarr v2 array's fill_value as its _FillValue and ignores a v3 one
        if let ArrayMetadata::V2(metadata) = array.metadata()
            && !attributes.contains_key("_FillValue")
        {
            missing_values.extend(metadata.fill_value.as_f64());
        }
        Ok(Packing {
            scale_factor: single_number(array, "scale_factor")?,
            add_offset: single_number(array, "add_offset")?,
            missing_values,
        })
    }

    fn unpack(&self, raw: f64) -> f64 {
        if raw.is_nan() || self.missing_values.contains(&raw) {
            return f64::NAN;
        }
        raw * self.scale_factor.unwrap_or(1.0) + self.add_offset.unwrap_or(0.0)
    }
}

fn attribute_numbers(array: &ZarrArray, name: &str) -> core::result::Result<Vec<f64>, String> {
    let values: Vec<&Value> = match array.attributes().get(name) {
        None => return Ok(Vec::new()),
        Some(Value::Array(values)) => values.iter().collect(),
        Some(value) => vec![value],
    };
    values
        .into_iter()
        .map(|value| {
            // zarr writes NaN and infinities as strings, which the fill value parser reads
            serde_json::from_value::<FillValueMetadata>(value.clone())
                .ok()
                .and_then(|number| number.as_f64())
                .ok_or_else(|| format!("attribute {name} holds {value}, not a number"))
        })
        .collect()
}

fn single_number(array: &ZarrArray, name: &str) -> core::result::Result<Option<f64>, String> {
    match attribute_numbers(array, name)?.as_slice() {
        [] => Ok(None),
        [value] => Ok(Some(*value)),
        values => Err(format!(
            "attribute {name} holds {} values, CF packing takes one",
            values.len()
        )),
    }
}

fn dimension_names(array: &ZarrArray) -> core::result::Result<Vec<Option<String>>, String> {
    let dimension_count = array.dimensionality();
    if let Some(names) = array.dimension_names() {
        return Ok(names.clone());
    }
    // xarray names a zarr v2 array's dimensions in this attribute
    let names = match array.attributes().get("_ARRAY_DIMENSIONS") {
        None => return Ok(vec![None; dimension_count]),
        Some(Value::Array(names)) => names
            .iter()
            .map(|name| match name {
                Value::String(name) => Ok(Some(name.clone())),
                other => Err(format!("_ARRAY_DIMENSIONS holds {other}, not a name")),
            })
            .collect::<core::result::Result<Vec<Option<String>>, String>>()?,
        Some(other) => return Err(format!("_ARRAY_DIMENSIONS is {other}, not a list")),
    };
    if names.len() != dimension_count {
        return Err(format!(
            "_ARRAY_DIMENSIONS names {} dimensions of {dimension_count}",
            names.len()
        ));
    }
    Ok(names)
}

fn coordinate_array(
    storage: &ReadableStorage,
    group: &str,
    name: &str,
) -> core::result::Result<Option<ZarrArray>, String> {
    match ZarrArray::open(storage.clone(), &format!("{group}/{name}")) {
        Ok(array) => Ok(Some(array)),
        Err(ArrayCreateError::MissingMetadata) => Ok(None),
        Err(e) => Err(format!("coordinate {name}: {e}")),
    }
}

fn coordinate_values(
    array: &ZarrArray,
    name: &str,
    length: usize,
) -> core::result::Result<Vec<f64>, String> {
    if array.shape() != [length as u64] {
        return Err(format!(
            "coordinate {name} has shape {:?}, its dimension has {length} cells",
            array.shape()
        ));
    }
    StoredType::of(array)
        .map_err(|e| format!("coordinate {name}: {e}"))?
        .read(array, &[0..length as u64])
        .map_err(|e| format!("coordinate {name}: {e}"))
}

fn array_axis(array: &ZarrArray, name: &str) -> core::result::Result<Axis, String> {
    Ok(coordinate_axis(
        name,
        text_attribute(array, name, "standard_name")?,
        text_attribute(array, name, "units")?,
    ))
}

fn text_attribute<'a>(
    array: &'a ZarrArray,
    name: &str,
    attribute: &str,
) -> core::result::Result<Option<&'a str>, String> {
    match array.attributes().get(attribute) {
        None => Ok(None),
        Some(Value::String(text)) => Ok(Some(text.as_str())),
        Some(_) => Err(format!("{name} attribute {attribute} is not text")),
    }
}

struct ZarrHttpStore {
    base_url: String,
}

impl ZarrHttpStore {
    fn object(&self, key: &StoreKey) -> HttpRange {
        HttpRange::new(format!("{}/{}", self.base_url, key.as_str()))
    }
}

fn storage_error(error: impl std::fmt::Display) -> StorageError {
    StorageError::Other(error.to_string())
}

impl ReadableStorageTraits for ZarrHttpStore {
    fn get(&self, key: &StoreKey) -> core::result::Result<MaybeBytes, StorageError> {
        let object = self.object(key).read_object().map_err(storage_error)?;
        Ok(object.map(|bytes| Bytes::copy_from_slice(&bytes)))
    }

    fn get_partial_many<'a>(
        &'a self,
        key: &StoreKey,
        byte_ranges: ByteRangeIterator<'a>,
    ) -> core::result::Result<MaybeBytesIterator<'a>, StorageError> {
        let byte_ranges: Vec<ByteRange> = byte_ranges.collect();
        // zarrs decodes a compressed chunk from this one open range
        if byte_ranges == [ByteRange::FromStart(0, None)] {
            let Some(bytes) = self.get(key)? else {
                return Ok(None);
            };
            return Ok(Some(Box::new(std::iter::once(Ok::<Bytes, StorageError>(
                bytes,
            )))));
        }
        let object = self.object(key);
        let bounded: Option<Vec<(u64, u64)>> = byte_ranges
            .iter()
            .map(|range| match range {
                ByteRange::FromStart(offset, Some(length)) => Some((*offset, *length)),
                ByteRange::FromStart(_, None) | ByteRange::Suffix(_) => None,
            })
            .collect();
        let spans = match bounded {
            Some(spans) => spans,
            None => {
                let Some(size) = object.length_if_found().map_err(storage_error)? else {
                    return Ok(None);
                };
                byte_ranges
                    .iter()
                    .map(|range| (range.start(size), range.length(size)))
                    .collect()
            }
        };
        // HttpRange asks for an inclusive end byte, which an empty span has none of
        let nonempty: Vec<(u64, u64)> = spans
            .iter()
            .copied()
            .filter(|(_, length)| *length > 0)
            .collect();
        let mut bodies = object
            .read_ranges(&nonempty)
            .map_err(storage_error)?
            .into_iter();
        let out: Vec<core::result::Result<Bytes, StorageError>> = spans
            .iter()
            .map(|(_, length)| match length {
                0 => Ok(Bytes::new()),
                _ => Ok(Bytes::from(
                    bodies.next().expect("one body per nonempty span"),
                )),
            })
            .collect();
        Ok(Some(Box::new(out.into_iter())))
    }

    fn size_key(&self, key: &StoreKey) -> core::result::Result<Option<u64>, StorageError> {
        self.object(key).length_if_found().map_err(storage_error)
    }

    fn supports_get_partial(&self) -> bool {
        true
    }
}

// zarrs is built without its zstd feature, which links the C zstd library
#[derive(Debug)]
struct RuzstdDecoder {
    configuration: Option<Configuration>,
}

zarrs::plugin::impl_extension_aliases!(RuzstdDecoder, v3: "zstd", v2: "zstd");

inventory::submit! {
    CodecPluginV3::new::<RuzstdDecoder>()
}

inventory::submit! {
    CodecPluginV2::new::<RuzstdDecoder>()
}

impl CodecTraitsV3 for RuzstdDecoder {
    fn create(metadata: &MetadataV3) -> core::result::Result<Codec, PluginCreateError> {
        Ok(Codec::BytesToBytes(Arc::new(RuzstdDecoder {
            configuration: metadata.configuration().cloned(),
        })))
    }
}

impl CodecTraitsV2 for RuzstdDecoder {
    fn create(metadata: &MetadataV2) -> core::result::Result<Codec, PluginCreateError> {
        Ok(Codec::BytesToBytes(Arc::new(RuzstdDecoder {
            configuration: Some(metadata.configuration().clone()),
        })))
    }
}

impl CodecTraits for RuzstdDecoder {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn configuration(
        &self,
        _version: ZarrVersion,
        _options: &CodecMetadataOptions,
    ) -> Option<Configuration> {
        self.configuration.clone()
    }

    fn partial_decoder_capability(&self) -> PartialDecoderCapability {
        PartialDecoderCapability {
            partial_read: false,
            partial_decode: false,
        }
    }

    fn partial_encoder_capability(&self) -> PartialEncoderCapability {
        PartialEncoderCapability {
            partial_encode: false,
        }
    }
}

impl BytesToBytesCodecTraits for RuzstdDecoder {
    fn into_dyn(self: Arc<Self>) -> Arc<dyn BytesToBytesCodecTraits> {
        self
    }

    fn recommended_concurrency(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> core::result::Result<RecommendedConcurrency, CodecError> {
        Ok(RecommendedConcurrency::new_maximum(1))
    }

    fn encoded_representation(
        &self,
        _decoded_representation: &BytesRepresentation,
    ) -> BytesRepresentation {
        BytesRepresentation::UnboundedSize
    }

    fn encode<'a>(
        &self,
        _decoded_value: ArrayBytesRaw<'a>,
        _options: &CodecOptions,
    ) -> core::result::Result<ArrayBytesRaw<'a>, CodecError> {
        Err(CodecError::Other(
            "zstd chunks are read only, geoplumb writes no zarr".into(),
        ))
    }

    fn decode<'a>(
        &self,
        encoded_value: ArrayBytesRaw<'a>,
        _decoded_representation: &BytesRepresentation,
        _options: &CodecOptions,
    ) -> core::result::Result<ArrayBytesRaw<'a>, CodecError> {
        let mut decoded = Vec::new();
        StreamingDecoder::new(encoded_value.as_ref())
            .map_err(|e| CodecError::Other(format!("zstd chunk: {e}")))?
            .read_to_end(&mut decoded)
            .map_err(|e| CodecError::Other(format!("zstd chunk: {e}")))?;
        Ok(Cow::Owned(decoded))
    }
}
