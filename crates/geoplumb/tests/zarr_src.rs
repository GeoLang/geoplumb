use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::path::Path;
use std::sync::Arc;

use geoplumb::caps::{Dtype, ValueKind};
use geoplumb::element::Source;
use geoplumb::elements::stac::parse_rfc3339;
use geoplumb::elements::{ZarrLayer, ZarrSrc, ZarrStore};
use geoplumb::{Bbox, Crs, RasterChunk, TimeInterval, WindowReq};
use ruzstd::encoding::{CompressionLevel, compress_to_vec};
use serde_json::{Map, Value, json};
use zarrs::array::{
    Array, ArrayBuilder, ArrayMetadata, ArrayMetadataV2, FillValueMetadata, data_type,
};
use zarrs::filesystem::FilesystemStore;
use zarrs::metadata::v2::DataTypeMetadataV2;

fn object(value: Value) -> Map<String, Value> {
    value.as_object().expect("a json object").clone()
}

fn filesystem_store(dir: &Path) -> Arc<FilesystemStore> {
    Arc::new(FilesystemStore::new(dir).unwrap())
}

fn v3_coordinate(store: &Arc<FilesystemStore>, name: &str, values: Vec<f64>, attributes: Value) {
    let length = values.len() as u64;
    let array = ArrayBuilder::new(vec![length], vec![length], data_type::float64(), f64::NAN)
        .dimension_names(Some([name]))
        .attributes(object(attributes))
        .build(store.clone(), &format!("/{name}"))
        .unwrap();
    array.store_metadata().unwrap();
    array
        .store_array_subset(&array.subset_all(), values)
        .unwrap();
}

fn layer(array: &str) -> ZarrLayer {
    ZarrLayer {
        array: array.into(),
        x_coordinate: None,
        y_coordinate: None,
        select: BTreeMap::new(),
        crs: None,
        datetime: None,
        value_kind: ValueKind::Continuous,
    }
}

async fn read(
    src: &ZarrSrc,
    bbox: Bbox,
    resolution: f64,
    time: Option<TimeInterval>,
) -> RasterChunk {
    let req = WindowReq {
        bbox,
        resolution,
        time,
    };
    src.read(&req).await.unwrap().into_raster().unwrap()
}

fn interval(start: &str, end: &str) -> TimeInterval {
    TimeInterval::new(parse_rfc3339(start).unwrap(), parse_rfc3339(end).unwrap())
}

fn assert_values(chunk: &RasterChunk, expected: &[f64]) {
    let actual = chunk.bands.bands()[0].to_raster();
    let actual = actual.data();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.is_nan() && e.is_nan()) || (a - e).abs() < 1e-9,
            "pixel {i}: read {a}, expected {e}"
        );
    }
}

const FILL: f32 = -999.0;

fn sst_value(step: u64, row: u64, column: u64) -> f32 {
    match (step, row, column) {
        (1, 3, 2) => FILL,
        _ => (step * 100 + row * 10 + column) as f32,
    }
}

fn write_v3_time_series(dir: &Path, fill_attribute: Value) {
    // sst is time, lat and lon of 3 by 4 by 6 in chunks of 1 by 2 by 4, lat stored south to north
    let store = filesystem_store(dir);
    v3_coordinate(
        &store,
        "lat",
        vec![-1.5, -0.5, 0.5, 1.5],
        json!({"units": "degrees_north"}),
    );
    v3_coordinate(
        &store,
        "lon",
        vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        json!({"units": "degrees_east"}),
    );
    let time = ArrayBuilder::new(vec![3], vec![3], data_type::int64(), 0i64)
        .dimension_names(Some(["time"]))
        .attributes(object(json!({
            "units": "hours since 2024-01-01 00:00:00",
            "calendar": "proleptic_gregorian"
        })))
        .build(store.clone(), "/time")
        .unwrap();
    time.store_metadata().unwrap();
    time.store_array_subset(&time.subset_all(), vec![0i64, 6, 12])
        .unwrap();

    let sst = ArrayBuilder::new(vec![3, 4, 6], vec![1, 2, 4], data_type::float32(), f32::NAN)
        .dimension_names(Some(["time", "lat", "lon"]))
        .attributes(object(json!({"_FillValue": fill_attribute})))
        .build(store.clone(), "/sst")
        .unwrap();
    sst.store_metadata().unwrap();
    let values: Vec<f32> = (0..3)
        .flat_map(|step| {
            (0..4).flat_map(move |row| (0..6).map(move |column| sst_value(step, row, column)))
        })
        .collect();
    sst.store_array_subset(&[0..3, 0..4, 0..6], values).unwrap();
}

#[tokio::test]
async fn v3_windows_and_time_steps_read_back_in_their_own_dtype() {
    let dir = tempfile::tempdir().unwrap();
    write_v3_time_series(dir.path(), json!(FILL));
    let mut sst = layer("sst");
    sst.datetime = Some(interval("2024-01-01T00:00:00Z", "2024-01-02T00:00:00Z"));
    let store = ZarrStore::Path(dir.path().to_path_buf());
    let src = ZarrSrc::open(&store, &sst).unwrap();
    assert!(src.time_varying());
    let start = parse_rfc3339("2024-01-01T00:00:00Z").unwrap();
    let hour = 3_600_000;
    assert_eq!(
        src.step_instants_ms(),
        &[start, start + 6 * hour, start + 12 * hour]
    );
    let grid = src.grid();
    assert_eq!(
        (grid.origin_x, grid.origin_y, grid.base_resolution),
        (0.0, 2.0, 1.0)
    );

    // north row first: lat 1.5 is stored row 3, lat 0.5 row 2
    let window = Bbox::new(1.0, 0.0, 4.0, 2.0);
    let last = read(&src, window, 1.0, None).await;
    assert_eq!(last.crs, Crs::WGS84);
    assert_eq!(Dtype::of(&last.bands.bands()[0]), Dtype::F32);
    assert_values(&last, &[231.0, 232.0, 233.0, 221.0, 222.0, 223.0]);

    let around_six = interval("2024-01-01T05:00:00Z", "2024-01-01T07:00:00Z");
    let middle = read(&src, window, 1.0, Some(around_six)).await;
    assert_values(&middle, &[131.0, f64::NAN, 133.0, 121.0, 122.0, 123.0]);

    let first = interval("2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z");
    let coarse = read(&src, Bbox::new(0.0, -2.0, 4.0, 2.0), 2.0, Some(first)).await;
    assert_values(&coarse, &[25.5, 27.5, 5.5, 7.5]);

    let february = interval("2024-02-01T00:00:00Z", "2024-02-02T00:00:00Z");
    let empty = read(&src, window, 1.0, Some(february)).await;
    assert_values(&empty, &[f64::NAN; 6]);

    // a chunk only the south east window touches, so the north west window never reads it
    std::fs::write(dir.path().join("sst/c/2/0/1"), b"bad").unwrap();
    assert_values(
        &read(&src, window, 1.0, None).await,
        &[231.0, 232.0, 233.0, 221.0, 222.0, 223.0],
    );
    let south_east = WindowReq {
        bbox: Bbox::new(4.0, -2.0, 6.0, 0.0),
        resolution: 1.0,
        time: None,
    };
    assert!(src.read(&south_east).await.is_err());
}

#[tokio::test]
async fn xarray_base64_fill_value_masks_float_cells() {
    let dir = tempfile::tempdir().unwrap();
    // xarray writes a v3 float _FillValue as base64 of a little endian f64, this is -999
    write_v3_time_series(dir.path(), json!("AAAAAAA4j8A="));
    let mut sst = layer("sst");
    sst.datetime = Some(interval("2024-01-01T05:00:00Z", "2024-01-01T07:00:00Z"));
    let src = ZarrSrc::open(&ZarrStore::Path(dir.path().to_path_buf()), &sst).unwrap();
    let middle = read(&src, Bbox::new(1.0, 0.0, 4.0, 2.0), 1.0, None).await;
    assert_values(&middle, &[131.0, f64::NAN, 133.0, 121.0, 122.0, 123.0]);
}

fn nonzero(values: &[u64]) -> Vec<NonZeroU64> {
    values
        .iter()
        .map(|&v| NonZeroU64::new(v).unwrap())
        .collect()
}

fn v2_array(
    store: &Arc<FilesystemStore>,
    path: &str,
    (shape, chunks): (Vec<u64>, Vec<u64>),
    dtype: &str,
    fill_value: FillValueMetadata,
    attributes: Value,
) -> Array<FilesystemStore> {
    let metadata = ArrayMetadataV2::new(
        shape,
        nonzero(&chunks),
        DataTypeMetadataV2::Simple(dtype.into()),
        fill_value,
        None,
        None,
    )
    .with_attributes(object(attributes));
    let array = Array::new_with_metadata(store.clone(), path, ArrayMetadata::V2(metadata)).unwrap();
    array.store_metadata().unwrap();
    array
}

const PACKED_FILL: i16 = -9999;

fn write_v2_levels(dir: &Path) {
    // t2m is level, y and x of 2 by 3 by 4 packed int16, rows stored north first
    let store = filesystem_store(dir);
    let x = v2_array(
        &store,
        "/x",
        (vec![4], vec![4]),
        "<f8",
        FillValueMetadata::Null,
        json!({"_ARRAY_DIMENSIONS": ["x"]}),
    );
    x.store_array_subset(&x.subset_all(), vec![500.0f64, 600.0, 700.0, 800.0])
        .unwrap();
    let y = v2_array(
        &store,
        "/y",
        (vec![3], vec![3]),
        "<f8",
        FillValueMetadata::Null,
        json!({"_ARRAY_DIMENSIONS": ["y"]}),
    );
    y.store_array_subset(&y.subset_all(), vec![300.0f64, 200.0, 100.0])
        .unwrap();
    let t2m = v2_array(
        &store,
        "/t2m",
        (vec![2, 3, 4], vec![1, 3, 2]),
        "<i2",
        FillValueMetadata::Number(serde_json::Number::from(PACKED_FILL)),
        json!({
            "_ARRAY_DIMENSIONS": ["level", "y", "x"],
            "scale_factor": 0.5,
            "add_offset": 10.0
        }),
    );
    let values: Vec<i16> = (0..2i16)
        .flat_map(|level| {
            (0..3i16).flat_map(move |row| {
                (0..4i16).map(move |column| match (level, row, column) {
                    (1, 0, 1) => PACKED_FILL,
                    _ => level * 1000 + row * 10 + column,
                })
            })
        })
        .collect();
    t2m.store_array_subset(&[0..2, 0..3, 0..4], values).unwrap();
}

#[tokio::test]
async fn v2_packed_values_unpack_and_mask_the_zarray_fill_value() {
    let dir = tempfile::tempdir().unwrap();
    write_v2_levels(dir.path());
    let store = ZarrStore::Path(dir.path().to_path_buf());

    let mut t2m = layer("t2m");
    t2m.select = BTreeMap::from([("level".to_string(), 1)]);
    let error = ZarrSrc::open(&store, &t2m)
        .err()
        .expect("x/y coordinates without a crs fail");
    assert!(error.to_string().contains("crs"), "{error}");

    t2m.crs = Some(Crs(32632));
    t2m.select = BTreeMap::new();
    let error = ZarrSrc::open(&store, &t2m)
        .err()
        .expect("a level dimension without a select index fails");
    assert!(error.to_string().contains("level"), "{error}");

    t2m.select = BTreeMap::from([("level".to_string(), 1)]);
    let src = ZarrSrc::open(&store, &t2m).unwrap();
    assert!(!src.time_varying());
    let grid = src.grid();
    assert_eq!(
        (grid.origin_x, grid.origin_y, grid.base_resolution),
        (450.0, 350.0, 100.0)
    );
    let chunk = read(&src, Bbox::new(550.0, 150.0, 750.0, 350.0), 100.0, None).await;
    assert_eq!(chunk.crs, Crs(32632));
    assert_eq!(Dtype::of(&chunk.bands.bands()[0]), Dtype::F64);
    let unpacked = |raw: f64| raw * 0.5 + 10.0;
    assert_values(
        &chunk,
        &[
            f64::NAN,
            unpacked(1002.0),
            unpacked(1011.0),
            unpacked(1012.0),
        ],
    );
}

fn write_v3_global(dir: &Path, longitudes: Vec<f64>) {
    // stored rows run south to north
    let store = filesystem_store(dir);
    let columns = longitudes.len() as u64;
    v3_coordinate(
        &store,
        "lat",
        vec![-45.0, 45.0],
        json!({"units": "degrees_north"}),
    );
    v3_coordinate(&store, "lon", longitudes, json!({"units": "degrees_east"}));
    let sst = ArrayBuilder::new(
        vec![2, columns],
        vec![2, columns],
        data_type::float64(),
        f64::NAN,
    )
    .dimension_names(Some(["lat", "lon"]))
    .build(store.clone(), "/sst")
    .unwrap();
    sst.store_metadata().unwrap();
    let values: Vec<f64> = (0..2)
        .flat_map(|row| (0..columns).map(move |column| (10 * row + column) as f64))
        .collect();
    sst.store_array_subset(&[0..2, 0..columns], values).unwrap();
}

#[tokio::test]
async fn longitudes_past_180_serve_west_of_the_antimeridian() {
    let dir = tempfile::tempdir().unwrap();
    write_v3_global(dir.path(), vec![45.0, 135.0, 225.0, 315.0]);
    let store = ZarrStore::Path(dir.path().to_path_buf());
    let src = ZarrSrc::open(&store, &layer("/sst")).unwrap();
    let chunk = read(&src, Bbox::new(-180.0, -90.0, 180.0, 90.0), 90.0, None).await;
    assert_eq!(Dtype::of(&chunk.bands.bands()[0]), Dtype::F64);
    // north row first, columns from -135 east: stored 225, 315, 45, 135
    assert_values(&chunk, &[12.0, 13.0, 10.0, 11.0, 2.0, 3.0, 0.0, 1.0]);
}

#[test]
fn irregular_coordinates_fail_naming_the_coordinate() {
    let dir = tempfile::tempdir().unwrap();
    write_v3_global(dir.path(), vec![0.0, 1.0, 3.0]);
    let store = ZarrStore::Path(dir.path().to_path_buf());
    let error = ZarrSrc::open(&store, &layer("sst"))
        .err()
        .expect("irregular longitudes fail");
    let message = error.to_string();
    assert!(message.contains("/sst"), "{message}");
    assert!(
        message.contains("coordinate lon is not regularly spaced"),
        "{message}"
    );
}

#[tokio::test]
async fn zstd_chunks_decode_through_ruzstd() {
    let dir = tempfile::tempdir().unwrap();
    write_v3_global(dir.path(), vec![45.0, 135.0, 225.0, 315.0]);
    // zarr-python 3 compresses v3 chunks with zstd unless told otherwise
    let metadata_path = dir.path().join("sst/zarr.json");
    let mut metadata: Value =
        serde_json::from_slice(&std::fs::read(&metadata_path).unwrap()).unwrap();
    metadata["codecs"]
        .as_array_mut()
        .expect("a codec list")
        .push(json!({"name": "zstd", "configuration": {"level": 0, "checksum": false}}));
    std::fs::write(&metadata_path, serde_json::to_vec(&metadata).unwrap()).unwrap();
    let chunk_path = dir.path().join("sst/c/0/0");
    let raw = std::fs::read(&chunk_path).unwrap();
    let compressed = compress_to_vec(raw.as_slice(), CompressionLevel::Fastest);
    assert_ne!(compressed, raw);
    std::fs::write(&chunk_path, compressed).unwrap();

    let store = ZarrStore::Path(dir.path().to_path_buf());
    let src = ZarrSrc::open(&store, &layer("sst")).unwrap();
    let chunk = read(&src, Bbox::new(-180.0, -90.0, 180.0, 90.0), 90.0, None).await;
    assert_values(&chunk, &[12.0, 13.0, 10.0, 11.0, 2.0, 3.0, 0.0, 1.0]);
}

#[test]
fn blosc_chunks_fail_naming_the_codec() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir(dir.path().join("sst")).unwrap();
    let zarray = json!({
        "zarr_format": 2,
        "shape": [2, 2],
        "chunks": [2, 2],
        "dtype": "<f4",
        "compressor": {"id": "blosc", "cname": "lz4", "clevel": 5, "shuffle": 1, "blocksize": 0},
        "fill_value": null,
        "order": "C",
        "filters": null
    });
    std::fs::write(
        dir.path().join("sst/.zarray"),
        serde_json::to_vec(&zarray).unwrap(),
    )
    .unwrap();
    let store = ZarrStore::Path(dir.path().to_path_buf());
    let error = ZarrSrc::open(&store, &layer("sst"))
        .err()
        .expect("a blosc array fails");
    assert!(error.to_string().contains("blosc"), "{error}");
}
