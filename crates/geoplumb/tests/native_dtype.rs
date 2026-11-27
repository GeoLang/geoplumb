use std::io::Cursor;
use std::sync::Arc;

use futures::future::BoxFuture;
use geoplumb::caps::{
    BandDtypes, BandKinds, CapsPattern, CapsSet, Constraint, Dtype, RasterPattern, ResRange,
    SetField, ValueKind,
};
use geoplumb::element::Source;
use geoplumb::elements::algebra::AlgebraOp;
use geoplumb::elements::{CogSrc, MapAlgebra, Mosaic, QualityMask, Viewshed};
use geoplumb::export::{ExportGrid, write_cog};
use geoplumb::resample::resample_to_grid;
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, NodeId, RasterBands, RasterChunk, WindowReq};
use terrano_core::{
    CogParams, CogReader, RangeRead, SampleFormat, SampleGrid, TypedBand, typed_tile_bytes,
    write_cog_tiles,
};

const SIDE: usize = 512;
const CELL: f64 = 1.0;
const ORIGIN_X: f64 = 0.0;
const ORIGIN_Y: f64 = 512.0;
const CRS: Crs = Crs(32632);
const I16_NODATA: i16 = i16::MIN;
const QUALITY_NODATA: u8 = 255;

fn grid<T>(value: impl Fn(usize, usize) -> T) -> SampleGrid<T> {
    let values = (0..SIDE * SIDE)
        .map(|index| value(index % SIDE, index / SIDE))
        .collect();
    SampleGrid::new(SIDE, SIDE, CELL, values).unwrap()
}

fn code(col: usize, row: usize) -> u8 {
    ((col * 7 + row * 3) % 11) as u8
}

fn elevation(col: usize, row: usize) -> i16 {
    if (col + row).is_multiple_of(13) {
        return I16_NODATA;
    }
    (col as i32 * 37 - row as i32 * 53) as i16
}

fn reflectance(col: usize, row: usize) -> f32 {
    if (col * row).is_multiple_of(17) {
        return f32::NAN;
    }
    col as f32 * 0.25 - row as f32 * 0.5
}

fn quality(col: usize, row: usize) -> u8 {
    ((col + 3 * row) % 11) as u8
}

fn u8_band(value: impl Fn(usize, usize) -> u8, nodata: Option<u8>) -> TypedBand {
    TypedBand::U8 {
        samples: grid(value),
        nodata,
    }
}

fn i16_band() -> TypedBand {
    TypedBand::I16 {
        samples: grid(elevation),
        nodata: Some(I16_NODATA),
    }
}

fn f32_band(value: impl Fn(usize, usize) -> f32) -> TypedBand {
    TypedBand::F32(grid(value))
}

fn full_bbox() -> Bbox {
    Bbox::new(
        ORIGIN_X,
        ORIGIN_Y - SIDE as f64 * CELL,
        ORIGIN_X + SIDE as f64 * CELL,
        ORIGIN_Y,
    )
}

fn chunk_of(bands: Vec<TypedBand>) -> RasterChunk {
    RasterChunk {
        bands: RasterBands::new(bands).unwrap(),
        bbox: full_bbox(),
        resolution: CELL,
        crs: CRS,
    }
}

fn window(col0: usize, row0: usize, col1: usize, row1: usize) -> Bbox {
    Bbox::new(
        ORIGIN_X + col0 as f64 * CELL,
        ORIGIN_Y - row1 as f64 * CELL,
        ORIGIN_X + col1 as f64 * CELL,
        ORIGIN_Y - row0 as f64 * CELL,
    )
}

fn u8s(band: &TypedBand) -> (&[u8], Option<u8>) {
    match band {
        TypedBand::U8 { samples, nodata } => (samples.values(), *nodata),
        other => panic!("expected a u8 band, got {}", other.format().name()),
    }
}

fn i16s(band: &TypedBand) -> (&[i16], Option<i16>) {
    match band {
        TypedBand::I16 { samples, nodata } => (samples.values(), *nodata),
        other => panic!("expected an i16 band, got {}", other.format().name()),
    }
}

fn u16s(band: &TypedBand) -> (&[u16], Option<u16>) {
    match band {
        TypedBand::U16 { samples, nodata } => (samples.values(), *nodata),
        other => panic!("expected a u16 band, got {}", other.format().name()),
    }
}

fn f32s(band: &TypedBand) -> &[f32] {
    match band {
        TypedBand::F32(samples) => samples.values(),
        other => panic!("expected an f32 band, got {}", other.format().name()),
    }
}

fn in_window<T>(bbox: &Bbox, value: impl Fn(usize, usize) -> T) -> Vec<T> {
    let col0 = ((bbox.min_x - ORIGIN_X) / CELL) as usize;
    let row0 = ((ORIGIN_Y - bbox.max_y) / CELL) as usize;
    let cols = (bbox.width() / CELL) as usize;
    let rows = (bbox.height() / CELL) as usize;
    (0..cols * rows)
        .map(|index| value(col0 + index % cols, row0 + index / cols))
        .collect()
}

fn same_floats(got: &[f32], want: &[f32]) -> bool {
    got.len() == want.len()
        && got
            .iter()
            .zip(want)
            .all(|(g, w)| g == w || (g.is_nan() && w.is_nan()))
}

struct TypedSrc {
    name: &'static str,
    full: RasterChunk,
    value_kinds: BandKinds,
}

impl TypedSrc {
    fn new(name: &'static str, bands: Vec<TypedBand>, value_kind: ValueKind) -> TypedSrc {
        TypedSrc {
            name,
            full: chunk_of(bands),
            value_kinds: BandKinds::from(value_kind),
        }
    }
}

impl Source for TypedSrc {
    fn identity(&self) -> String {
        format!("TypedSrc {}", self.name)
    }

    fn constraint(&self) -> Constraint {
        let dtypes = self.full.bands.bands().iter().map(Dtype::of).collect();
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtypes: SetField::one(BandDtypes::from_bands(dtypes)),
            bands: SetField::one(self.full.bands.band_count() as u16),
            crs: SetField::one(CRS),
            resolution: ResRange::at_least(CELL),
            chunk_px: SetField::Any,
            value_kinds: SetField::one(self.value_kinds.clone()),
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: ORIGIN_X,
            origin_y: ORIGIN_Y,
            base_resolution: CELL,
            chunk_px: 256,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        Box::pin(async move { Ok(Chunk::Raster(self.full.crop_to(&req.bbox))) })
    }
}

async fn pull(engine: &Engine, node: NodeId, bbox: Bbox) -> RasterChunk {
    let req = WindowReq {
        bbox,
        resolution: CELL,
        time: None,
    };
    engine.pull(node, req).await.unwrap().into_raster().unwrap()
}

fn engine_over(source: TypedSrc) -> (Engine, NodeId) {
    let mut graph = Graph::new();
    let node = graph.add_source(Box::new(source));
    (Engine::new(graph, 256 << 20).unwrap(), node)
}

#[test]
fn a_chunk_counts_each_band_at_its_own_sample_width() {
    let side = 256;
    let byte_band = TypedBand::U8 {
        samples: SampleGrid::new(side, side, CELL, vec![1u8; side * side]).unwrap(),
        nodata: Some(0),
    };
    let chunk = RasterChunk {
        bands: RasterBands::new(vec![byte_band.clone()]).unwrap(),
        bbox: Bbox::new(0.0, 0.0, side as f64, side as f64),
        resolution: CELL,
        crs: CRS,
    };
    assert_eq!(chunk.byte_size(), 65_536);

    let float_band =
        TypedBand::F32(SampleGrid::new(side, side, CELL, vec![0.5; side * side]).unwrap());
    let mixed = RasterChunk {
        bands: RasterBands::new(vec![byte_band, float_band]).unwrap(),
        ..chunk
    };
    assert_eq!(mixed.byte_size(), 65_536 + 4 * 65_536);
}

#[test]
fn a_crop_keeps_each_band_in_its_own_format() {
    let chunk = chunk_of(vec![
        u8_band(code, Some(0)),
        i16_band(),
        f32_band(reflectance),
    ]);
    let bbox = window(100, 37, 300, 261);
    let cropped = chunk.crop_to(&bbox);
    assert_eq!((cropped.width(), cropped.height()), (200, 224));
    assert_eq!(cropped.bbox, bbox);

    let (codes, nodata) = u8s(cropped.bands.band(0).unwrap());
    assert_eq!(codes, in_window(&bbox, code).as_slice());
    assert_eq!(nodata, Some(0));
    let (heights, nodata) = i16s(cropped.bands.band(1).unwrap());
    assert_eq!(heights, in_window(&bbox, elevation).as_slice());
    assert_eq!(nodata, Some(I16_NODATA));
    let reflectances = f32s(cropped.bands.band(2).unwrap());
    assert!(same_floats(reflectances, &in_window(&bbox, reflectance)));
}

#[tokio::test]
async fn a_pull_across_chunk_seams_assembles_native_samples() {
    let (engine, node) = engine_over(TypedSrc::new(
        "assemble",
        vec![u8_band(code, Some(0)), i16_band(), f32_band(reflectance)],
        ValueKind::Categorical,
    ));
    let bbox = window(200, 180, 330, 300);
    let pulled = pull(&engine, node, bbox).await;

    assert_eq!((pulled.width(), pulled.height()), (130, 120));
    let (codes, _) = u8s(pulled.bands.band(0).unwrap());
    assert_eq!(codes, in_window(&bbox, code).as_slice());
    let (heights, nodata) = i16s(pulled.bands.band(1).unwrap());
    assert_eq!(heights, in_window(&bbox, elevation).as_slice());
    assert_eq!(nodata, Some(I16_NODATA));
    assert!(same_floats(
        f32s(pulled.bands.band(2).unwrap()),
        &in_window(&bbox, reflectance)
    ));
    assert_eq!(pulled.byte_size(), 130 * 120 * (1 + 2 + 4));
}

#[test]
fn categorical_resampling_picks_codes_and_continuous_rounds_into_the_band_format() {
    let chunk = chunk_of(vec![i16_band()]);
    let wide = RasterChunk {
        bands: RasterBands::from_rasters(vec![
            chunk.bands.band(0).unwrap().to_raster().into_owned(),
        ])
        .unwrap(),
        ..chunk.clone()
    };
    let off_grid = Bbox::new(10.0 / 3.0, 100.0, 10.0 / 3.0 + 64.0, 164.0);

    let continuous = resample_to_grid(
        &chunk,
        &off_grid,
        64,
        64,
        &BandKinds::from(ValueKind::Continuous),
    );
    let reference = resample_to_grid(
        &wide,
        &off_grid,
        64,
        64,
        &BandKinds::from(ValueKind::Continuous),
    );
    let (heights, nodata) = i16s(continuous.bands.band(0).unwrap());
    assert_eq!(nodata, Some(I16_NODATA));
    let reference = reference.bands.band(0).unwrap().to_raster();
    let mut fractional = 0;
    for (got, want) in heights.iter().zip(reference.data()) {
        if want.is_nan() {
            assert_eq!(*got, I16_NODATA);
            continue;
        }
        fractional += usize::from(want.fract() != 0.0);
        assert_eq!(
            f64::from(*got),
            want.round(),
            "bilinear {want} narrowed to {got}"
        );
    }
    assert!(fractional > 0, "no bilinear value needed rounding");

    let codes = chunk_of(vec![u8_band(code, Some(0))]);
    let categorical = resample_to_grid(
        &codes,
        &off_grid,
        64,
        64,
        &BandKinds::from(ValueKind::Categorical),
    );
    let (picked, nodata) = u8s(categorical.bands.band(0).unwrap());
    assert_eq!(nodata, Some(0));
    for (index, got) in picked.iter().enumerate() {
        let x = off_grid.min_x + (index % 64) as f64 + 0.5;
        let y = off_grid.max_y - (index / 64) as f64 - 0.5;
        let (col, row) = (
            (x - ORIGIN_X).floor() as usize,
            (ORIGIN_Y - y).floor() as usize,
        );
        assert_eq!(*got, code(col, row), "pixel {index}");
    }
}

#[tokio::test]
async fn a_mosaic_fills_the_first_input_gaps_in_the_input_format() {
    let holed = |col: usize, row: usize| if col < 256 { 0 } else { code(col, row) };
    let filler = |col: usize, row: usize| 1 + ((col + row) % 9) as u8;
    let mut graph = Graph::new();
    let first = graph.add_source(Box::new(TypedSrc::new(
        "mosaic first",
        vec![u8_band(holed, Some(0))],
        ValueKind::Categorical,
    )));
    let second = graph.add_source(Box::new(TypedSrc::new(
        "mosaic second",
        vec![u8_band(filler, Some(0))],
        ValueKind::Categorical,
    )));
    let mosaic = graph.add_fanin(&[first, second], Box::new(Mosaic::default()));
    let engine = Engine::new(graph, 256 << 20).unwrap();
    let bbox = window(200, 0, 320, 64);
    let pulled = pull(&engine, mosaic, bbox).await;

    let (codes, nodata) = u8s(pulled.bands.band(0).unwrap());
    assert_eq!(nodata, Some(0));
    let want = in_window(&bbox, |col, row| match holed(col, row) {
        0 => filler(col, row),
        kept => kept,
    });
    assert_eq!(codes, want.as_slice());

    let mut graph = Graph::new();
    let first = graph.add_source(Box::new(TypedSrc::new(
        "mosaic first float",
        vec![f32_band(reflectance)],
        ValueKind::Continuous,
    )));
    let second = graph.add_source(Box::new(TypedSrc::new(
        "mosaic second float",
        vec![f32_band(|_, _| -1.0)],
        ValueKind::Continuous,
    )));
    let mosaic = graph.add_fanin(&[first, second], Box::new(Mosaic::default()));
    let engine = Engine::new(graph, 256 << 20).unwrap();
    let pulled = pull(&engine, mosaic, bbox).await;
    let want = in_window(&bbox, |col, row| match reflectance(col, row) {
        hole if hole.is_nan() => -1.0,
        value => value,
    });
    assert!(same_floats(f32s(pulled.bands.band(0).unwrap()), &want));
}

#[tokio::test]
async fn a_quality_mask_writes_each_band_sentinel_or_nan() {
    let keep = [4.0, 5.0];
    let (engine, node) = engine_over(TypedSrc::new(
        "quality",
        vec![
            u8_band(code, Some(0)),
            f32_band(reflectance),
            u8_band(quality, Some(QUALITY_NODATA)),
        ],
        ValueKind::Categorical,
    ));
    let mut graph = Graph::new();
    let source = graph.add_source(Box::new(TypedSrc::new(
        "quality",
        vec![
            u8_band(code, Some(0)),
            f32_band(reflectance),
            u8_band(quality, Some(QUALITY_NODATA)),
        ],
        ValueKind::Categorical,
    )));
    let masked = graph.add_transform(source, Box::new(QualityMask::new(2, keep.to_vec())));
    let masked_engine = Engine::new(graph, 256 << 20).unwrap();
    let bbox = window(0, 0, 96, 80);
    let raw = pull(&engine, node, bbox).await;
    let got = pull(&masked_engine, masked, bbox).await;

    let kept = in_window(&bbox, |col, row| {
        keep.contains(&f64::from(quality(col, row)))
    });
    assert!(kept.contains(&true) && kept.contains(&false));
    let (codes, nodata) = u8s(got.bands.band(0).unwrap());
    assert_eq!(nodata, Some(0));
    let (raw_codes, _) = u8s(raw.bands.band(0).unwrap());
    let floats = f32s(got.bands.band(1).unwrap());
    let raw_floats = f32s(raw.bands.band(1).unwrap());
    let (qualities, quality_nodata) = u8s(got.bands.band(2).unwrap());
    assert_eq!(quality_nodata, Some(QUALITY_NODATA));
    for (index, keep) in kept.iter().enumerate() {
        if *keep {
            assert_eq!(codes[index], raw_codes[index], "kept code at {index}");
            assert!(same_floats(
                &floats[index..=index],
                &raw_floats[index..=index]
            ));
        } else {
            assert_eq!(codes[index], 0, "masked code at {index}");
            assert!(floats[index].is_nan(), "masked float at {index}");
            assert_eq!(qualities[index], QUALITY_NODATA);
        }
    }

    let mut graph = Graph::new();
    let source = graph.add_source(Box::new(TypedSrc::new(
        "quality without nodata",
        vec![u8_band(code, None), u8_band(quality, Some(QUALITY_NODATA))],
        ValueKind::Categorical,
    )));
    graph.add_transform(source, Box::new(QualityMask::new(1, keep.to_vec())));
    let refused = Engine::new(graph, 256 << 20)
        .err()
        .expect("a band with no nodata");
    assert!(refused.to_string().contains("nodata"), "{refused}");
}

fn widened(chunk: &RasterChunk) -> RasterChunk {
    let rasters = chunk
        .bands
        .bands()
        .iter()
        .map(|band| band.to_raster().into_owned())
        .collect();
    RasterChunk {
        bands: RasterBands::from_rasters(rasters).unwrap(),
        ..chunk.clone()
    }
}

#[test]
fn a_png_of_a_byte_band_renders_its_sentinel_transparent_as_nan_renders() {
    let chunk = chunk_of(vec![u8_band(code, Some(0))]).crop_to(&window(0, 0, 64, 64));
    let png = geoplumb::encode::png_gray(&chunk, 0.0, 10.0).unwrap();
    let reference = geoplumb::encode::png_gray(&widened(&chunk), 0.0, 10.0).unwrap();
    assert_eq!(png, reference);

    let unmarked = chunk_of(vec![u8_band(code, None)]).crop_to(&window(0, 0, 64, 64));
    let opaque = geoplumb::encode::png_gray(&unmarked, 0.0, 10.0).unwrap();
    assert_ne!(opaque, png, "code 0 with no nodata rendered transparent");
}

fn assert_same_samples(got: &TypedBand, want: &TypedBand) {
    match want {
        TypedBand::U8 { .. } => assert_eq!(u8s(got), u8s(want)),
        TypedBand::I16 { .. } => assert_eq!(i16s(got), i16s(want)),
        TypedBand::F32(_) => assert!(same_floats(f32s(got), f32s(want))),
        other => panic!("unexpected {}", other.format().name()),
    }
}

#[test]
fn a_geotiff_tile_keeps_the_band_format_and_writes_sentinels() {
    let bbox = window(0, 0, 64, 48);
    for chunk in [
        chunk_of(vec![u8_band(code, Some(0))]),
        chunk_of(vec![i16_band()]),
        chunk_of(vec![f32_band(reflectance)]),
    ] {
        let chunk = chunk.crop_to(&bbox);
        let band = chunk.bands.band(0).unwrap();
        let tif = geoplumb::encode::geotiff(&chunk, None).unwrap();
        let reader = CogReader::open(tif.as_slice()).unwrap();
        let base = &reader.levels()[0];
        assert_eq!(base.sample_format(), band.format());
        assert_eq!(base.nodata(), band.integer_nodata());
        assert_eq!(base.palette(), None);
        let read = reader.read_window_typed(0, 0, 0, 64, 48).unwrap();
        assert_same_samples(&read[0], band);
    }
}

async fn exported(source: TypedSrc) -> Vec<u8> {
    let (engine, node) = engine_over(source);
    let grid = ExportGrid {
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_size: CELL,
        cols: SIDE,
        rows: SIDE,
    };
    write_cog(Arc::new(engine), node, grid, None, Cursor::new(Vec::new()))
        .await
        .unwrap()
        .into_inner()
}

#[tokio::test(flavor = "multi_thread")]
async fn an_export_writes_the_link_dtype_and_its_nodata() {
    let bytes = exported(TypedSrc::new(
        "export u8",
        vec![u8_band(code, Some(0))],
        ValueKind::Categorical,
    ))
    .await;
    let reader = CogReader::open(bytes.as_slice()).unwrap();
    let base = &reader.levels()[0];
    assert_eq!(base.sample_format(), SampleFormat::U8);
    assert_eq!(base.nodata(), Some(0.0));
    let read = reader.read_window_typed(0, 0, 0, SIDE, SIDE).unwrap();
    let (codes, nodata) = u8s(&read[0]);
    assert_eq!(nodata, Some(0));
    assert_eq!(codes, grid(code).values());

    let bytes = exported(TypedSrc::new(
        "export i16",
        vec![i16_band()],
        ValueKind::Continuous,
    ))
    .await;
    let reader = CogReader::open(bytes.as_slice()).unwrap();
    assert_eq!(reader.levels()[0].sample_format(), SampleFormat::I16);
    assert_eq!(reader.levels()[0].nodata(), Some(f64::from(I16_NODATA)));
    let read = reader.read_window_typed(0, 0, 0, SIDE, SIDE).unwrap();
    assert_eq!(i16s(&read[0]).0, grid(elevation).values());

    let bytes = exported(TypedSrc::new(
        "export f32",
        vec![f32_band(reflectance)],
        ValueKind::Continuous,
    ))
    .await;
    let reader = CogReader::open(bytes.as_slice()).unwrap();
    assert_eq!(reader.levels()[0].sample_format(), SampleFormat::F32);
    assert_eq!(reader.levels()[0].nodata(), None);
    let read = reader.read_window_typed(0, 0, 0, SIDE, SIDE).unwrap();
    assert!(same_floats(f32s(&read[0]), grid(reflectance).values()));
}

#[tokio::test]
async fn a_fan_in_widens_a_narrower_input_through_a_spliced_cast() {
    let holed = |col: usize, row: usize| {
        if col < 16 {
            0
        } else {
            1000 + (col + row) as u16
        }
    };
    let wide = || TypedBand::U16 {
        samples: grid(holed),
        nodata: Some(0),
    };
    let mut graph = Graph::new();
    let first = graph.add_source(Box::new(TypedSrc::new(
        "cast u16",
        vec![wide()],
        ValueKind::Categorical,
    )));
    let second = graph.add_source(Box::new(TypedSrc::new(
        "cast u8",
        vec![u8_band(code, Some(0))],
        ValueKind::Categorical,
    )));
    graph.add_fanin(&[second, first], Box::new(Mosaic::default()));
    let refused = Engine::new(graph, 256 << 20)
        .err()
        .expect("u16 to u8 loses values");
    assert!(
        refused.to_string().contains("without losing values"),
        "{refused}"
    );

    let mut graph = Graph::new();
    let first = graph.add_source(Box::new(TypedSrc::new(
        "cast u16",
        vec![wide()],
        ValueKind::Categorical,
    )));
    let second = graph.add_source(Box::new(TypedSrc::new(
        "cast u8",
        vec![u8_band(code, Some(0))],
        ValueKind::Categorical,
    )));
    let mosaic = graph.add_fanin(&[first, second], Box::new(Mosaic::default()));
    let engine = Engine::new(graph, 256 << 20).unwrap();
    assert_eq!(engine.node_count(), 4, "a cast was spliced in");
    assert_eq!(
        engine.caps(second).raster().dtypes.band(0),
        Dtype::U8 { nodata: Some(0) }
    );
    assert_eq!(
        engine.caps(mosaic).raster().dtypes.band(0),
        Dtype::U16 { nodata: Some(0) }
    );
    let bbox = window(0, 0, 32, 32);
    let mosaicked = pull(&engine, mosaic, bbox).await;
    let (values, nodata) = u16s(mosaicked.bands.band(0).unwrap());
    assert_eq!(nodata, Some(0));
    let want = in_window(&bbox, |col, row| match holed(col, row) {
        0 => u16::from(code(col, row)),
        kept => kept,
    });
    assert_eq!(values, want.as_slice());
}

struct CogBytes(Vec<u8>);

impl RangeRead for CogBytes {
    fn read_range(&self, offset: u64, len: u64) -> Result<Vec<u8>, terrano_core::Error> {
        self.0.as_slice().read_range(offset, len)
    }
}

fn cog_of(band: &TypedBand, palette: Option<Vec<[u16; 3]>>) -> CogBytes {
    let params = CogParams {
        tile_width: 256,
        tile_height: 256,
        overview_levels: 0,
        epsg: CRS.0 as u16,
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        deflate: true,
        nodata: band.integer_nodata(),
        format: band.format(),
        palette,
    };
    let mut file = Cursor::new(Vec::new());
    write_cog_tiles(
        band.width(),
        band.height(),
        1,
        &params,
        |_, tile_col, tile_row| {
            typed_tile_bytes(std::slice::from_ref(band), tile_col, tile_row, &params)
        },
        &mut file,
    )
    .unwrap();
    CogBytes(file.into_inner())
}

fn engine_over_cog(source: CogSrc<CogBytes>) -> (Engine, NodeId) {
    let mut graph = Graph::new();
    let node = graph.add_source(Box::new(source));
    (Engine::new(graph, 256 << 20).unwrap(), node)
}

async fn pull_at(engine: &Engine, node: NodeId, bbox: Bbox, resolution: f64) -> RasterChunk {
    let req = WindowReq {
        bbox,
        resolution,
        time: None,
    };
    engine.pull(node, req).await.unwrap().into_raster().unwrap()
}

#[tokio::test]
async fn a_cog_source_serves_each_file_in_its_own_dtype() {
    for (band, value_kind) in [
        (u8_band(code, Some(0)), ValueKind::Categorical),
        (i16_band(), ValueKind::Continuous),
        (f32_band(reflectance), ValueKind::Continuous),
    ] {
        let name = band.format().name();
        let open = |file| CogSrc::open(file, value_kind, None).unwrap();
        let (engine, node) = engine_over_cog(open(cog_of(&band, None)));
        assert_eq!(
            engine.caps(node).raster().dtypes.band(0),
            Dtype::of(&band),
            "{name}"
        );

        let bbox = window(100, 37, 300, 261);
        let pulled = pull_at(&engine, node, bbox, CELL).await;
        let expected = chunk_of(vec![band.clone()]).crop_to(&bbox);
        assert_same_samples(
            pulled.bands.band(0).unwrap(),
            expected.bands.band(0).unwrap(),
        );

        // the window starts 16 cells west of and above the image
        let req = WindowReq {
            bbox: Bbox::new(
                ORIGIN_X - 16.0 * CELL,
                ORIGIN_Y - 32.0 * CELL,
                ORIGIN_X + 16.0 * CELL,
                ORIGIN_Y + 16.0 * CELL,
            ),
            resolution: CELL,
            time: None,
        };
        let edge = open(cog_of(&band, None))
            .read(&req)
            .await
            .unwrap()
            .into_raster()
            .unwrap();
        let edge = edge.bands.band(0).unwrap();
        assert_eq!(Dtype::of(edge), Dtype::of(&band), "{name}");
        let (got, want) = (edge.to_raster(), band.to_raster());
        for (index, value) in got.data().iter().enumerate() {
            let (col, row) = (index % 32, index / 32);
            if col < 16 || row < 16 {
                assert!(
                    value.is_nan(),
                    "{name} cell {index} outside the image holds {value}"
                );
                continue;
            }
            let inside = want.data()[(row - 16) * SIDE + col - 16];
            assert!(
                *value == inside || (value.is_nan() && inside.is_nan()),
                "{name} cell {index}: {value} vs {inside}"
            );
        }

        // the reference file holds the same samples widened to f64
        let wide = TypedBand::F64(band.to_raster().into_owned());
        let (reference, reference_node) = engine_over_cog(open(cog_of(&wide, None)));
        let bbox = window(0, 0, 256, 256);
        let native = pull_at(&engine, node, bbox, 2.0 * CELL).await;
        let widened = pull_at(&reference, reference_node, bbox, 2.0 * CELL).await;
        let native = native.bands.band(0).unwrap();
        assert_eq!(Dtype::of(native), Dtype::of(&band), "{name} decimated");
        let narrowed = |value: f64| match band.format() {
            SampleFormat::I16 => value.round(),
            SampleFormat::F32 => f64::from(value as f32),
            _ => value,
        };
        let (native, widened) = (
            native.to_raster(),
            widened.bands.band(0).unwrap().to_raster(),
        );
        let mut averaged = 0;
        for (got, reference) in native.data().iter().zip(widened.data()) {
            averaged += usize::from(reference.fract() != 0.0);
            assert!(
                *got == narrowed(*reference) || (got.is_nan() && reference.is_nan()),
                "{name} decimated {got} vs {reference}"
            );
        }
        if value_kind == ValueKind::Continuous {
            assert!(averaged > 0, "{name} decimation averaged nothing");
        }
    }
}

#[tokio::test]
async fn a_layer_nodata_marks_absent_cells_of_a_file_that_declares_none() {
    let band = u8_band(code, None);
    let plain = CogSrc::open(cog_of(&band, None), ValueKind::Categorical, None).unwrap();
    let (engine, node) = engine_over_cog(plain);
    assert_eq!(
        engine.caps(node).raster().dtypes.band(0),
        Dtype::U8 { nodata: None }
    );

    let marked = CogSrc::open(cog_of(&band, None), ValueKind::Categorical, Some(3.0)).unwrap();
    let req = WindowReq {
        bbox: Bbox::new(
            ORIGIN_X - 8.0 * CELL,
            ORIGIN_Y - 64.0 * CELL,
            ORIGIN_X + 64.0 * CELL,
            ORIGIN_Y,
        ),
        resolution: CELL,
        time: None,
    };
    let chunk = marked.read(&req).await.unwrap().into_raster().unwrap();
    let band = chunk.bands.band(0).unwrap();
    let (values, nodata) = u8s(band);
    assert_eq!(nodata, Some(3));
    for (index, value) in values.iter().enumerate() {
        let (col, row) = (index % 72, index / 72);
        let want = if col < 8 { 3 } else { code(col - 8, row) };
        assert_eq!(*value, want, "cell {index}");
        assert_eq!(band.is_absent(index), want == 3, "cell {index}");
    }
}

fn test_palette() -> Vec<[u16; 3]> {
    (0..256u32)
        .map(|index| {
            [
                (index * 257) as u16,
                (65535 - index * 257) as u16,
                (index * 100) as u16,
            ]
        })
        .collect()
}

// esa worldcover class codes
const LAND_COVER: [u8; 11] = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100];

fn land_cover(col: usize, row: usize) -> u8 {
    LAND_COVER[(col / 7 + row / 5) % LAND_COVER.len()]
}

#[tokio::test(flavor = "multi_thread")]
async fn a_paletted_cog_exports_its_palette_and_a_computed_layer_does_not() {
    let palette = test_palette();
    let classes = u8_band(land_cover, Some(0));
    let source = CogSrc::open(
        cog_of(&classes, Some(palette.clone())),
        ValueKind::Categorical,
        None,
    )
    .unwrap();
    let mut graph = Graph::new();
    let node = graph.add_source(Box::new(source));
    let reclassified = graph.add_transform(
        node,
        Box::new(MapAlgebra::new(AlgebraOp::Reclassify(vec![
            (0.0, 50.0, 1.0),
            (50.0, 256.0, 2.0),
        ]))),
    );
    let engine = Arc::new(Engine::new(graph, 256 << 20).unwrap());
    assert_eq!(engine.palette(node), Some(palette.as_slice()));
    assert_eq!(engine.palette(reclassified), None);

    let grid = ExportGrid {
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_size: CELL,
        cols: SIDE,
        rows: SIDE,
    };
    let export = |node| {
        let engine = engine.clone();
        async move {
            write_cog(engine, node, grid, None, Cursor::new(Vec::new()))
                .await
                .unwrap()
                .into_inner()
        }
    };
    let bytes = export(node).await;
    let reader = CogReader::open(bytes.as_slice()).unwrap();
    let base = &reader.levels()[0];
    assert_eq!(base.sample_format(), SampleFormat::U8);
    assert_eq!(base.nodata(), Some(0.0));
    assert_eq!(base.palette(), Some(palette.as_slice()));
    let read = reader.read_window_typed(0, 0, 0, SIDE, SIDE).unwrap();
    assert_same_samples(&read[0], &classes);

    let bytes = export(reclassified).await;
    let reader = CogReader::open(bytes.as_slice()).unwrap();
    assert_eq!(reader.levels()[0].palette(), None);
    assert_eq!(reader.levels()[0].sample_format(), SampleFormat::U8);

    let tile = pull_at(&engine, node, window(0, 0, 64, 64), CELL).await;
    let tif = geoplumb::encode::geotiff(&tile, engine.palette(node)).unwrap();
    let reader = CogReader::open(tif.as_slice()).unwrap();
    assert_eq!(reader.levels()[0].palette(), Some(palette.as_slice()));
}

#[tokio::test]
async fn a_viewshed_chunk_is_a_byte_band_with_nodata_255() {
    let wall_column = 300;
    let dem = f32_band(|col, _| if col == wall_column { 400.0 } else { 0.0 });
    let mut graph = Graph::new();
    let source = graph.add_source(Box::new(TypedSrc::new(
        "viewshed dem",
        vec![dem],
        ValueKind::Continuous,
    )));
    let radius_cells = 100.0;
    let seen = graph.add_transform(
        source,
        Box::new(Viewshed {
            observer_x: ORIGIN_X + 256.5 * CELL,
            observer_y: ORIGIN_Y - 256.5 * CELL,
            observer_height: 2.0,
            radius: radius_cells * CELL,
        }),
    );
    let engine = Engine::new(graph, 256 << 20).unwrap();
    assert_eq!(
        engine.caps(seen).raster().dtypes.band(0),
        Dtype::U8 { nodata: Some(255) }
    );
    let pulled = pull(&engine, seen, window(128, 128, 384, 384)).await;
    let (values, nodata) = u8s(pulled.bands.band(0).unwrap());
    assert_eq!(nodata, Some(255));
    let (mut visible, mut hidden) = (0, 0);
    for (index, value) in values.iter().enumerate() {
        let (col, row) = (128 + index % 256, 128 + index / 256);
        let distance = (col as f64 - 256.0).hypot(row as f64 - 256.0);
        if distance > radius_cells + 1.0 {
            assert_eq!(*value, 255, "col {col} row {row} lies past the radius");
            continue;
        }
        if distance < radius_cells - 1.0 && col > wall_column + 1 {
            assert_eq!(*value, 0, "col {col} row {row} is behind the wall");
        }
        match value {
            0 => hidden += 1,
            1 => visible += 1,
            other => assert_eq!(*other, 255, "col {col} row {row}"),
        }
    }
    assert!(
        visible > 100 && hidden > 50,
        "visible {visible}, hidden {hidden}"
    );
}

#[tokio::test]
async fn reclassify_picks_the_smallest_integer_dtype_with_its_top_value_free() {
    let reclassified = |classes: Vec<(f64, f64, f64)>| {
        let mut graph = Graph::new();
        let source = graph.add_source(Box::new(TypedSrc::new(
            "reclassify",
            vec![i16_band()],
            ValueKind::Continuous,
        )));
        let node = graph.add_transform(
            source,
            Box::new(MapAlgebra::new(AlgebraOp::Reclassify(classes))),
        );
        (Engine::new(graph, 256 << 20).unwrap(), node)
    };
    let lowest = f64::from(i16::MIN);
    let cases = [
        (
            vec![(lowest, 0.0, 1.0), (0.0, 40000.0, 2.0)],
            Dtype::U8 { nodata: Some(255) },
        ),
        (
            vec![(lowest, 0.0, -1.0), (0.0, 40000.0, 5.0)],
            Dtype::I8 { nodata: Some(127) },
        ),
        (
            vec![(lowest, 40000.0, 255.0)],
            Dtype::U16 {
                nodata: Some(65535),
            },
        ),
        (
            vec![(lowest, 0.0, -200.0), (0.0, 40000.0, 40000.0)],
            Dtype::I32 {
                nodata: Some(i32::MAX),
            },
        ),
        (vec![(lowest, 40000.0, 0.5)], Dtype::F64),
    ];
    for (classes, dtype) in cases {
        let (engine, node) = reclassified(classes.clone());
        assert_eq!(
            engine.caps(node).raster().dtypes.band(0),
            dtype,
            "{classes:?}"
        );
    }

    let (engine, node) = reclassified(vec![(lowest, 0.0, 1.0), (0.0, 40000.0, 2.0)]);
    let bbox = window(0, 0, 64, 64);
    let pulled = pull(&engine, node, bbox).await;
    let (values, nodata) = u8s(pulled.bands.band(0).unwrap());
    assert_eq!(nodata, Some(255));
    let want: Vec<u8> = in_window(&bbox, |col, row| match elevation(col, row) {
        I16_NODATA => 255,
        height if height < 0 => 1,
        _ => 2,
    });
    assert_eq!(values, want.as_slice());
}
