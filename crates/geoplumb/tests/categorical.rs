use geoplumb::caps::{BandKinds, ValueKind};
use geoplumb::elements::algebra::AlgebraOp;
use geoplumb::elements::{CogSrc, Combine, Hillshade, MapAlgebra, Mosaic, RasterSrc, Reproject};
use geoplumb::tile::{XyzTile, render_tile};
use geoplumb::{Bbox, Crs, Engine, Graph, NodeId, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, BinaryOp, CogParams, Raster, write_cog};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

const CODES: [f64; 4] = [10.0, 20.0, 30.0, 80.0];

fn class_map(width: usize, height: usize) -> Raster {
    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        for col in 0..width {
            let code = if row < 5 && col < 7 {
                f64::NAN
            } else {
                CODES[(col / 37 + row / 23 + col / 5 % 2) % CODES.len()]
            };
            data.push(code);
        }
    }
    Raster::from_vec(width, height, data, CELL, f64::NAN).unwrap()
}

fn class_src(value_kind: ValueKind) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![class_map(W, H)]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
        value_kind,
    )
}

fn shifted_class_src(value_kind: ValueKind) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![class_map(W, H)]).unwrap(),
        ORIGIN_X + CELL / 2.0,
        ORIGIN_Y - CELL / 2.0,
        Crs::WGS84,
        value_kind,
    )
}

fn filled_src(value: f64, value_kind: ValueKind) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![
            Raster::from_vec(W, H, vec![value; W * H], CELL, f64::NAN).unwrap(),
        ])
        .unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
        value_kind,
    )
}

fn window(px0: usize, py0: usize, px1: usize, py1: usize) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + px0 as f64 * CELL,
        max_x: ORIGIN_X + px1 as f64 * CELL,
        max_y: ORIGIN_Y - py0 as f64 * CELL,
        min_y: ORIGIN_Y - py1 as f64 * CELL,
    }
}

async fn pull(engine: &Engine, node: NodeId, bbox: Bbox, resolution: f64) -> RasterChunk {
    engine
        .pull(
            node,
            WindowReq {
                bbox,
                resolution,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap()
}

fn band(chunk: &RasterChunk, index: usize) -> Vec<f64> {
    chunk.bands.band(index).unwrap().to_raster().data().to_vec()
}

fn values(chunk: &RasterChunk) -> Vec<f64> {
    band(chunk, 0)
}

fn off_code_in(values: &[f64], codes: &[f64]) -> Vec<f64> {
    values
        .iter()
        .copied()
        .filter(|v| !v.is_nan() && !codes.contains(v))
        .collect()
}

fn off_code(chunk: &RasterChunk, codes: &[f64]) -> Vec<f64> {
    off_code_in(&values(chunk), codes)
}

fn tile_of(z: u8, lon: f64, lat: f64) -> XyzTile {
    let n = f64::from(1u32 << z);
    let x = ((lon + 180.0) / 360.0 * n).floor() as u32;
    let r = lat.to_radians();
    let y =
        ((1.0 - (r.tan() + 1.0 / r.cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor() as u32;
    XyzTile { z, x, y }
}

fn engine_over(src: RasterSrc) -> (Engine, NodeId) {
    let mut g = Graph::new();
    let node = g.add_source(Box::new(src));
    (Engine::new(g, 64 << 20).unwrap(), node)
}

#[tokio::test]
async fn raster_src_decimates_categories_by_mode_and_continuous_by_mean() {
    let nan = f64::NAN;
    #[rustfmt::skip]
    let block = vec![
        10.0, 10.0, 20.0, 30.0,
        20.0, 30.0, 30.0, 20.0,
        nan,  nan,  40.0, nan,
        nan,  nan,  50.0, 50.0,
    ];
    let src = |value_kind: ValueKind| {
        RasterSrc::new(
            BandedRaster::new(vec![
                Raster::from_vec(4, 4, block.clone(), CELL, nan).unwrap(),
            ])
            .unwrap(),
            ORIGIN_X,
            ORIGIN_Y,
            Crs::WGS84,
            value_kind,
        )
    };

    let (engine, node) = engine_over(src(ValueKind::Categorical));
    let coarse = pull(&engine, node, window(0, 0, 4, 4), 2.0 * CELL).await;
    let got = values(&coarse);
    assert_eq!(got.len(), 4);
    assert_eq!(got[0], 10.0);
    assert_eq!(got[1], 20.0, "a tie did not go to the smaller code");
    assert!(got[2].is_nan(), "all-nodata block came out {}", got[2]);
    assert_eq!(got[3], 50.0);

    let (engine, node) = engine_over(src(ValueKind::Continuous));
    let coarse = pull(&engine, node, window(0, 0, 4, 4), 2.0 * CELL).await;
    let got = values(&coarse);
    assert_eq!(got[0], 17.5);
    assert_eq!(got[1], 25.0);
    assert!(got[2].is_nan());
    assert_eq!(got[3], 140.0 / 3.0);
}

#[tokio::test]
async fn cog_src_decimates_categories_by_mode() {
    let side = 64;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("classes.tif");
    let params = CogParams {
        tile_width: 64,
        tile_height: 64,
        // terrano averages the overviews it writes
        overview_levels: 0,
        epsg: 4326,
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        pixel_width: CELL,
        pixel_height: CELL,
        deflate: false,
        ..CogParams::default()
    };
    write_cog(
        &class_map(side, side),
        &params,
        &mut std::fs::File::create(&path).unwrap(),
    )
    .unwrap();
    let open = |value_kind: ValueKind| {
        CogSrc::open(std::fs::File::open(&path).unwrap(), value_kind).unwrap()
    };
    let bbox = window(0, 0, side, side);
    let resolution = 4.0 * CELL;

    let mut g = Graph::new();
    let node = g.add_source(Box::new(open(ValueKind::Categorical)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let categorical = pull(&engine, node, bbox, resolution).await;
    assert_eq!(categorical.width(), side / 4);
    assert_eq!(off_code(&categorical, &CODES), Vec::<f64>::new());

    let memory = RasterSrc::new(
        BandedRaster::new(vec![class_map(side, side)]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
        ValueKind::Categorical,
    );
    let (engine, node) = engine_over(memory);
    let reference = pull(&engine, node, bbox, resolution).await;
    for (i, (a, b)) in values(&categorical)
        .iter()
        .zip(&values(&reference))
        .enumerate()
    {
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "pixel {i}: cog {a} vs memory {b}"
        );
    }

    let mut g = Graph::new();
    let node = g.add_source(Box::new(open(ValueKind::Continuous)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let continuous = pull(&engine, node, bbox, resolution).await;
    assert!(
        !off_code(&continuous, &CODES).is_empty(),
        "no block of the fixture mixes codes"
    );
}

#[tokio::test]
async fn reprojected_categories_stay_class_codes() {
    let center = tile_of(
        10,
        ORIGIN_X + W as f64 * CELL / 2.0,
        ORIGIN_Y - H as f64 * CELL / 2.0,
    );
    let render = move |value_kind| async move {
        let mut g = Graph::new();
        let src = g.add_source(Box::new(class_src(value_kind)));
        let rp = g.add_transform(src, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
        let engine = Engine::new(g, 64 << 20).unwrap();
        assert_eq!(
            engine.caps(rp).raster().value_kinds,
            BandKinds::from(value_kind)
        );
        let warped = pull(&engine, rp, center.bbox(), center.resolution()).await;
        let rendered = render_tile(&engine, rp, center).await.unwrap();
        (warped, rendered)
    };

    let (warped, rendered) = render(ValueKind::Categorical).await;
    assert_eq!(off_code(&warped, &CODES), Vec::<f64>::new());
    assert_eq!(off_code(&rendered, &CODES), Vec::<f64>::new());
    assert!(
        values(&rendered).iter().any(|v| !v.is_nan()),
        "the tile missed the class map"
    );

    let (warped, rendered) = render(ValueKind::Continuous).await;
    assert!(
        !off_code(&warped, &CODES).is_empty(),
        "the tile crosses no class boundary"
    );
    assert!(!off_code(&rendered, &CODES).is_empty());
}

#[tokio::test]
async fn reclassified_tiles_hold_only_class_values() {
    let mut dem = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            dem.push(500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos());
        }
    }
    let dem = Raster::from_vec(W, H, dem, CELL, f64::NAN).unwrap();
    let classes = [1.0, 2.0, 3.0];
    let mut g = Graph::new();
    let src = g.add_source(Box::new(RasterSrc::new(
        BandedRaster::new(vec![dem]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
        ValueKind::Continuous,
    )));
    let classified = g.add_transform(
        src,
        Box::new(MapAlgebra::new(AlgebraOp::Reclassify(vec![
            (0.0, 450.0, classes[0]),
            (450.0, 550.0, classes[1]),
            (550.0, 1000.0, classes[2]),
        ]))),
    );
    let rp = g.add_transform(classified, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let tile = tile_of(
        10,
        ORIGIN_X + W as f64 * CELL / 2.0,
        ORIGIN_Y - H as f64 * CELL / 2.0,
    );
    let rendered = render_tile(&engine, rp, tile).await.unwrap();
    assert_eq!(off_code(&rendered, &classes), Vec::<f64>::new());
    let present: Vec<f64> = classes
        .iter()
        .copied()
        .filter(|class| values(&rendered).contains(class))
        .collect();
    assert!(present.len() >= 2, "the tile crosses no class boundary");
}

#[test]
fn reclassify_pins_categorical_and_hillshade_pins_continuous() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(class_src(ValueKind::Categorical)));
    let shaded = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let classified = g.add_transform(
        shaded,
        Box::new(MapAlgebra::new(AlgebraOp::Reclassify(vec![(
            0.0, 128.0, 1.0,
        )]))),
    );
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(
        engine.caps(src).raster().value_kinds,
        BandKinds::from(ValueKind::Categorical)
    );
    assert_eq!(
        engine.caps(shaded).raster().value_kinds,
        BandKinds::from(ValueKind::Continuous)
    );
    assert_eq!(
        engine.caps(classified).raster().value_kinds,
        BandKinds::from(ValueKind::Categorical)
    );
}

#[tokio::test]
async fn mosaic_samples_categories_from_the_containing_pixel() {
    let mosaic = |value_kind| async move {
        let mut g = Graph::new();
        let empty = g.add_source(Box::new(filled_src(f64::NAN, value_kind)));
        let classes = g.add_source(Box::new(shifted_class_src(value_kind)));
        let joined = g.add_fanin(&[empty, classes], Box::new(Mosaic::default()));
        let engine = Engine::new(g, 64 << 20).unwrap();
        assert_eq!(
            engine.caps(joined).raster().value_kinds,
            BandKinds::from(value_kind)
        );
        pull(&engine, joined, window(20, 20, 300, 200), CELL).await
    };

    let categorical = mosaic(ValueKind::Categorical).await;
    assert_eq!(off_code(&categorical, &CODES), Vec::<f64>::new());
    assert!(values(&categorical).iter().any(|v| !v.is_nan()));

    let continuous = mosaic(ValueKind::Continuous).await;
    assert!(
        !off_code(&continuous, &CODES).is_empty(),
        "the half-cell shift blends no codes"
    );
}

#[tokio::test]
async fn combine_samples_each_input_by_its_own_kind_and_outputs_continuous() {
    let multiply = |class_kind| async move {
        let mut g = Graph::new();
        let ones = g.add_source(Box::new(filled_src(1.0, ValueKind::Continuous)));
        let classes = g.add_source(Box::new(shifted_class_src(class_kind)));
        let product = g.add_fanin(&[ones, classes], Box::new(Combine::new(BinaryOp::Multiply)));
        let engine = Engine::new(g, 64 << 20).unwrap();
        assert_eq!(
            engine.caps(product).raster().value_kinds,
            BandKinds::from(ValueKind::Continuous)
        );
        assert_eq!(
            engine.caps(classes).raster().value_kinds,
            BandKinds::from(class_kind)
        );
        pull(&engine, product, window(20, 20, 300, 200), CELL).await
    };
    // bilinear weights over the ones input sum to one only within rounding
    let off_codes = |chunk: &RasterChunk| -> Vec<f64> {
        values(chunk)
            .iter()
            .copied()
            .filter(|v| !v.is_nan() && !CODES.iter().any(|code| (v - code).abs() < 1e-9))
            .collect()
    };

    let categorical = multiply(ValueKind::Categorical).await;
    assert_eq!(off_codes(&categorical), Vec::<f64>::new());
    assert!(values(&categorical).iter().any(|v| !v.is_nan()));

    let continuous = multiply(ValueKind::Continuous).await;
    assert!(
        !off_codes(&continuous).is_empty(),
        "the half-cell shift blends no codes"
    );
}

#[test]
fn a_mosaic_of_two_value_kinds_fails_naming_value_kind() {
    let mut g = Graph::new();
    let categorical = g.add_source(Box::new(class_src(ValueKind::Categorical)));
    let continuous = g.add_source(Box::new(class_src(ValueKind::Continuous)));
    g.add_fanin(&[categorical, continuous], Box::new(Mosaic::default()));
    let err = Engine::new(g, 64 << 20)
        .err()
        .expect("a mosaic mixing value kinds solved");
    assert!(
        err.to_string().contains("no common value for value_kinds"),
        "{err}"
    );
}

fn reflectance_map(width: usize, height: usize) -> Raster {
    let mut data = Vec::with_capacity(width * height);
    for row in 0..height {
        for col in 0..width {
            data.push(1000.0 + 3.0 * col as f64 + 7.0 * row as f64);
        }
    }
    Raster::from_vec(width, height, data, CELL, f64::NAN).unwrap()
}

fn empty_band() -> Raster {
    Raster::from_vec(W, H, vec![f64::NAN; W * H], CELL, f64::NAN).unwrap()
}

fn banded_src(bands: Vec<Raster>, shift: f64, value_kinds: BandKinds) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(bands).unwrap(),
        ORIGIN_X + shift,
        ORIGIN_Y - shift,
        Crs::WGS84,
        value_kinds,
    )
}

fn reflectance_and_classes() -> BandKinds {
    BandKinds::from_bands(vec![ValueKind::Continuous, ValueKind::Categorical])
}

fn two_band_src(value_kinds: BandKinds) -> RasterSrc {
    banded_src(
        vec![reflectance_map(W, H), class_map(W, H)],
        0.0,
        value_kinds,
    )
}

fn reflectance_src() -> RasterSrc {
    banded_src(
        vec![reflectance_map(W, H)],
        0.0,
        BandKinds::from(ValueKind::Continuous),
    )
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn band_kinds_shared_by_every_band_equal_the_single_kind() {
    let both_categorical = BandKinds::from_bands(vec![ValueKind::Categorical; 2]);
    assert_eq!(both_categorical, BandKinds::from(ValueKind::Categorical));
    assert_eq!(both_categorical.band(1), ValueKind::Categorical);
    let mixed = reflectance_and_classes();
    assert_ne!(mixed, BandKinds::from(ValueKind::Continuous));
    assert_eq!(mixed.band(0), ValueKind::Continuous);
    assert_eq!(mixed.band(1), ValueKind::Categorical);
}

#[tokio::test]
async fn a_two_band_source_decimates_each_band_by_its_own_kind() {
    let bbox = window(0, 0, W, H);
    let resolution = 4.0 * CELL;
    let decimated = move |src: RasterSrc| async move {
        let (engine, node) = engine_over(src);
        let kinds = engine.caps(node).raster().value_kinds.clone();
        (kinds, pull(&engine, node, bbox, resolution).await)
    };

    let (kinds, mixed) = decimated(two_band_src(reflectance_and_classes())).await;
    assert_eq!(kinds, reflectance_and_classes());
    assert_eq!(off_code_in(&band(&mixed, 1), &CODES), Vec::<f64>::new());
    assert!(band(&mixed, 1).iter().any(|v| !v.is_nan()));
    let (_, reference) = decimated(reflectance_src()).await;
    assert_eq!(bits(&band(&mixed, 0)), bits(&band(&reference, 0)));

    let (_, every_band_continuous) =
        decimated(two_band_src(BandKinds::from(ValueKind::Continuous))).await;
    assert!(
        !off_code_in(&band(&every_band_continuous, 1), &CODES).is_empty(),
        "no block of the fixture mixes codes"
    );
    let (_, every_band_categorical) =
        decimated(two_band_src(BandKinds::from(ValueKind::Categorical))).await;
    assert_ne!(
        bits(&band(&every_band_categorical, 0)),
        bits(&band(&reference, 0))
    );
}

#[tokio::test]
async fn a_two_band_warp_samples_each_band_by_its_own_kind() {
    let center = tile_of(
        10,
        ORIGIN_X + W as f64 * CELL / 2.0,
        ORIGIN_Y - H as f64 * CELL / 2.0,
    );
    let render = move |src: RasterSrc| async move {
        let mut g = Graph::new();
        let node = g.add_source(Box::new(src));
        let rp = g.add_transform(node, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
        let engine = Engine::new(g, 64 << 20).unwrap();
        let warped = pull(&engine, rp, center.bbox(), center.resolution()).await;
        let rendered = render_tile(&engine, rp, center).await.unwrap();
        (warped, rendered)
    };

    let (warped, rendered) = render(two_band_src(reflectance_and_classes())).await;
    let (warped_reference, rendered_reference) = render(reflectance_src()).await;
    for chunk in [&warped, &rendered] {
        assert_eq!(off_code_in(&band(chunk, 1), &CODES), Vec::<f64>::new());
        assert!(
            band(chunk, 1).iter().any(|v| !v.is_nan()),
            "the tile missed the class map"
        );
    }
    assert_eq!(bits(&band(&warped, 0)), bits(&band(&warped_reference, 0)));
    assert_eq!(
        bits(&band(&rendered, 0)),
        bits(&band(&rendered_reference, 0))
    );

    let (warped, rendered) = render(two_band_src(BandKinds::from(ValueKind::Continuous))).await;
    assert!(
        !off_code_in(&band(&warped, 1), &CODES).is_empty(),
        "the tile crosses no class boundary"
    );
    assert!(!off_code_in(&band(&rendered, 1), &CODES).is_empty());
    let (warped, _) = render(two_band_src(BandKinds::from(ValueKind::Categorical))).await;
    assert_ne!(bits(&band(&warped, 0)), bits(&band(&warped_reference, 0)));
}

#[tokio::test]
async fn a_two_band_mosaic_samples_each_band_by_its_own_kind() {
    let mosaic = |bands: Vec<Raster>, value_kinds: BandKinds| async move {
        let empties = bands.iter().map(|_| empty_band()).collect();
        let mut g = Graph::new();
        let empty = g.add_source(Box::new(banded_src(empties, 0.0, value_kinds.clone())));
        let shifted = g.add_source(Box::new(banded_src(bands, CELL / 2.0, value_kinds)));
        let joined = g.add_fanin(&[empty, shifted], Box::new(Mosaic::default()));
        let engine = Engine::new(g, 64 << 20).unwrap();
        pull(&engine, joined, window(20, 20, 300, 200), CELL).await
    };
    let both = || vec![reflectance_map(W, H), class_map(W, H)];

    let mixed = mosaic(both(), reflectance_and_classes()).await;
    assert_eq!(off_code_in(&band(&mixed, 1), &CODES), Vec::<f64>::new());
    assert!(band(&mixed, 1).iter().any(|v| !v.is_nan()));
    let reference = mosaic(
        vec![reflectance_map(W, H)],
        BandKinds::from(ValueKind::Continuous),
    )
    .await;
    assert_eq!(bits(&band(&mixed, 0)), bits(&band(&reference, 0)));

    let every_band_continuous = mosaic(both(), BandKinds::from(ValueKind::Continuous)).await;
    assert!(
        !off_code_in(&band(&every_band_continuous, 1), &CODES).is_empty(),
        "the half-cell shift blends no codes"
    );
    let every_band_categorical = mosaic(both(), BandKinds::from(ValueKind::Categorical)).await;
    assert_ne!(
        bits(&band(&every_band_categorical, 0)),
        bits(&band(&reference, 0))
    );
}

#[cfg(feature = "gdal")]
mod gdal {
    use super::*;
    use geoplumb::element::Source;
    use geoplumb::elements::GdalSrc;
    use terrano_core::{GeoTiffMetadata, SampleFormat, write_geotiff_bands};

    async fn read(src: &impl Source, req: &WindowReq) -> RasterChunk {
        src.read(req).await.unwrap().into_raster().unwrap()
    }

    #[tokio::test]
    async fn gdal_src_decimates_categories_by_mode() {
        let side = 64;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("classes.tif");
        let meta = GeoTiffMetadata {
            origin_x: ORIGIN_X,
            origin_y: ORIGIN_Y,
            pixel_width: CELL,
            pixel_height: CELL,
            epsg: 4326,
        };
        let mut bytes = Vec::new();
        write_geotiff_bands(
            &BandedRaster::new(vec![class_map(side, side)]).unwrap(),
            &meta,
            SampleFormat::F64,
            &mut bytes,
        )
        .unwrap();
        std::fs::write(&path, &bytes).unwrap();
        let req = WindowReq {
            bbox: window(0, 0, side, side),
            resolution: 4.0 * CELL,
            time: None,
        };

        let categorical = read(&GdalSrc::open(&path, ValueKind::Categorical).unwrap(), &req).await;
        assert_eq!(categorical.width(), side / 4);
        assert_eq!(off_code(&categorical, &CODES), Vec::<f64>::new());

        let memory = RasterSrc::new(
            BandedRaster::new(vec![class_map(side, side)]).unwrap(),
            ORIGIN_X,
            ORIGIN_Y,
            Crs::WGS84,
            ValueKind::Categorical,
        );
        let reference = read(&memory, &req).await;
        for (i, (a, b)) in values(&categorical)
            .iter()
            .zip(&values(&reference))
            .enumerate()
        {
            assert!(
                a == b || (a.is_nan() && b.is_nan()),
                "pixel {i}: gdal {a} vs memory {b}"
            );
        }

        let continuous = read(&GdalSrc::open(&path, ValueKind::Continuous).unwrap(), &req).await;
        assert!(
            !off_code(&continuous, &CODES).is_empty(),
            "no block of the fixture mixes codes"
        );
    }
}
