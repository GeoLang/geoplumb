use geoplumb::caps::{BandDtypes, BandKinds, Dtype, ValueKind};
use geoplumb::elements::color_relief::{ColorFile, ColorSelection, ValueRange};
use geoplumb::elements::{ColorRelief, RasterSrc};
use geoplumb::{Bbox, Caps, Crs, Engine, Graph, WindowReq};
use terrano_core::{BandedRaster, Raster, TypedBand};

const W: usize = 4;
const H: usize = 2;
const CELL: f64 = 1.0;
const ORIGIN_X: f64 = 1000.0;
const ORIGIN_Y: f64 = 2000.0;

const ELEVATIONS: [f64; W * H] = [0.0, 50.0, 100.0, f64::NAN, 25.0, 75.0, 200.0, -10.0];

const COLORS: &str = "0 0 0 0\n100 200 100 50\nnv 255 0 0 128\n";

fn elevation_source() -> RasterSrc {
    let raster = Raster::from_vec(W, H, ELEVATIONS.to_vec(), CELL, f64::NAN).unwrap();
    RasterSrc::new(
        BandedRaster::new(vec![raster]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WEB_MERCATOR,
        ValueKind::Continuous,
    )
}

async fn pull_relief(relief: ColorRelief) -> (Caps, Vec<[u8; 4]>) {
    let mut graph = Graph::new();
    let source = graph.add_source(Box::new(elevation_source()));
    let node = graph.add_transform(source, Box::new(relief));
    let engine = Engine::new(graph, 16 << 20).unwrap();
    let request = WindowReq {
        bbox: Bbox::new(
            ORIGIN_X,
            ORIGIN_Y - H as f64 * CELL,
            ORIGIN_X + W as f64 * CELL,
            ORIGIN_Y,
        ),
        resolution: CELL,
        time: None,
    };
    let chunk = engine
        .pull(node, request)
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let channels: Vec<&[u8]> = chunk
        .bands
        .bands()
        .iter()
        .map(|band| match band {
            TypedBand::U8 {
                samples,
                nodata: None,
            } => samples.values(),
            other => panic!("expected u8 bands without nodata, got {other:?}"),
        })
        .collect();
    assert_eq!(channels.len(), 4);
    let pixels = (0..W * H)
        .map(|index| {
            [
                channels[0][index],
                channels[1][index],
                channels[2][index],
                channels[3][index],
            ]
        })
        .collect();
    (engine.caps(node).clone(), pixels)
}

fn relief(text: &str, selection: ColorSelection, range: Option<ValueRange>) -> ColorRelief {
    ColorRelief::new(ColorFile::parse(text).unwrap(), selection, range).unwrap()
}

#[tokio::test]
async fn color_relief_turns_one_band_into_four_rgba_byte_bands() {
    let (caps, pixels) = pull_relief(relief(COLORS, ColorSelection::Interpolate, None)).await;
    assert_eq!(
        pixels,
        vec![
            [0, 0, 0, 255],
            [100, 50, 25, 255],
            [200, 100, 50, 255],
            [255, 0, 0, 128],
            // gdal rounds 12.5 and 37.5 up
            [50, 25, 13, 255],
            [150, 75, 38, 255],
            [200, 100, 50, 255],
            [0, 0, 0, 255],
        ]
    );
    let raster = caps.raster();
    assert_eq!(raster.bands, 4);
    assert_eq!(raster.dtypes, BandDtypes::from(Dtype::U8 { nodata: None }));
    assert_eq!(raster.value_kinds, BandKinds::from(ValueKind::Continuous));
}

#[tokio::test]
async fn nodata_without_an_nv_entry_is_transparent() {
    let (_, pixels) = pull_relief(relief(
        "0 0 0 0\n100 200 100 50\n",
        ColorSelection::Interpolate,
        None,
    ))
    .await;
    assert_eq!(pixels[3], [0, 0, 0, 0]);
    assert_eq!(pixels[1], [100, 50, 25, 255]);
}

#[tokio::test]
async fn exact_selection_leaves_values_between_stops_transparent() {
    let (_, pixels) = pull_relief(relief(COLORS, ColorSelection::Exact, None)).await;
    assert_eq!(pixels[0], [0, 0, 0, 255]);
    assert_eq!(pixels[1], [0, 0, 0, 0]);
    assert_eq!(pixels[2], [200, 100, 50, 255]);
}

#[tokio::test]
async fn percent_stops_resolve_against_the_given_range() {
    let range = ValueRange {
        min: -100.0,
        max: 300.0,
    };
    let (_, pixels) = pull_relief(relief(
        "25% 0 0 0\n50% 200 100 50\n",
        ColorSelection::Interpolate,
        Some(range),
    ))
    .await;
    // 25% of -100 to 300 is 0 and 50% is 100
    assert_eq!(pixels[1], [100, 50, 25, 255]);
    let percent_file = ColorFile::parse("0% black\n100% white\n").unwrap();
    assert!(ColorRelief::new(percent_file, ColorSelection::Interpolate, None).is_err());
}
