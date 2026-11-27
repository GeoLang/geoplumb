use futures::future::BoxFuture;
use geoplumb::caps::{
    BandDtypes, BandKinds, CapsPattern, CapsSet, Constraint, Dtype, RasterPattern, ResRange,
    SetField, ValueKind,
};
use geoplumb::element::Source;
use geoplumb::elements::{ClipRaster, VecSrc};
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, NodeId, RasterBands, RasterChunk, WindowReq};
use terrano_core::{SampleGrid, TypedBand};

const SIDE: usize = 16;
const UTM: Crs = Crs(32632);

struct Grid {
    origin_x: f64,
    origin_y: f64,
    cell: f64,
    crs: Crs,
}

// pixel centres on a 1 m grid from (0, 16) sit at half metres
const UNIT_GRID: Grid = Grid {
    origin_x: 0.0,
    origin_y: 16.0,
    cell: 1.0,
    crs: UTM,
};

impl Grid {
    fn bbox(&self) -> Bbox {
        let side = SIDE as f64 * self.cell;
        Bbox::new(
            self.origin_x,
            self.origin_y - side,
            self.origin_x + side,
            self.origin_y,
        )
    }

    fn centre(&self, col: usize, row: usize) -> (f64, f64) {
        (
            self.origin_x + (col as f64 + 0.5) * self.cell,
            self.origin_y - (row as f64 + 0.5) * self.cell,
        )
    }
}

fn samples<T>(grid: &Grid, value: impl Fn(usize, usize) -> T) -> SampleGrid<T> {
    let values = (0..SIDE * SIDE)
        .map(|index| value(index % SIDE, index / SIDE))
        .collect();
    SampleGrid::new(SIDE, SIDE, grid.cell, values).unwrap()
}

struct BandsSrc {
    full: RasterChunk,
    chunk_px: u32,
}

impl BandsSrc {
    fn new(grid: &Grid, bands: Vec<TypedBand>, chunk_px: u32) -> BandsSrc {
        BandsSrc {
            full: RasterChunk {
                bands: RasterBands::new(bands).unwrap(),
                bbox: grid.bbox(),
                resolution: grid.cell,
                crs: grid.crs,
            },
            chunk_px,
        }
    }
}

impl Source for BandsSrc {
    fn identity(&self) -> String {
        format!("BandsSrc chunk_px={}", self.chunk_px)
    }

    fn constraint(&self) -> Constraint {
        let dtypes = self.full.bands.bands().iter().map(Dtype::of).collect();
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtypes: SetField::one(BandDtypes::from_bands(dtypes)),
            bands: SetField::one(self.full.bands.band_count() as u16),
            crs: SetField::one(self.full.crs),
            resolution: ResRange::at_least(self.full.resolution),
            chunk_px: SetField::one(self.chunk_px),
            value_kinds: SetField::one(BandKinds::from(ValueKind::Continuous)),
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.full.bbox.min_x,
            origin_y: self.full.bbox.max_y,
            base_resolution: self.full.resolution,
            chunk_px: self.chunk_px,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        Box::pin(async move { Ok(Chunk::Raster(self.full.crop_to(&req.bbox))) })
    }
}

fn polygon_layer(rings: &[Vec<[f64; 2]>], crs: Crs) -> VecSrc {
    let json = serde_json::json!({
        "type": "Feature",
        "properties": {},
        "geometry": { "type": "Polygon", "coordinates": rings },
    });
    VecSrc::from_geojson(&json.to_string(), crs).unwrap()
}

fn rectangle(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Vec<[f64; 2]> {
    vec![
        [min_x, min_y],
        [max_x, min_y],
        [max_x, max_y],
        [min_x, max_y],
        [min_x, min_y],
    ]
}

// edges 0.1 m off the pixel centres on either side, around a hole
fn holed_square() -> VecSrc {
    polygon_layer(
        &[
            rectangle(2.4, 2.4, 13.6, 13.6),
            rectangle(6.6, 6.6, 9.4, 9.4),
        ],
        UTM,
    )
}

fn inside_holed_square(x: f64, y: f64) -> bool {
    let within = |low: f64, high: f64| low < x && x < high && low < y && y < high;
    within(2.4, 13.6) && !within(6.6, 9.4)
}

fn clipped(
    source: BandsSrc,
    cutline: VecSrc,
    clip: ClipRaster,
) -> geoplumb::Result<(Engine, NodeId)> {
    let mut graph = Graph::new();
    let raster = graph.add_source(Box::new(source));
    let polygons = graph.add_source(Box::new(cutline));
    let node = graph.add_fanin(&[raster, polygons], Box::new(clip));
    Ok((Engine::new(graph, 64 << 20)?, node))
}

async fn pull_whole(engine: &Engine, node: NodeId, grid: &Grid) -> RasterChunk {
    let req = WindowReq {
        bbox: grid.bbox(),
        resolution: grid.cell,
        time: None,
    };
    engine.pull(node, req).await.unwrap().into_raster().unwrap()
}

fn f32s(band: &TypedBand) -> &[f32] {
    match band {
        TypedBand::F32(samples) => samples.values(),
        other => panic!("expected an f32 band, got {}", other.format().name()),
    }
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

fn reflectance(col: usize, row: usize) -> f32 {
    (col * SIDE + row) as f32 + 0.25
}

fn code(col: usize, row: usize) -> u8 {
    ((col * 7 + row * 3) % 11) as u8
}

fn elevation(col: usize, row: usize) -> i16 {
    col as i16 * 37 - row as i16 * 53
}

#[tokio::test]
async fn a_polygon_with_a_hole_keeps_the_pixels_whose_centres_fall_inside_it() {
    let source = BandsSrc::new(
        &UNIT_GRID,
        vec![TypedBand::F32(samples(&UNIT_GRID, reflectance))],
        16,
    );
    let (engine, node) = clipped(source, holed_square(), ClipRaster::new(None)).unwrap();
    let chunk = pull_whole(&engine, node, &UNIT_GRID).await;
    assert_eq!(chunk.bbox, UNIT_GRID.bbox());
    let values = f32s(chunk.bands.band(0).unwrap());

    // row 7 has its centres at y 8.5, through the hole
    let at = |col: usize| values[7 * SIDE + col];
    assert!(at(1).is_nan(), "centre 1.5 is outside the edge at 2.4");
    assert_eq!(
        at(2),
        reflectance(2, 7),
        "centre 2.5 is inside the edge at 2.4"
    );
    assert_eq!(
        at(6),
        reflectance(6, 7),
        "centre 6.5 is outside the hole edge at 6.6"
    );
    assert!(at(7).is_nan(), "centre 7.5 is inside the hole");
    assert!(at(8).is_nan(), "centre 8.5 is inside the hole");
    assert_eq!(
        at(9),
        reflectance(9, 7),
        "centre 9.5 is outside the hole edge at 9.4"
    );
    assert_eq!(
        at(13),
        reflectance(13, 7),
        "centre 13.5 is inside the edge at 13.6"
    );
    assert!(at(14).is_nan(), "centre 14.5 is outside the edge at 13.6");

    for row in 0..SIDE {
        for col in 0..SIDE {
            let (x, y) = UNIT_GRID.centre(col, row);
            let got = values[row * SIDE + col];
            if inside_holed_square(x, y) {
                assert_eq!(got, reflectance(col, row), "pixel {col},{row}");
            } else {
                assert!(got.is_nan(), "pixel {col},{row} is outside, got {got}");
            }
        }
    }
}

#[tokio::test]
async fn every_band_is_masked_in_its_own_dtype_and_sentinel() {
    let bands = vec![
        TypedBand::U8 {
            samples: samples(&UNIT_GRID, code),
            nodata: Some(255),
        },
        TypedBand::I16 {
            samples: samples(&UNIT_GRID, elevation),
            nodata: Some(i16::MIN),
        },
        TypedBand::F32(samples(&UNIT_GRID, reflectance)),
    ];
    let source = BandsSrc::new(&UNIT_GRID, bands, 16);
    let (engine, node) = clipped(source, holed_square(), ClipRaster::new(None)).unwrap();
    let dtypes = BandDtypes::from_bands(vec![
        Dtype::U8 { nodata: Some(255) },
        Dtype::I16 {
            nodata: Some(i16::MIN),
        },
        Dtype::F32,
    ]);
    assert_eq!(engine.caps(node).raster().dtypes, dtypes);

    let chunk = pull_whole(&engine, node, &UNIT_GRID).await;
    let (codes, code_nodata) = u8s(chunk.bands.band(0).unwrap());
    let (heights, height_nodata) = i16s(chunk.bands.band(1).unwrap());
    let reflectances = f32s(chunk.bands.band(2).unwrap());
    assert_eq!(code_nodata, Some(255));
    assert_eq!(height_nodata, Some(i16::MIN));
    let mut outside = 0;
    for row in 0..SIDE {
        for col in 0..SIDE {
            let (x, y) = UNIT_GRID.centre(col, row);
            let index = row * SIDE + col;
            if inside_holed_square(x, y) {
                assert_eq!(codes[index], code(col, row), "code {col},{row}");
                assert_eq!(heights[index], elevation(col, row), "height {col},{row}");
                assert_eq!(
                    reflectances[index],
                    reflectance(col, row),
                    "f32 {col},{row}"
                );
            } else {
                outside += 1;
                assert_eq!(codes[index], 255, "code {col},{row}");
                assert_eq!(heights[index], i16::MIN, "height {col},{row}");
                assert!(reflectances[index].is_nan(), "f32 {col},{row}");
            }
        }
    }
    assert!(outside > 0 && outside < SIDE * SIDE);
}

#[tokio::test]
async fn a_band_without_nodata_takes_the_clip_nodata_as_its_sentinel() {
    let bands = vec![TypedBand::U8 {
        samples: samples(&UNIT_GRID, code),
        nodata: None,
    }];
    let source = BandsSrc::new(&UNIT_GRID, bands, 16);
    let (engine, node) = clipped(source, holed_square(), ClipRaster::new(Some(0.0))).unwrap();
    assert_eq!(
        engine.caps(node).raster().dtypes,
        BandDtypes::from(Dtype::U8 { nodata: Some(0) })
    );

    let chunk = pull_whole(&engine, node, &UNIT_GRID).await;
    let (codes, nodata) = u8s(chunk.bands.band(0).unwrap());
    assert_eq!(nodata, Some(0));
    let mut moved_off_nodata = 0;
    for row in 0..SIDE {
        for col in 0..SIDE {
            let (x, y) = UNIT_GRID.centre(col, row);
            let got = codes[row * SIDE + col];
            let want = match (inside_holed_square(x, y), code(col, row)) {
                (false, _) => 0,
                // gdal's warp moves a valid value off the nodata one step up at the type's minimum
                (true, 0) => {
                    moved_off_nodata += 1;
                    1
                }
                (true, value) => value,
            };
            assert_eq!(got, want, "pixel {col},{row}");
        }
    }
    assert!(
        moved_off_nodata > 0,
        "the fixture puts a 0 inside the polygon"
    );
}

#[test]
fn a_clip_refuses_a_band_it_cannot_mark_and_a_nodata_it_cannot_use() {
    let unmarked = || {
        BandsSrc::new(
            &UNIT_GRID,
            vec![TypedBand::U8 {
                samples: samples(&UNIT_GRID, code),
                nodata: None,
            }],
            16,
        )
    };
    let Err(error) = clipped(unmarked(), holed_square(), ClipRaster::new(None)) else {
        panic!("a u8 band without nodata has no value for the pixels outside");
    };
    assert!(error.to_string().contains("nodata"), "{error}");

    let floats = BandsSrc::new(
        &UNIT_GRID,
        vec![TypedBand::F32(samples(&UNIT_GRID, reflectance))],
        16,
    );
    let Err(error) = clipped(floats, holed_square(), ClipRaster::new(Some(0.0))) else {
        panic!("NaN marks the absent cells of an f32 band");
    };
    assert!(error.to_string().contains("takes no nodata"), "{error}");

    let Err(error) = clipped(unmarked(), holed_square(), ClipRaster::new(Some(300.0))) else {
        panic!("300 is not a u8 value");
    };
    assert!(error.to_string().contains("300"), "{error}");
}

// vertices off every pixel centre, the slanted edges crossing the chunk seam at x 8
fn triangle() -> VecSrc {
    polygon_layer(
        &[vec![[1.3, 1.1], [14.7, 3.9], [9.2, 14.6], [1.3, 1.1]]],
        UTM,
    )
}

fn inside_triangle(x: f64, y: f64) -> bool {
    let corners = [(1.3, 1.1), (14.7, 3.9), (9.2, 14.6)];
    (0..3).all(|i| {
        let (ax, ay) = corners[i];
        let (bx, by) = corners[(i + 1) % 3];
        (bx - ax) * (y - ay) - (by - ay) * (x - ax) > 0.0
    })
}

#[tokio::test]
async fn chunks_meeting_at_a_seam_clip_as_one_chunk_does() {
    let band = || vec![TypedBand::F32(samples(&UNIT_GRID, reflectance))];
    let (whole_engine, whole_node) = clipped(
        BandsSrc::new(&UNIT_GRID, band(), 16),
        triangle(),
        ClipRaster::new(None),
    )
    .unwrap();
    let (tiled_engine, tiled_node) = clipped(
        BandsSrc::new(&UNIT_GRID, band(), 8),
        triangle(),
        ClipRaster::new(None),
    )
    .unwrap();
    assert_eq!(whole_engine.grid(whole_node).chunk_px, 16);
    assert_eq!(tiled_engine.grid(tiled_node).chunk_px, 8);

    let whole = pull_whole(&whole_engine, whole_node, &UNIT_GRID).await;
    let tiled = pull_whole(&tiled_engine, tiled_node, &UNIT_GRID).await;
    let whole = f32s(whole.bands.band(0).unwrap());
    let tiled = f32s(tiled.bands.band(0).unwrap());
    let mut kept = 0;
    for row in 0..SIDE {
        for col in 0..SIDE {
            let index = row * SIDE + col;
            let (x, y) = UNIT_GRID.centre(col, row);
            let want = if inside_triangle(x, y) {
                kept += 1;
                reflectance(col, row)
            } else {
                f32::NAN
            };
            for (name, got) in [("whole", whole[index]), ("tiled", tiled[index])] {
                assert!(
                    got == want || (got.is_nan() && want.is_nan()),
                    "{name} pixel {col},{row}: {got} against {want}"
                );
            }
        }
    }
    assert!(kept > 0 && kept < SIDE * SIDE);
}

#[tokio::test]
async fn a_cutline_in_another_crs_is_reprojected_onto_the_raster() {
    // a 10 m web mercator grid near 7 east, 47 north
    let grid = Grid {
        origin_x: 779_000.0,
        origin_y: 5_942_200.0,
        cell: 10.0,
        crs: Crs::WEB_MERCATOR,
    };
    let (min_x, min_y, max_x, max_y) = (779_033.0, 5_942_071.0, 779_121.0, 5_942_157.0);
    let to_lonlat = projicio_core::Transform::new("EPSG:3857", "EPSG:4326").unwrap();
    let (west, south) = to_lonlat.convert(min_x, min_y).unwrap();
    let (east, north) = to_lonlat.convert(max_x, max_y).unwrap();
    let cutline = polygon_layer(&[rectangle(west, south, east, north)], Crs::WGS84);

    let source = BandsSrc::new(&grid, vec![TypedBand::F32(samples(&grid, reflectance))], 16);
    let (engine, node) = clipped(source, cutline, ClipRaster::new(None)).unwrap();
    assert_eq!(engine.caps(node).crs(), Crs::WEB_MERCATOR);
    let chunk = pull_whole(&engine, node, &grid).await;
    let values = f32s(chunk.bands.band(0).unwrap());
    let mut kept = 0;
    for row in 0..SIDE {
        for col in 0..SIDE {
            // the nearest edge is 2 m from any centre, far past the round trip's error
            let (x, y) = grid.centre(col, row);
            let got = values[row * SIDE + col];
            if min_x < x && x < max_x && min_y < y && y < max_y {
                kept += 1;
                assert_eq!(got, reflectance(col, row), "pixel {col},{row}");
            } else {
                assert!(got.is_nan(), "pixel {col},{row} is outside, got {got}");
            }
        }
    }
    assert!(kept > 0 && kept < SIDE * SIDE);
}
