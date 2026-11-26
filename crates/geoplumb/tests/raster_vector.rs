use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use geoplumb::caps::{
    BandKinds, CapsPattern, CapsSet, Constraint, RasterPattern, SetField, ValueKind,
};
use geoplumb::element::{Source, Transform};
use geoplumb::elements::{Contours, Polygonize, RasterSrc, Viewshed};
use geoplumb::window::{ChunkKey, GridSpec};
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, NodeId, RasterChunk, VectorChunk, WindowReq};
use terrano_core::{BandedRaster, Raster};
use topoi_core::Coord;
use topoi_core::geojson::FeatureGeometry;

const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;
// not a binary fraction, so two formulas for one tile edge can round apart
const CELL: f64 = 0.001;
const W: usize = 96;
const H: usize = 80;

fn source(
    cell: f64,
    columns: usize,
    rows: usize,
    value_kind: ValueKind,
    value: impl Fn(usize, usize) -> f64,
) -> RasterSrc {
    let mut data = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for column in 0..columns {
            data.push(value(column, row));
        }
    }
    RasterSrc::new(
        BandedRaster::new(vec![
            Raster::from_vec(columns, rows, data, cell, f64::NAN).unwrap(),
        ])
        .unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
        value_kind,
    )
}

fn window(cell: f64, px0: usize, py0: usize, px1: usize, py1: usize) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + px0 as f64 * cell,
        max_x: ORIGIN_X + px1 as f64 * cell,
        max_y: ORIGIN_Y - py0 as f64 * cell,
        min_y: ORIGIN_Y - py1 as f64 * cell,
    }
}

struct ChunkPin(u32);

impl Transform for ChunkPin {
    fn identity(&self) -> String {
        format!("ChunkPin {}", self.0)
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Raster(RasterPattern {
            chunk_px: SetField::one(self.0),
            ..RasterPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        Ok(Chunk::Raster(input.raster()?.crop_to(&out.bbox)))
    }
}

fn engine_over(
    src: RasterSrc,
    chunk_px: Option<u32>,
    element: Box<dyn Transform>,
) -> (Engine, NodeId) {
    let mut g = Graph::new();
    let mut node = g.add_source(Box::new(src));
    if let Some(chunk_px) = chunk_px {
        node = g.add_transform(node, Box::new(ChunkPin(chunk_px)));
    }
    let node = g.add_transform(node, element);
    let engine = Engine::new(g, 64 << 20).unwrap();
    if let Some(chunk_px) = chunk_px {
        assert_eq!(engine.grid(node).chunk_px, chunk_px);
    }
    (engine, node)
}

async fn pull(engine: &Engine, node: NodeId, bbox: Bbox, resolution: f64) -> Chunk {
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
}

fn lines_of(geometry: &FeatureGeometry) -> Vec<&[Coord]> {
    match geometry {
        FeatureGeometry::LineString(l) => vec![l.coords()],
        FeatureGeometry::MultiLineString(mls) => {
            mls.linestrings().iter().map(|l| l.coords()).collect()
        }
        other => panic!("contours gave a {other:?}"),
    }
}

fn polygon_area(geometry: &FeatureGeometry) -> f64 {
    match geometry {
        FeatureGeometry::Polygon(p) => p.area(),
        FeatureGeometry::MultiPolygon(mp) => mp.area(),
        other => panic!("polygonize gave a {other:?}"),
    }
}

fn tile(ix: i64, iy: i64) -> WindowReq {
    let grid = GridSpec {
        origin_x: ORIGIN_X,
        origin_y: ORIGIN_Y,
        base_resolution: CELL,
        chunk_px: 16,
    };
    WindowReq {
        bbox: grid.chunk_bbox(ChunkKey {
            level: 0,
            ix,
            iy,
            time: None,
        }),
        resolution: CELL,
        time: None,
    }
}

async fn computed(element: &dyn Transform, src: RasterSrc, out: WindowReq) -> VectorChunk {
    let mut g = Graph::new();
    let node = g.add_source(Box::new(src));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let input = engine.pull(node, element.plan(&out)).await.unwrap();
    element
        .compute(&out, &input)
        .unwrap()
        .into_vector()
        .unwrap()
}

struct Seam {
    first: WindowReq,
    second: WindowReq,
    at: f64,
    across: fn(&Coord) -> f64,
    along: fn(&Coord) -> f64,
}

// vertical seams along tile row 2 and horizontal seams down tile column 2
fn seams() -> Vec<Seam> {
    let mut seams = Vec::new();
    for ix in 0..5 {
        let (left, right) = (tile(ix, 2), tile(ix + 1, 2));
        assert_eq!(left.bbox.max_x.to_bits(), right.bbox.min_x.to_bits());
        seams.push(Seam {
            first: left,
            second: right,
            at: left.bbox.max_x,
            across: |c| c.x,
            along: |c| c.y,
        });
    }
    for iy in 0..4 {
        let (upper, lower) = (tile(2, iy), tile(2, iy + 1));
        assert_eq!(upper.bbox.min_y.to_bits(), lower.bbox.max_y.to_bits());
        seams.push(Seam {
            first: upper,
            second: lower,
            at: upper.bbox.min_y,
            across: |c| c.y,
            along: |c| c.x,
        });
    }
    seams
}

fn cone(column: usize, row: usize) -> f64 {
    500.0 - 4.0 * (column as f64 - 47.3).hypot(row as f64 - 38.6)
}

fn cone_source(hole: impl Fn(usize, usize) -> bool) -> RasterSrc {
    source(CELL, W, H, ValueKind::Continuous, |column, row| {
        if hole(column, row) {
            f64::NAN
        } else {
            cone(column, row)
        }
    })
}

fn cone_contours() -> Contours {
    Contours {
        interval: 10.0,
        base: 0.0,
    }
}

#[derive(Debug, PartialEq)]
struct Level {
    pieces: usize,
    closed: usize,
    length: f64,
}

fn levels(dissolved: &VectorChunk) -> BTreeMap<u64, Level> {
    dissolved
        .features
        .iter()
        .map(|f| {
            let lines = lines_of(&f.geometry);
            let closed = lines
                .iter()
                .filter(|l| {
                    let (first, last) = (l[0], l[l.len() - 1]);
                    l.len() > 2 && first.distance_to(&last) < CELL * 1e-9
                })
                .count();
            let length = lines
                .iter()
                .flat_map(|l| l.windows(2))
                .map(|pair| pair[0].distance_to(&pair[1]))
                .sum();
            (
                f.id,
                Level {
                    pieces: lines.len(),
                    closed,
                    length,
                },
            )
        })
        .collect()
}

async fn dissolved_levels(src: RasterSrc, chunk_px: Option<u32>) -> BTreeMap<u64, Level> {
    let (engine, node) = engine_over(src, chunk_px, Box::new(cone_contours()));
    let pulled = pull(&engine, node, window(CELL, 0, 0, W, H), CELL)
        .await
        .into_vector()
        .unwrap();
    levels(&pulled.dissolve())
}

fn assert_same_levels(whole: &BTreeMap<u64, Level>, chunked: &BTreeMap<u64, Level>) {
    assert_eq!(
        whole.keys().collect::<Vec<_>>(),
        chunked.keys().collect::<Vec<_>>()
    );
    for (id, level) in whole {
        let other = &chunked[id];
        assert_eq!(
            (level.pieces, level.closed),
            (other.pieces, other.closed),
            "level id {id:x}: whole {level:?}, chunked {other:?}"
        );
        assert!(
            (level.length - other.length).abs() <= 1e-9 * level.length,
            "level id {id:x}: whole {level:?}, chunked {other:?}"
        );
    }
}

#[tokio::test]
async fn plane_contours_cross_at_the_exact_cell_edge_and_join_across_seams() {
    // binary exact, so every crossing has one exact f64 to land on
    let cell = 1.0 / 1024.0;
    let (columns, rows) = (64, 48);
    let bbox = window(cell, 0, 0, columns, rows);
    for chunk_px in [None, Some(16)] {
        // (z - base) / interval is column / 2 + 1 / 4, level k on the edge of columns 2k - 1 and 2k
        let plane = source(cell, columns, rows, ValueKind::Continuous, |column, _| {
            10.0 * column as f64 + 5.0
        });
        let contours = Contours {
            interval: 20.0,
            base: 0.0,
        };
        let (engine, node) = engine_over(plane, chunk_px, Box::new(contours));
        let dissolved = pull(&engine, node, bbox, cell)
            .await
            .into_vector()
            .unwrap()
            .dissolve();
        assert_eq!(dissolved.features.len(), 31, "chunk {chunk_px:?}");
        for feature in &dissolved.features {
            let elevation = feature.properties["elevation"].as_f64().unwrap();
            let edge = ORIGIN_X + 2.0 * (elevation / 20.0) * cell;
            let lines = lines_of(&feature.geometry);
            assert_eq!(
                lines.len(),
                1,
                "level {elevation} at chunk {chunk_px:?} came back in pieces"
            );
            let line = lines[0];
            for c in line {
                assert_eq!(
                    c.x, edge,
                    "level {elevation} at chunk {chunk_px:?} strays off its edge"
                );
            }
            let top = line.iter().map(|c| c.y).fold(f64::NEG_INFINITY, f64::max);
            let bottom = line.iter().map(|c| c.y).fold(f64::INFINITY, f64::min);
            assert_eq!(top, ORIGIN_Y - 0.5 * cell, "level {elevation}");
            assert_eq!(
                bottom,
                ORIGIN_Y - (rows as f64 - 0.5) * cell,
                "level {elevation}"
            );
        }
    }
}

#[tokio::test]
async fn neighbouring_tiles_cut_a_contour_at_the_same_seam_point() {
    let contours = cone_contours();
    let mut crossings = 0;
    for seam in seams() {
        let ends = |chunk: &VectorChunk| {
            let mut ends: Vec<(u64, u64)> = chunk
                .features
                .iter()
                .flat_map(|f| lines_of(&f.geometry))
                .flat_map(|line| [line[0], line[line.len() - 1]])
                .filter(|c| ((seam.across)(c) - seam.at).abs() < CELL * 1e-6)
                .map(|c| (c.x.to_bits(), c.y.to_bits()))
                .collect();
            ends.sort_unstable();
            ends
        };
        let from_first = ends(&computed(&contours, cone_source(|_, _| false), seam.first).await);
        let from_second = ends(&computed(&contours, cone_source(|_, _| false), seam.second).await);
        assert_eq!(from_first, from_second, "seam at {}", seam.at);
        crossings += from_first.len();
    }
    assert!(crossings >= 10, "only {crossings} contours cross the seams");
}

#[tokio::test]
async fn cone_contours_dissolve_to_the_same_lines_at_two_chunk_sizes() {
    let whole = dissolved_levels(cone_source(|_, _| false), None).await;
    let chunked = dissolved_levels(cone_source(|_, _| false), Some(16)).await;
    assert_same_levels(&whole, &chunked);
    let rings = whole
        .values()
        .filter(|l| l.pieces == 1 && l.closed == 1)
        .count();
    assert!(rings >= 5, "only {rings} levels close into one ring");
}

#[tokio::test]
async fn a_contour_along_a_seam_that_turns_into_the_west_tile_appears_once() {
    // level 105 runs on the column 16 seam for rows 0 to 2, then one column west
    let step = |column: usize, row: usize| {
        let east_from = if row <= 2 { 16 } else { 15 };
        if column >= east_from { 110.0 } else { 100.0 }
    };
    let mut dissolved = Vec::new();
    for chunk_px in [None, Some(16)] {
        let contours = Contours {
            interval: 10.0,
            base: 5.0,
        };
        let (engine, node) = engine_over(
            source(CELL, W, H, ValueKind::Continuous, step),
            chunk_px,
            Box::new(contours),
        );
        let pulled = pull(&engine, node, window(CELL, 0, 0, W, H), CELL)
            .await
            .into_vector()
            .unwrap();
        dissolved.push(levels(&pulled.dissolve()));
    }
    assert_eq!(dissolved[0].len(), 1, "{:?}", dissolved[0]);
    assert_same_levels(&dissolved[0], &dissolved[1]);
}

#[tokio::test]
async fn contours_stop_at_nodata_and_never_cross_a_hole() {
    let in_hole = |column: usize, row: usize| (40..52).contains(&column) && (30..40).contains(&row);
    let whole = dissolved_levels(cone_source(in_hole), None).await;
    let chunked = dissolved_levels(cone_source(in_hole), Some(16)).await;
    assert_same_levels(&whole, &chunked);

    let (engine, node) = engine_over(cone_source(in_hole), Some(16), Box::new(cone_contours()));
    let dissolved = pull(&engine, node, window(CELL, 0, 0, W, H), CELL)
        .await
        .into_vector()
        .unwrap()
        .dissolve();
    // squares with a corner in the hole cover the hole widened by half a cell
    let margin = CELL * 1e-6;
    let skipped = Bbox {
        min_x: ORIGIN_X + 39.5 * CELL,
        max_x: ORIGIN_X + 52.5 * CELL,
        max_y: ORIGIN_Y - 29.5 * CELL,
        min_y: ORIGIN_Y - 40.5 * CELL,
    };
    let lines: Vec<&[Coord]> = dissolved
        .features
        .iter()
        .flat_map(|f| lines_of(&f.geometry))
        .collect();
    for c in lines.iter().flat_map(|l| l.iter()) {
        assert!(c.x.is_finite() && c.y.is_finite());
        let inside = c.x > skipped.min_x + margin
            && c.x < skipped.max_x - margin
            && c.y > skipped.min_y + margin
            && c.y < skipped.max_y - margin;
        assert!(
            !inside,
            "a vertex at {c:?} sits on a square with a nodata corner"
        );
    }
    let on_rim = |c: &Coord| {
        c.x >= skipped.min_x - margin
            && c.x <= skipped.max_x + margin
            && c.y >= skipped.min_y - margin
            && c.y <= skipped.max_y + margin
    };
    let ends_at_hole = lines
        .iter()
        .flat_map(|l| [l[0], l[l.len() - 1]])
        .filter(on_rim)
        .count();
    assert!(
        ends_at_hole >= 4,
        "only {ends_at_hole} line ends meet the hole"
    );
}

const CODES: [f64; 4] = [1.0, 2.0, 3.0, 7.0];

fn class_at(column: usize, row: usize) -> f64 {
    if column < 6 && row < 5 {
        f64::NAN
    } else {
        CODES[(column / 11 + row / 7 + usize::from(column % 13 == 4)) % CODES.len()]
    }
}

fn class_source(value_kind: ValueKind) -> RasterSrc {
    source(CELL, W, H, value_kind, class_at)
}

#[tokio::test]
async fn polygonized_classes_cover_exactly_their_cells() {
    let mut cells: BTreeMap<i64, usize> = BTreeMap::new();
    for row in 0..H {
        for column in 0..W {
            let code = class_at(column, row);
            if !code.is_nan() {
                *cells.entry(code as i64).or_default() += 1;
            }
        }
    }
    for chunk_px in [None, Some(16)] {
        let (engine, node) = engine_over(
            class_source(ValueKind::Categorical),
            chunk_px,
            Box::new(Polygonize),
        );
        let dissolved = pull(&engine, node, window(CELL, 0, 0, W, H), CELL)
            .await
            .into_vector()
            .unwrap()
            .dissolve();
        let areas: BTreeMap<i64, f64> = dissolved
            .features
            .iter()
            .map(|f| {
                let code = f.properties["value"].as_i64().unwrap();
                (code, polygon_area(&f.geometry))
            })
            .collect();
        assert_eq!(
            areas.keys().collect::<Vec<_>>(),
            cells.keys().collect::<Vec<_>>(),
            "chunk {chunk_px:?}"
        );
        assert_eq!(
            dissolved.features.len(),
            cells.len(),
            "one feature per class"
        );
        for (code, count) in &cells {
            let expected = *count as f64 * CELL * CELL;
            // topoi's union behind dissolve snaps vertices
            assert!(
                (areas[code] - expected).abs() <= 1e-7 * expected,
                "class {code} at chunk {chunk_px:?}: area {} against {count} cells",
                areas[code]
            );
        }
    }
}

#[tokio::test]
async fn polygon_edges_on_a_seam_are_the_same_f64_in_both_tiles() {
    let mut shared = 0;
    for seam in seams() {
        let seam_vertices = |chunk: &VectorChunk| {
            let mut along_seam = Vec::new();
            for feature in &chunk.features {
                let FeatureGeometry::MultiPolygon(mp) = &feature.geometry else {
                    panic!("polygonize gave a {:?}", feature.geometry);
                };
                let rings = mp
                    .polygons()
                    .iter()
                    .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()));
                for c in rings.flat_map(|r| r.coords()) {
                    if ((seam.across)(c) - seam.at).abs() < CELL / 2.0 {
                        assert_eq!(
                            (seam.across)(c).to_bits(),
                            seam.at.to_bits(),
                            "a vertex beside the seam at {} missed it",
                            seam.at
                        );
                        along_seam.push((seam.along)(c).to_bits());
                    }
                }
            }
            along_seam.sort_unstable();
            along_seam.dedup();
            along_seam
        };
        let categorical = || class_source(ValueKind::Categorical);
        let from_first = seam_vertices(&computed(&Polygonize, categorical(), seam.first).await);
        let from_second = seam_vertices(&computed(&Polygonize, categorical(), seam.second).await);
        // both tiles put a corner where a class boundary meets the seam
        shared += from_first
            .iter()
            .filter(|bits| from_second.contains(*bits))
            .count();
    }
    assert!(
        shared >= 9,
        "only {shared} seam corners appear on both sides"
    );
}

#[test]
fn polygonize_refuses_a_continuous_raster_naming_value_kind() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(class_source(ValueKind::Continuous)));
    g.add_transform(src, Box::new(Polygonize));
    let err = Engine::new(g, 64 << 20)
        .err()
        .expect("polygonize solved over a continuous raster");
    assert!(
        err.to_string().contains("no common value for value_kinds"),
        "{err}"
    );
}

#[tokio::test]
async fn polygonize_refuses_a_fractional_class_code() {
    let fractional = source(CELL, 8, 8, ValueKind::Categorical, |_, _| 2.5);
    let (engine, node) = engine_over(fractional, None, Box::new(Polygonize));
    let err = engine
        .pull(
            node,
            WindowReq {
                bbox: window(CELL, 0, 0, 8, 8),
                resolution: CELL,
                time: None,
            },
        )
        .await
        .expect_err("polygonize took a class code of 2.5");
    assert!(err.to_string().contains("integer class codes"), "{err}");
}

const WALL_COLUMN: usize = 30;
const OBSERVER: (usize, usize) = (20, 24);
const RADIUS_CELLS: f64 = 20.0;

fn wall_source() -> RasterSrc {
    source(CELL, 64, 48, ValueKind::Continuous, |column, _| {
        if column == WALL_COLUMN { 400.0 } else { 0.0 }
    })
}

fn wall_viewshed() -> Viewshed {
    Viewshed {
        observer_x: ORIGIN_X + (OBSERVER.0 as f64 + 0.5) * CELL,
        observer_y: ORIGIN_Y - (OBSERVER.1 as f64 + 0.5) * CELL,
        observer_height: 2.0,
        radius: RADIUS_CELLS * CELL,
    }
}

#[tokio::test]
async fn viewshed_hides_what_the_wall_hides_pulled_whole_or_chunked() {
    let bbox = window(CELL, 0, 0, 64, 48);
    let mut pulled: Vec<RasterChunk> = Vec::new();
    for chunk_px in [None, Some(16)] {
        let (engine, node) = engine_over(wall_source(), chunk_px, Box::new(wall_viewshed()));
        assert_eq!(
            engine.caps(node).raster().value_kinds,
            BandKinds::from(ValueKind::Categorical)
        );
        pulled.push(pull(&engine, node, bbox, CELL).await.into_raster().unwrap());
    }
    let values = |chunk: &RasterChunk| chunk.bands.band(0).unwrap().data().to_vec();
    let (whole, chunked) = (values(&pulled[0]), values(&pulled[1]));
    assert_eq!(whole.len(), 64 * 48);
    for (i, (a, b)) in whole.iter().zip(&chunked).enumerate() {
        assert!(
            a == b || (a.is_nan() && b.is_nan()),
            "cell {i}: whole {a}, chunked {b}"
        );
    }

    let (mut visible, mut hidden) = (0, 0);
    for row in 0..48 {
        for column in 0..64 {
            let value = whole[row * 64 + column];
            let distance =
                (column as f64 - OBSERVER.0 as f64).hypot(row as f64 - OBSERVER.1 as f64);
            if (distance - RADIUS_CELLS).abs() < 0.5 {
                continue;
            }
            let at = format!("row {row} column {column}, value {value}");
            if distance > RADIUS_CELLS {
                assert!(value.is_nan(), "{at} lies past the radius");
            } else if column <= WALL_COLUMN {
                assert_eq!(value, 1.0, "{at} is in front of the wall or on it");
                visible += 1;
            } else {
                assert_eq!(value, 0.0, "{at} is behind the wall");
                hidden += 1;
            }
        }
    }
    assert!(
        visible > 100 && hidden > 50,
        "visible {visible}, hidden {hidden}"
    );
}

#[test]
fn viewshed_spreads_a_dirty_cell_in_its_disc_to_the_whole_disc() {
    let viewshed = wall_viewshed();
    let radius = RADIUS_CELLS * CELL;
    let disc = Bbox {
        min_x: viewshed.observer_x - radius,
        max_x: viewshed.observer_x + radius,
        min_y: viewshed.observer_y - radius,
        max_y: viewshed.observer_y + radius,
    }
    .expand(CELL);
    let inside = window(CELL, 25, 24, 26, 25);
    assert_eq!(viewshed.spread(&inside, CELL), disc);
    let outside = window(CELL, 60, 0, 61, 1);
    assert_eq!(viewshed.spread(&outside, CELL), outside);
}

struct CountingSrc {
    inner: RasterSrc,
    reads: Arc<AtomicUsize>,
}

impl Source for CountingSrc {
    fn identity(&self) -> String {
        format!("CountingSrc {}", self.inner.identity())
    }

    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.inner.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read(req)
    }
}

#[tokio::test]
async fn a_viewshed_tile_clear_of_the_disc_reads_no_input() {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let src = g.add_source(Box::new(CountingSrc {
        inner: wall_source(),
        reads: Arc::clone(&reads),
    }));
    let pinned = g.add_transform(src, Box::new(ChunkPin(16)));
    let node = g.add_transform(pinned, Box::new(wall_viewshed()));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.grid(node).chunk_px, 16);

    // the disc spans columns 0 to 41, so the 16 px chunk at columns 48 to 64 lies clear of it
    let clear = pull(&engine, node, window(CELL, 48, 0, 64, 16), CELL)
        .await
        .into_raster()
        .unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        0,
        "a tile clear of the disc read its input"
    );
    let values = clear.bands.band(0).unwrap().data();
    assert_eq!(values.len(), 16 * 16);
    assert!(values.iter().all(|value| value.is_nan()), "{values:?}");

    pull(&engine, node, window(CELL, 16, 16, 32, 32), CELL).await;
    assert!(
        reads.load(Ordering::SeqCst) > 0,
        "a tile in the disc read no input either, so the count shows nothing"
    );
}
