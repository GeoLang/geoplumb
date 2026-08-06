//! window-local vector to vector elements plus the dissolve at the pull
//! boundary: fragments merged back per feature, filter, schema map and clip
//! semantics with the chunked-equals-whole invariant, and the vector
//! reproject adapter spliced by the solver

use geoplumb::caps::{CapsPattern, CapsSet, Constraint, SetField, VectorCaps, VectorPattern};
use geoplumb::element::Transform;
use geoplumb::elements::{VecClip, VecFilter, VecReproject, VecSchema, VecSrc};
use geoplumb::{Bbox, Caps, Chunk, Crs, Engine, Graph, VectorChunk, WindowReq};
use serde_json::json;
use std::collections::HashMap;
use topoi_core::geojson::{Feature, FeatureCollection, FeatureGeometry};
use topoi_core::{Coord, LineString, MultiPoint, MultiPolygon, Point, Polygon, Ring};

/// vertices one unit apart along each edge, axis-aligned edges only. every
/// coordinate stays an integer, so the source's base resolution is exactly
/// 1.0 and no clip or stitch step can round
fn along(corners: &[(f64, f64)]) -> Vec<Coord> {
    let mut out = Vec::new();
    for w in corners.windows(2) {
        let (a, b) = (w[0], w[1]);
        // signum of zero is one, so an axis with no travel needs its own case
        let unit = |d: f64| if d == 0.0 { 0.0 } else { d.signum() };
        let (dx, dy) = (unit(b.0 - a.0), unit(b.1 - a.1));
        let steps = ((b.0 - a.0).abs() + (b.1 - a.1).abs()).round() as usize;
        for s in 0..steps {
            out.push(Coord::new(a.0 + dx * s as f64, a.1 + dy * s as f64));
        }
    }
    let last = *corners.last().expect("corners");
    out.push(Coord::new(last.0, last.1));
    out
}

fn rect(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Ring {
    Ring::new(along(&[
        (min_x, min_y),
        (max_x, min_y),
        (max_x, max_y),
        (min_x, max_y),
        (min_x, min_y),
    ]))
}

fn props(pairs: &[(&str, serde_json::Value)]) -> HashMap<String, serde_json::Value> {
    pairs
        .iter()
        .map(|(k, v)| ((*k).to_string(), v.clone()))
        .collect()
}

/// the grid is anchored at (0, 32) with 8-unit tiles, so the zone polygon
/// covers 3x3 tiles, its hole straddles the x=8 and y=16 seams, the road
/// crosses x=8 at one of its own vertices, the scatter lands in three tiles
/// and the patch sits inside a single tile
fn collection() -> FeatureCollection {
    let zone = Polygon::new(
        rect(0.0, 8.0, 24.0, 32.0),
        vec![rect(6.0, 14.0, 10.0, 18.0)],
    );
    let road = LineString::new(along(&[(2.0, 12.0), (14.0, 12.0)]));
    let scatter = MultiPoint::new(vec![
        Point::new(1.5, 30.5),
        Point::new(9.5, 30.5),
        Point::new(17.5, 10.5),
    ]);
    let patch = Polygon::new(rect(17.0, 2.0, 22.0, 6.0), vec![]);
    FeatureCollection {
        features: vec![
            feature(
                FeatureGeometry::Polygon(zone),
                props(&[("kind", json!("zone")), ("v", json!(3))]),
            ),
            feature(
                FeatureGeometry::LineString(road),
                props(&[("kind", json!("road")), ("v", json!(9))]),
            ),
            feature(
                FeatureGeometry::MultiPoint(scatter),
                props(&[("kind", json!("poi")), ("v", json!(5))]),
            ),
            feature(
                FeatureGeometry::Polygon(patch),
                props(&[("kind", json!("zone")), ("v", json!(1))]),
            ),
        ],
    }
}

fn feature(geometry: FeatureGeometry, properties: HashMap<String, serde_json::Value>) -> Feature {
    Feature {
        geometry: Some(geometry),
        properties,
    }
}

fn src() -> VecSrc {
    VecSrc::new(collection(), Crs::WGS84).unwrap()
}

const WINDOW: Bbox = Bbox {
    min_x: 0.0,
    min_y: 0.0,
    max_x: 24.0,
    max_y: 32.0,
};

/// vector identity that narrows the negotiated chunk size, so the tiles are
/// small enough for features to straddle seams
struct VecChunks(u32);

impl Transform for VecChunks {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Vector(VectorPattern {
            chunk_px: SetField::one(self.0),
            ..VectorPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        Ok(Chunk::Vector(input.vector()?.crop_to(&out.bbox)))
    }
}

fn coords_of(geometry: &FeatureGeometry) -> Vec<Coord> {
    match geometry {
        FeatureGeometry::Point(p) => vec![p.0],
        FeatureGeometry::MultiPoint(mp) => mp.points().iter().map(|p| p.0).collect(),
        FeatureGeometry::LineString(l) => l.coords().to_vec(),
        FeatureGeometry::MultiLineString(mls) => mls
            .linestrings()
            .iter()
            .flat_map(|l| l.coords().iter().copied())
            .collect(),
        FeatureGeometry::Polygon(p) => polygon_coords(p),
        FeatureGeometry::MultiPolygon(mp) => {
            mp.polygons().iter().flat_map(polygon_coords).collect()
        }
        FeatureGeometry::GeometryCollection(members) => {
            members.iter().flat_map(coords_of).collect()
        }
    }
}

fn polygon_coords(p: &Polygon) -> Vec<Coord> {
    let mut coords = p.exterior().coords().to_vec();
    for hole in p.interiors() {
        coords.extend_from_slice(hole.coords());
    }
    coords
}

/// id, geometry kind and exact coordinate bits, the comparison key for the
/// chunked-equals-whole checks
type Fingerprint = (u64, &'static str, Vec<(u64, u64)>);

fn fingerprint(chunk: &VectorChunk) -> Vec<Fingerprint> {
    chunk
        .features
        .iter()
        .map(|f| {
            let kind = match &f.geometry {
                FeatureGeometry::Point(_) => "point",
                FeatureGeometry::MultiPoint(_) => "multipoint",
                FeatureGeometry::LineString(_) => "linestring",
                FeatureGeometry::MultiLineString(_) => "multilinestring",
                FeatureGeometry::Polygon(_) => "polygon",
                FeatureGeometry::MultiPolygon(_) => "multipolygon",
                FeatureGeometry::GeometryCollection(_) => "collection",
            };
            let bits = coords_of(&f.geometry)
                .iter()
                .map(|c| (c.x.to_bits(), c.y.to_bits()))
                .collect();
            (f.id, kind, bits)
        })
        .collect()
}

fn polygon_area(geometry: &FeatureGeometry) -> f64 {
    match geometry {
        FeatureGeometry::Polygon(p) => p.area(),
        FeatureGeometry::MultiPolygon(mp) => mp.area(),
        _ => 0.0,
    }
}

/// pull the whole window at base resolution off a graph whose vector link
/// is tiled at `chunk_px`
async fn pull_window(chunk_px: u32) -> VectorChunk {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(src()));
    g.add_transform(vec, Box::new(VecChunks(chunk_px)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.grid(vec).base_resolution, 1.0);
    engine
        .pull(
            vec,
            WindowReq {
                bbox: WINDOW,
                resolution: 1.0,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_vector()
        .unwrap()
}

/// an anchor polygon fixing the grid at (0, 32) plus three lines that lie
/// exactly on a seam: one on the interior x=8 seam, one along the
/// envelope's east edge, one along its south edge
fn seam_collection() -> FeatureCollection {
    FeatureCollection {
        features: vec![
            feature(
                FeatureGeometry::Polygon(Polygon::new(rect(0.0, 8.0, 16.0, 32.0), vec![])),
                props(&[("kind", json!("anchor"))]),
            ),
            feature(
                FeatureGeometry::LineString(LineString::new(along(&[(8.0, 12.0), (8.0, 28.0)]))),
                props(&[("kind", json!("seam"))]),
            ),
            feature(
                FeatureGeometry::LineString(LineString::new(along(&[(16.0, 12.0), (16.0, 28.0)]))),
                props(&[("kind", json!("east"))]),
            ),
            feature(
                FeatureGeometry::LineString(LineString::new(along(&[(2.0, 8.0), (14.0, 8.0)]))),
                props(&[("kind", json!("south"))]),
            ),
        ],
    }
}

async fn seam_pull(window: Bbox) -> VectorChunk {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(
        VecSrc::new(seam_collection(), Crs::WGS84).unwrap(),
    ));
    g.add_transform(vec, Box::new(VecChunks(8)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.grid(vec).base_resolution, 1.0);
    assert_eq!(engine.grid(vec).origin_x, 0.0);
    assert_eq!(engine.grid(vec).origin_y, 32.0);
    engine
        .pull(
            vec,
            WindowReq {
                bbox: window,
                resolution: 1.0,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_vector()
        .unwrap()
}

fn one_line(chunk: &VectorChunk, id: u64) -> Vec<Coord> {
    let merged = chunk.dissolve();
    let feature = merged
        .features
        .iter()
        .find(|f| f.id == id)
        .unwrap_or_else(|| panic!("feature {id} is missing from the pull"));
    match &feature.geometry {
        FeatureGeometry::LineString(l) => l.coords().to_vec(),
        other => panic!("feature {id} came back as {other:?}, not one linestring"),
    }
}

/// a line lying along a seam is inside both neighbouring tiles as far as
/// rect clipping is concerned, so without the membership rule it comes back
/// doubled: two identical parts the stitch cannot join head to tail
#[tokio::test]
async fn a_line_on_an_interior_seam_is_not_duplicated() {
    let pulled = seam_pull(Bbox::new(0.0, 8.0, 16.0, 32.0)).await;
    assert_eq!(
        pulled.features.iter().filter(|f| f.id == 1).count(),
        3,
        "one fragment per tile row, not one per row and column"
    );
    assert_eq!(one_line(&pulled, 1), along(&[(8.0, 12.0), (8.0, 28.0)]));
}

/// the geodukt driver widens a full-extent window past max_x and min_y to
/// take in the features on those edges, which must not double them
#[tokio::test]
async fn a_line_on_a_widened_window_edge_appears_once() {
    let east = seam_pull(Bbox::new(0.0, 8.0, 17.0, 32.0)).await;
    assert_eq!(
        east.features.iter().filter(|f| f.id == 2).count(),
        3,
        "the east edge line belongs to the tiles starting at x=16"
    );
    assert_eq!(one_line(&east, 2), along(&[(16.0, 12.0), (16.0, 28.0)]));

    let south = seam_pull(Bbox::new(0.0, 7.0, 17.0, 32.0)).await;
    assert_eq!(
        south.features.iter().filter(|f| f.id == 3).count(),
        2,
        "the south edge line belongs to the tile row ending at y=8"
    );
    assert_eq!(one_line(&south, 3), along(&[(2.0, 8.0), (14.0, 8.0)]));
}

/// same rule seen from the window: a line on the excluded edge belongs to
/// the neighbouring tile, exactly as a point there does
#[tokio::test]
async fn a_line_on_the_windows_excluded_edge_is_absent() {
    let clipped_at_the_seam = seam_pull(Bbox::new(0.0, 8.0, 8.0, 32.0)).await;
    assert!(
        !clipped_at_the_seam.features.iter().any(|f| f.id == 1),
        "the line on the window's max_x edge is the neighbour's"
    );
    let clipped_at_the_south = seam_pull(Bbox::new(0.0, 8.0, 17.0, 32.0)).await;
    assert!(
        !clipped_at_the_south.features.iter().any(|f| f.id == 3),
        "the line on the window's min_y edge is the neighbour's"
    );
}

#[tokio::test]
async fn dissolve_merges_the_fragments_of_each_feature() {
    let fragments = pull_window(8).await;
    let count = |id: u64| fragments.features.iter().filter(|f| f.id == id).count();
    assert!(count(0) > 1, "the zone polygon must arrive split");
    assert!(count(1) > 1, "the road must arrive split");
    assert!(count(2) > 1, "the scatter must arrive split");
    assert_eq!(count(3), 1, "the patch fits in one tile");

    let merged = fragments.dissolve();
    let ids: Vec<u64> = merged.features.iter().map(|f| f.id).collect();
    assert_eq!(ids, vec![0, 1, 2, 3], "one feature per id, ordered by id");
    assert_eq!(merged.bbox, fragments.bbox);
    assert_eq!(merged.resolution, fragments.resolution);

    let zone = &merged.features[0];
    assert_eq!(
        zone.properties.get("kind"),
        Some(&json!("zone")),
        "properties come from the first fragment"
    );
    let FeatureGeometry::Polygon(zone_poly) = &zone.geometry else {
        panic!(
            "the zone merges back to one polygon, got {:?}",
            zone.geometry
        );
    };
    // 24 x 24 outer minus the 4 x 4 hole
    assert!(
        (zone_poly.area() - 560.0).abs() < 1e-9,
        "merged area {}",
        zone_poly.area()
    );
    assert_eq!(
        zone_poly.interiors().len(),
        1,
        "the hole survives the seams it straddles"
    );
    assert!((zone_poly.interiors()[0].area() - 16.0).abs() < 1e-9);

    let FeatureGeometry::LineString(road) = &merged.features[1].geometry else {
        panic!("the road stitches back to one linestring");
    };
    assert_eq!(
        road.coords(),
        along(&[(2.0, 12.0), (14.0, 12.0)]),
        "stitching restores the source vertices"
    );

    let FeatureGeometry::MultiPoint(scatter) = &merged.features[2].geometry else {
        panic!("the scatter merges back to one multipoint");
    };
    let mut got: Vec<(u64, u64)> = scatter
        .points()
        .iter()
        .map(|p| (p.0.x.to_bits(), p.0.y.to_bits()))
        .collect();
    got.sort();
    let mut want: Vec<(u64, u64)> = [(1.5, 30.5), (9.5, 30.5), (17.5, 10.5)]
        .iter()
        .map(|(x, y)| (f64::to_bits(*x), f64::to_bits(*y)))
        .collect();
    want.sort();
    assert_eq!(got, want, "every member is kept exactly once");
}

/// what the stitch rests on off the integer grid: adjacent tiles cut a
/// crossing segment at the same coordinate, because Liang-Barsky solves the
/// same ratio from either side. re-clipping in assembly rebuilds a kept
/// endpoint as start plus (end - start), which is why the match is
/// ulp-scale rather than exact
#[test]
fn adjacent_tiles_cut_a_seam_crossing_at_the_same_point() {
    let line = FeatureGeometry::LineString(LineString::new(vec![
        Coord::new(0.3, 5.7),
        Coord::new(9.13, 11.27),
        Coord::new(15.77, 3.31),
    ]));
    let window = Bbox::new(0.0, 0.0, 16.0, 16.0);
    let tile = |min_x: f64| {
        let cut = geoplumb::chunk::clip_geometry(&line, &Bbox::new(min_x, 0.0, min_x + 8.0, 16.0));
        assert_eq!(cut.len(), 1, "one piece per tile");
        // assembly re-clips every fragment to the pulled window
        geoplumb::chunk::clip_geometry(&cut[0], &window)
            .pop()
            .expect("the piece is inside the window")
    };
    let (left, right) = (tile(0.0), tile(8.0));
    let (left_coords, right_coords) = (coords_of(&left), coords_of(&right));
    let close = |a: &Coord, b: &Coord| {
        (a.x - b.x).abs() <= 8.0 * f64::EPSILON * a.x.abs().max(1.0)
            && (a.y - b.y).abs() <= 8.0 * f64::EPSILON * a.y.abs().max(1.0)
    };
    let (seam_left, seam_right) = (left_coords.last().unwrap(), right_coords.first().unwrap());
    assert!(
        close(seam_left, seam_right),
        "the tiles cut the seam at {seam_left:?} and {seam_right:?}"
    );
    assert!(seam_left.x == 8.0 || seam_right.x == 8.0, "cut on the seam");

    let feature = |geometry| geoplumb::VectorFeature {
        id: 0,
        geometry,
        properties: HashMap::new(),
    };
    let merged =
        VectorChunk::new(vec![feature(right), feature(left)], window, 1.0, Crs::WGS84).dissolve();
    assert_eq!(merged.features.len(), 1, "the halves stitch back together");
    let mut want = left_coords.clone();
    want.extend_from_slice(&right_coords[1..]);
    assert_eq!(coords_of(&merged.features[0].geometry), want);
    assert!(close(want.first().unwrap(), &Coord::new(0.3, 5.7)));
    assert!(close(want.last().unwrap(), &Coord::new(15.77, 3.31)));
}

#[tokio::test]
async fn dissolve_passes_an_unsplit_feature_through_untouched() {
    let merged = pull_window(8).await.dissolve();
    let patch = merged.features.iter().find(|f| f.id == 3).unwrap();
    let FeatureGeometry::Polygon(p) = &patch.geometry else {
        panic!("the patch stays a polygon");
    };
    assert_eq!(p.exterior().coords(), rect(17.0, 2.0, 22.0, 6.0).coords());
}

#[tokio::test]
async fn dissolve_of_an_untiled_pull_is_the_source_geometry() {
    // one tile over the whole window: nothing to merge, so this pins the
    // seam-split case above against the unsplit reference
    let whole = pull_window(64).await.dissolve();
    let tiled = pull_window(8).await.dissolve();
    let area = |c: &VectorChunk, id: u64| {
        polygon_area(&c.features.iter().find(|f| f.id == id).unwrap().geometry)
    };
    assert!((area(&whole, 0) - area(&tiled, 0)).abs() < 1e-9);
    assert!((area(&whole, 3) - area(&tiled, 3)).abs() < 1e-9);
}

/// apply an element to a whole-window chunk, the reference every chunked
/// pull is compared against
fn whole_window(element: &dyn Transform, chunk: &VectorChunk) -> VectorChunk {
    let req = WindowReq {
        bbox: chunk.bbox,
        resolution: chunk.resolution,
        time: None,
    };
    element
        .compute(&req, &Chunk::Vector(chunk.clone()))
        .unwrap()
        .into_vector()
        .unwrap()
}

async fn pull_through(element: Box<dyn Transform>) -> (VectorChunk, VectorChunk) {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(src()));
    let op = g.add_transform(vec, element);
    g.add_transform(op, Box::new(VecChunks(8)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let req = WindowReq {
        bbox: WINDOW,
        resolution: 1.0,
        time: None,
    };
    let chunked = engine.pull(op, req).await.unwrap().into_vector().unwrap();
    let source = engine.pull(vec, req).await.unwrap().into_vector().unwrap();
    (chunked, source)
}

#[tokio::test]
async fn filter_keeps_matching_features_chunk_by_chunk() {
    let (chunked, source) = pull_through(Box::new(VecFilter::new("kind", json!("zone")))).await;
    let ids: Vec<u64> = chunked.dissolve().features.iter().map(|f| f.id).collect();
    assert_eq!(ids, vec![0, 3], "only the zone-kind features survive");

    let reference = whole_window(&VecFilter::new("kind", json!("zone")), &source);
    assert_eq!(fingerprint(&chunked), fingerprint(&reference));
}

#[test]
fn filter_drops_missing_properties_and_type_mismatches() {
    let chunk = |properties: HashMap<String, serde_json::Value>| {
        VectorChunk::new(
            vec![geoplumb::VectorFeature {
                id: 0,
                geometry: FeatureGeometry::Point(Point::new(1.0, 1.0)),
                properties,
            }],
            WINDOW,
            1.0,
            Crs::WGS84,
        )
    };
    let kept = |filter: &VecFilter, properties: HashMap<String, serde_json::Value>| {
        whole_window(filter, &chunk(properties)).features.len()
    };
    let by_number = VecFilter::new("v", json!(3));
    assert_eq!(kept(&by_number, props(&[("v", json!(3))])), 1);
    assert_eq!(kept(&by_number, props(&[("v", json!(4))])), 0);
    assert_eq!(kept(&by_number, props(&[("other", json!(3))])), 0);
    assert_eq!(
        kept(&by_number, props(&[("v", json!("3"))])),
        0,
        "a string is not the integer three"
    );
    let by_float = VecFilter::new("v", json!(2.5));
    assert_eq!(kept(&by_float, props(&[("v", json!(2.5))])), 1);
    assert_eq!(kept(&by_float, props(&[("v", json!(2.5000001))])), 0);
    let by_flag = VecFilter::new("ok", json!(true));
    assert_eq!(kept(&by_flag, props(&[("ok", json!(true))])), 1);
    assert_eq!(kept(&by_flag, props(&[("ok", json!(false))])), 0);
}

fn schema() -> VecSchema {
    VecSchema {
        rename: HashMap::from([("kind".to_string(), "class".to_string())]),
        drop: vec!["v".to_string()],
        add: HashMap::from([("src".to_string(), json!("vec"))]),
    }
}

#[tokio::test]
async fn schema_renames_drops_and_adds_chunk_by_chunk() {
    let (chunked, source) = pull_through(Box::new(schema())).await;
    let merged = chunked.dissolve();
    let zone = merged.features.iter().find(|f| f.id == 0).unwrap();
    assert_eq!(zone.properties.get("class"), Some(&json!("zone")));
    assert!(!zone.properties.contains_key("kind"), "renamed away");
    assert!(!zone.properties.contains_key("v"), "dropped");
    assert_eq!(zone.properties.get("src"), Some(&json!("vec")));

    let reference = whole_window(&schema(), &source);
    assert_eq!(fingerprint(&chunked), fingerprint(&reference));
    assert!(
        (polygon_area(&zone.geometry) - 560.0).abs() < 1e-9,
        "geometry is untouched"
    );
}

fn boundary() -> MultiPolygon {
    MultiPolygon::new(vec![Polygon::new(rect(4.0, 12.0, 20.0, 28.0), vec![])])
}

#[tokio::test]
async fn clip_cuts_fragments_to_the_boundary_chunk_by_chunk() {
    let (chunked, source) = pull_through(Box::new(VecClip {
        boundary: boundary(),
    }))
    .await;
    let chunked_area: f64 = chunked
        .features
        .iter()
        .filter(|f| f.id == 0)
        .map(|f| polygon_area(&f.geometry))
        .sum();
    // the 16 x 16 boundary minus the 4 x 4 hole it fully contains
    assert!(
        (chunked_area - 240.0).abs() < 1e-6,
        "clipped area {chunked_area}"
    );

    let reference = whole_window(
        &VecClip {
            boundary: boundary(),
        },
        &source,
    );
    let whole_area: f64 = reference
        .features
        .iter()
        .filter(|f| f.id == 0)
        .map(|f| polygon_area(&f.geometry))
        .sum();
    assert!(
        (chunked_area - whole_area).abs() < 1e-6,
        "chunked {chunked_area} vs whole {whole_area}"
    );
    assert!(
        !chunked.features.iter().any(|f| f.id == 3),
        "the patch lies outside the boundary and is dropped"
    );
}

#[test]
fn clip_cuts_lines_and_drops_outside_points() {
    let features = vec![
        geoplumb::VectorFeature {
            id: 0,
            geometry: FeatureGeometry::LineString(LineString::new(along(&[
                (0.0, 20.0),
                (24.0, 20.0),
            ]))),
            properties: HashMap::new(),
        },
        geoplumb::VectorFeature {
            id: 1,
            geometry: FeatureGeometry::Point(Point::new(10.0, 20.0)),
            properties: HashMap::new(),
        },
        geoplumb::VectorFeature {
            id: 2,
            geometry: FeatureGeometry::Point(Point::new(1.0, 20.0)),
            properties: HashMap::new(),
        },
    ];
    let clipped = whole_window(
        &VecClip {
            boundary: boundary(),
        },
        &VectorChunk::new(features, WINDOW, 1.0, Crs::WGS84),
    );
    assert_eq!(
        clipped.features.iter().map(|f| f.id).collect::<Vec<_>>(),
        vec![0, 1],
        "the point outside the boundary is dropped"
    );
    let FeatureGeometry::LineString(line) = &clipped.features[0].geometry else {
        panic!("the line stays one piece");
    };
    assert_eq!(line.coords().first(), Some(&Coord::new(4.0, 20.0)));
    assert_eq!(line.coords().last(), Some(&Coord::new(20.0, 20.0)));
    assert!(
        line.coords()
            .iter()
            .all(|c| c.x >= 4.0 && c.x <= 20.0 && c.y == 20.0),
        "nothing outside the boundary survives"
    );
}

/// vector identity that insists on web mercator, the demand that makes the
/// solver splice a vector reproject
struct MercatorVec;

impl Transform for MercatorVec {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Vector(VectorPattern {
            crs: SetField::one(Crs::WEB_MERCATOR),
            ..VectorPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        Ok(Chunk::Vector(input.vector()?.crop_to(&out.bbox)))
    }
}

/// a square in wgs84 with quarter-degree vertex spacing, so the source's
/// base resolution is a binary-exact 0.25
fn lonlat_src() -> VecSrc {
    let ring = Ring::new(vec![
        Coord::new(7.0, 46.5),
        Coord::new(7.25, 46.5),
        Coord::new(7.5, 46.5),
        Coord::new(7.5, 46.75),
        Coord::new(7.5, 47.0),
        Coord::new(7.25, 47.0),
        Coord::new(7.0, 47.0),
        Coord::new(7.0, 46.75),
        Coord::new(7.0, 46.5),
    ]);
    VecSrc::new(
        FeatureCollection {
            features: vec![feature(
                FeatureGeometry::Polygon(Polygon::new(ring, vec![])),
                props(&[("kind", json!("zone"))]),
            )],
        },
        Crs::WGS84,
    )
    .unwrap()
}

#[tokio::test]
async fn a_vector_crs_demand_autoplugs_a_vector_reproject() {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(lonlat_src()));
    let merc = g.add_transform(vec, Box::new(MercatorVec));
    let burn = g.add_transform(
        merc,
        Box::new(geoplumb::elements::Rasterize {
            burn: geoplumb::elements::Burn::Constant(1.0),
        }),
    );
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert!(matches!(engine.caps(vec), Caps::Vector(_)));
    assert_eq!(
        engine.caps(vec).vector().crs,
        Crs::WGS84,
        "the source keeps its own crs"
    );
    assert_eq!(engine.caps(merc).vector().crs, Crs::WEB_MERCATOR);
    assert_eq!(
        engine.caps(burn).raster().crs,
        Crs::WEB_MERCATOR,
        "the burn takes the reprojected crs"
    );

    let fwd = projicio_core::Transform::new("EPSG:4326", "EPSG:3857").unwrap();
    let (x0, y1) = fwd.convert(7.0, 47.0).unwrap();
    let (x1, y0) = fwd.convert(7.5, 46.5).unwrap();
    let res = engine.grid(merc).base_resolution;
    let pulled = engine
        .pull(
            merc,
            WindowReq {
                bbox: Bbox::new(x0 - res, y0 - res, x1 + res, y1 + res),
                resolution: res,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_vector()
        .unwrap();
    assert_eq!(pulled.crs, Crs::WEB_MERCATOR);
    let fragments: Vec<Coord> = pulled
        .features
        .iter()
        .flat_map(|f| coords_of(&f.geometry))
        .collect();
    assert!(
        fragments
            .iter()
            .all(|c| c.x >= x0 - 1.0 && c.x <= x1 + 1.0 && c.y >= y0 - 1.0 && c.y <= y1 + 1.0),
        "the projected square lands inside its mercator envelope"
    );
    for (edge, want) in [("west", x0), ("east", x1)] {
        assert!(
            fragments.iter().any(|c| (c.x - want).abs() < 1e-9),
            "the {edge} edge is where projicio puts it"
        );
    }

    let merged = pulled.dissolve();
    assert_eq!(merged.features.len(), 1);
    // lon and lat map to x and y independently, so the square stays a
    // rectangle and its area is the projected side lengths. the union
    // behind dissolve snaps vertices, hence a relative tolerance
    let area = polygon_area(&merged.features[0].geometry);
    let want = (x1 - x0) * (y1 - y0);
    assert!((area / want - 1.0).abs() < 1e-7, "merged area {area}");
}

#[test]
fn vec_reproject_matches_projicio_vertex_by_vertex() {
    let source = Polygon::new(
        Ring::new(vec![
            Coord::new(7.0, 46.5),
            Coord::new(7.5, 46.5),
            Coord::new(7.5, 47.0),
            Coord::new(7.0, 47.0),
            Coord::new(7.0, 46.5),
        ]),
        vec![],
    );
    let mut element = VecReproject::new(Crs::WEB_MERCATOR);
    let caps = |crs| {
        Caps::Vector(VectorCaps {
            crs,
            resolution: geoplumb::caps::ResRange::ANY,
            chunk_px: 256,
        })
    };
    element
        .configure(&caps(Crs::WGS84), &caps(Crs::WEB_MERCATOR))
        .unwrap();

    let fwd = projicio_core::Transform::new("EPSG:4326", "EPSG:3857").unwrap();
    let (x0, y0) = fwd.convert(7.0, 46.5).unwrap();
    let (x1, y1) = fwd.convert(7.5, 47.0).unwrap();
    let out = WindowReq {
        bbox: Bbox::new(x0 - 1000.0, y0 - 1000.0, x1 + 1000.0, y1 + 1000.0),
        resolution: 100.0,
        time: None,
    };
    let input = Chunk::Vector(VectorChunk::new(
        vec![geoplumb::VectorFeature {
            id: 7,
            geometry: FeatureGeometry::Polygon(source.clone()),
            properties: props(&[("kind", json!("zone"))]),
        }],
        Bbox::new(7.0, 46.5, 7.5, 47.0),
        0.25,
        Crs::WGS84,
    ));
    let got = element
        .compute(&out, &input)
        .unwrap()
        .into_vector()
        .unwrap();
    assert_eq!(got.crs, Crs::WEB_MERCATOR);
    assert_eq!(got.features.len(), 1);
    assert_eq!(got.features[0].id, 7, "ids and properties ride along");
    assert_eq!(got.features[0].properties.get("kind"), Some(&json!("zone")));

    let want: Vec<Coord> = source
        .exterior()
        .coords()
        .iter()
        .map(|c| {
            let (x, y) = fwd.convert(c.x, c.y).unwrap();
            Coord::new(x, y)
        })
        .collect();
    let FeatureGeometry::Polygon(p) = &got.features[0].geometry else {
        panic!("a polygon in, a polygon out");
    };
    assert_eq!(p.exterior().coords(), want.as_slice());
}
