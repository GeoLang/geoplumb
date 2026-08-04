//! vector chunks end to end: negotiation across the kind boundary, per-level
//! simplification and sub-pixel drop, rasterize seam equality, spill reload

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use geoplumb::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use geoplumb::element::{Source, Transform};
use geoplumb::elements::{Burn, Hillshade, Rasterize, VecSrc};
use geoplumb::{Bbox, Caps, Chunk, Crs, Engine, Error, Graph, WindowReq};
use topoi_core::geojson::{Feature, FeatureCollection, FeatureGeometry};
use topoi_core::{Coord, LineString, MultiLineString, MultiPoint, Point, Polygon, Ring};

/// insert vertices every ~0.9 units so the median segment length, and with
/// it the source's base resolution, is well below the feature sizes
fn densify(corners: &[(f64, f64)]) -> Vec<Coord> {
    let mut out = Vec::new();
    for w in corners.windows(2) {
        let (a, b) = (Coord::new(w[0].0, w[0].1), Coord::new(w[1].0, w[1].1));
        let len = a.distance_to(&b);
        let steps = (len / 0.9).ceil().max(1.0) as usize;
        for s in 0..steps {
            let t = s as f64 / steps as f64;
            out.push(Coord::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)));
        }
    }
    let last = corners.last().unwrap();
    out.push(Coord::new(last.0, last.1));
    out
}

fn ring(corners: &[(f64, f64)]) -> Ring {
    let mut closed = corners.to_vec();
    closed.push(corners[0]);
    Ring::new(densify(&closed))
}

fn props(v: f64) -> HashMap<String, serde_json::Value> {
    HashMap::from([("v".to_string(), serde_json::json!(v))])
}

fn feature(geometry: FeatureGeometry, properties: HashMap<String, serde_json::Value>) -> Feature {
    Feature {
        geometry: Some(geometry),
        properties,
    }
}

/// polygon with a hole, a bent line, a point, a value-less polygon the burn
/// skips, a sub-pixel square that coarse levels drop, then one feature per
/// multi kind and a mixed collection
fn collection() -> FeatureCollection {
    let holed = Polygon::new(
        ring(&[(4.3, 4.2), (40.7, 4.2), (40.7, 30.6), (4.3, 30.6)]),
        vec![ring(&[
            (12.1, 10.4),
            (20.9, 10.4),
            (20.9, 18.2),
            (12.1, 18.2),
        ])],
    );
    let line = LineString::new(densify(&[(2.2, 50.3), (55.4, 34.7), (60.1, 60.2)]));
    let tiny = Polygon::new(
        Ring::new(vec![
            Coord::new(50.1, 10.1),
            Coord::new(50.5, 10.1),
            Coord::new(50.5, 10.5),
            Coord::new(50.1, 10.5),
            Coord::new(50.1, 10.1),
        ]),
        vec![],
    );
    let strands = MultiLineString::new(vec![
        LineString::new(densify(&[(5.3, 12.7), (30.1, 20.3)])),
        LineString::new(densify(&[(8.4, 38.6), (46.2, 41.1)])),
    ]);
    let scatter = MultiPoint::new(vec![Point::new(15.3, 25.7), Point::new(38.9, 51.3)]);
    let mixed = vec![
        FeatureGeometry::Polygon(Polygon::new(
            ring(&[(44.1, 12.3), (51.7, 12.3), (51.7, 19.9), (44.1, 19.9)]),
            vec![],
        )),
        FeatureGeometry::LineString(LineString::new(densify(&[(34.2, 24.6), (52.6, 30.4)]))),
    ];
    FeatureCollection {
        features: vec![
            feature(FeatureGeometry::Polygon(holed), props(3.0)),
            feature(FeatureGeometry::LineString(line), props(9.0)),
            feature(FeatureGeometry::Point(Point::new(33.5, 44.5)), props(5.0)),
            feature(
                FeatureGeometry::Polygon(Polygon::new(
                    ring(&[(45.2, 45.3), (52.8, 45.3), (52.8, 52.9), (45.2, 52.9)]),
                    vec![],
                )),
                HashMap::new(),
            ),
            feature(FeatureGeometry::Polygon(tiny), props(1.0)),
            feature(FeatureGeometry::MultiLineString(strands), props(7.0)),
            feature(FeatureGeometry::MultiPoint(scatter), props(6.0)),
            feature(FeatureGeometry::GeometryCollection(mixed), props(2.0)),
        ],
    }
}

fn src() -> VecSrc {
    VecSrc::new(collection(), Crs::WGS84).unwrap()
}

fn burn() -> Rasterize {
    Rasterize {
        burn: Burn::Property("v".to_string()),
    }
}

#[test]
fn vector_chain_negotiates_across_the_kind_boundary() {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(src()));
    let grid = g.add_transform(vec, Box::new(burn()));
    let hs = g.add_transform(grid, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let vector_caps = engine.caps(vec).vector();
    assert_eq!(vector_caps.crs, Crs::WGS84);
    let raster_caps = engine.caps(grid).raster();
    assert_eq!(raster_caps.crs, Crs::WGS84, "crs passes through the burn");
    assert_eq!(raster_caps.bands, 1);
    assert_eq!(engine.caps(hs).raster().crs, Crs::WGS84);
}

#[test]
fn vector_source_cannot_feed_a_raster_consumer_directly() {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(src()));
    g.add_transform(vec, Box::new(Hillshade::new(315.0, 45.0)));
    match Engine::new(g, 64 << 20) {
        Err(Error::EmptyLink { .. }) => {}
        Err(other) => panic!("expected EmptyLink, got {other:?}"),
        Ok(_) => panic!("kind mismatch must fail negotiation"),
    }
}

fn fragment_coord_count(chunk: &geoplumb::VectorChunk) -> usize {
    chunk
        .features
        .iter()
        .map(|f| geometry_coord_count(&f.geometry))
        .sum()
}

fn geometry_coord_count(geometry: &FeatureGeometry) -> usize {
    match geometry {
        FeatureGeometry::Point(_) => 1,
        FeatureGeometry::MultiPoint(mp) => mp.points().len(),
        FeatureGeometry::LineString(l) => l.coords().len(),
        FeatureGeometry::MultiLineString(mls) => {
            mls.linestrings().iter().map(|l| l.coords().len()).sum()
        }
        FeatureGeometry::Polygon(p) => polygon_coord_count(p),
        FeatureGeometry::MultiPolygon(mp) => mp.polygons().iter().map(polygon_coord_count).sum(),
        FeatureGeometry::GeometryCollection(members) => {
            members.iter().map(geometry_coord_count).sum()
        }
    }
}

fn polygon_coord_count(p: &Polygon) -> usize {
    p.exterior().coords().len()
        + p.interiors()
            .iter()
            .map(|r| r.coords().len())
            .sum::<usize>()
}

#[tokio::test]
async fn coarse_levels_simplify_and_drop_subpixel_features() {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(src()));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let base = engine.grid(vec).base_resolution;

    let bbox = Bbox::new(2.2, 4.2, 60.1, 60.2);
    let fine = engine
        .pull(
            vec,
            WindowReq {
                bbox,
                resolution: base,
            },
        )
        .await
        .unwrap()
        .into_vector()
        .unwrap();
    let coarse = engine
        .pull(
            vec,
            WindowReq {
                bbox,
                resolution: base * 8.0,
            },
        )
        .await
        .unwrap()
        .into_vector()
        .unwrap();

    assert!(
        fine.features.iter().any(|f| f.id == 4),
        "level 0 keeps the sub-pixel square"
    );
    assert!(
        !coarse.features.iter().any(|f| f.id == 4),
        "coarse level drops the sub-pixel square"
    );
    assert!(
        fragment_coord_count(&coarse) < fragment_coord_count(&fine) / 2,
        "coarse level should simplify, kept {} of {}",
        fragment_coord_count(&coarse),
        fragment_coord_count(&fine)
    );
    assert!(
        coarse.features.iter().any(|f| f.id == 2),
        "points survive the drop rule"
    );
}

/// identity transform that narrows the negotiated chunk size, forcing the
/// vector link onto small tiles so the seam test crosses chunk borders
struct SmallChunks;

impl Transform for SmallChunks {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Raster(RasterPattern {
            chunk_px: SetField::one(16),
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

#[tokio::test]
async fn chunked_rasterize_matches_the_whole_window_reference() {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(src()));
    let grid = g.add_transform(vec, Box::new(burn()));
    g.add_transform(grid, Box::new(SmallChunks));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(grid).raster().chunk_px, 16);
    assert_eq!(
        engine.caps(vec).vector().chunk_px,
        16,
        "chunk size passes back through the burn"
    );

    let base = engine.grid(grid).base_resolution;
    let got = engine
        .pull(
            grid,
            WindowReq {
                bbox: Bbox::new(3.0, 5.0, 58.0, 58.0),
                resolution: base,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap();

    let res = got.resolution;
    let (cols, rows) = (got.width(), got.height());
    let shapes: Vec<(FeatureGeometry, f64)> = collection()
        .features
        .into_iter()
        .filter_map(|f| {
            let v = f.properties.get("v")?.as_f64()?;
            Some((f.geometry?, v))
        })
        .collect();
    let reference = topoi_core::rasterize(
        &shapes,
        &topoi_core::GridWindow {
            origin_x: got.bbox.min_x,
            origin_y: got.bbox.min_y,
            width: cols,
            height: rows,
            cell_size: res,
        },
    );

    let band = got.bands.band(0).unwrap();
    for row in 0..rows {
        for col in 0..cols {
            let a = band.data()[row * cols + col];
            let r = reference[(rows - 1 - row) * cols + col];
            assert!(
                a == r || (a.is_nan() && r.is_nan()),
                "cell ({col},{row}): chunked {a} vs whole-window {r}"
            );
        }
    }
    assert!(band.data().contains(&3.0), "polygon fill must appear");
    assert!(band.data().contains(&9.0), "line burn must appear");
    assert!(band.data().contains(&7.0), "multilinestring must appear");
    assert!(band.data().contains(&6.0), "multipoint must appear");
    assert!(band.data().contains(&2.0), "collection must appear");
}

/// counts source reads so the spill test can tell a disk reload from a
/// recompute
struct CountingVec {
    inner: VecSrc,
    reads: Arc<AtomicUsize>,
}

impl Source for CountingVec {
    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> geoplumb::window::GridSpec {
        self.inner.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read(req)
    }
}

#[tokio::test]
async fn vector_chunks_spill_to_disk_and_reload() {
    // the budget holds the level-0 chunk and nothing more, so the coarse
    // pull evicts it and the second fine pull has to come off disk
    let probe = src();
    let bbox = Bbox::new(2.2, 4.2, 60.1, 60.2);
    let base = probe.grid().base_resolution;
    let fine = WindowReq {
        bbox,
        resolution: base,
    };
    let budget = probe.read(&fine).await.unwrap().byte_size();

    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(CountingVec {
        inner: src(),
        reads: reads.clone(),
    }));
    let engine = Engine::with_disk_cache(g, budget, std::env::temp_dir(), 64 << 20).unwrap();
    let coarse = WindowReq {
        bbox,
        resolution: base * 8.0,
    };

    let first = engine.pull(vec, fine).await.unwrap().into_vector().unwrap();
    engine.pull(vec, coarse).await.unwrap();
    let after = reads.load(Ordering::SeqCst);
    let again = engine.pull(vec, fine).await.unwrap().into_vector().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        after,
        "spilled chunk must reload from disk, not recompute"
    );

    let key = |f: &geoplumb::VectorFeature| {
        (
            f.id,
            geometry_shape(&f.geometry),
            geometry_bits(&f.geometry),
            f.properties.len(),
        )
    };
    let mut a: Vec<_> = first.features.iter().map(key).collect();
    let mut b: Vec<_> = again.features.iter().map(key).collect();
    a.sort();
    b.sort();
    assert_eq!(a, b, "reloaded fragments differ from the computed ones");
}

fn geometry_bits(geometry: &FeatureGeometry) -> Vec<(u64, u64)> {
    let bits = |c: &Coord| (c.x.to_bits(), c.y.to_bits());
    match geometry {
        FeatureGeometry::Point(p) => vec![bits(&p.0)],
        FeatureGeometry::MultiPoint(mp) => mp.points().iter().map(|p| bits(&p.0)).collect(),
        FeatureGeometry::LineString(l) => l.coords().iter().map(bits).collect(),
        FeatureGeometry::MultiLineString(mls) => mls
            .linestrings()
            .iter()
            .flat_map(|l| l.coords().iter().map(bits))
            .collect(),
        FeatureGeometry::Polygon(p) => polygon_bits(p),
        FeatureGeometry::MultiPolygon(mp) => mp.polygons().iter().flat_map(polygon_bits).collect(),
        FeatureGeometry::GeometryCollection(members) => {
            members.iter().flat_map(geometry_bits).collect()
        }
    }
}

/// variant and part counts, so a spill roundtrip that reloads the right
/// coordinates under the wrong geometry kind still fails
fn geometry_shape(geometry: &FeatureGeometry) -> String {
    match geometry {
        FeatureGeometry::Point(_) => "point".to_string(),
        FeatureGeometry::MultiPoint(mp) => format!("multipoint{}", mp.points().len()),
        FeatureGeometry::LineString(_) => "linestring".to_string(),
        FeatureGeometry::MultiLineString(mls) => {
            format!("multilinestring{}", mls.linestrings().len())
        }
        FeatureGeometry::Polygon(p) => format!("polygon{}", p.interiors().len()),
        FeatureGeometry::MultiPolygon(mp) => format!("multipolygon{}", mp.polygons().len()),
        FeatureGeometry::GeometryCollection(members) => format!(
            "collection[{}]",
            members
                .iter()
                .map(geometry_shape)
                .collect::<Vec<_>>()
                .join(",")
        ),
    }
}

fn polygon_bits(p: &Polygon) -> Vec<(u64, u64)> {
    let mut bits: Vec<(u64, u64)> = p
        .exterior()
        .coords()
        .iter()
        .map(|c| (c.x.to_bits(), c.y.to_bits()))
        .collect();
    for hole in p.interiors() {
        bits.extend(hole.coords().iter().map(|c| (c.x.to_bits(), c.y.to_bits())));
    }
    bits
}

/// raster identity that insists on web mercator, downstream of the burn
struct MercatorOnly;

impl Transform for MercatorOnly {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Raster(RasterPattern {
            crs: SetField::one(Crs::WEB_MERCATOR),
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

#[test]
fn crs_demand_downstream_of_the_burn_autoplugs_a_reproject() {
    let mut g = Graph::new();
    let vec = g.add_source(Box::new(src()));
    let grid = g.add_transform(vec, Box::new(burn()));
    let t = g.add_transform(grid, Box::new(MercatorOnly));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert!(matches!(engine.caps(vec), Caps::Vector(_)));
    assert_eq!(engine.caps(grid).raster().crs, Crs::WGS84);
    assert_eq!(engine.caps(t).raster().crs, Crs::WEB_MERCATOR);
}
