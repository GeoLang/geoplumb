//! reproject auto-plug: a link empty only because of crs gets a reproject
//! spliced in during the solve instead of failing negotiation

use geoplumb::caps::{CapsPattern, CapsSet, Constraint, FieldMask, RasterPattern, SetField};
use geoplumb::element::{Adapter, Transform};
use geoplumb::elements::{Hillshade, Mosaic, RasterSrc};
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, RasterChunk, Result, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

fn elevation(lon: f64, lat: f64) -> f64 {
    500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos()
}

/// left part of the scene as a wgs84 source
fn wgs84_src(cols: usize) -> RasterSrc {
    let mut data = Vec::with_capacity(cols * H);
    for row in 0..H {
        for col in 0..cols {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(elevation(lon, lat));
        }
    }
    RasterSrc::new(
        BandedRaster::new(vec![
            Raster::from_vec(cols, H, data, CELL, f64::NAN).unwrap(),
        ])
        .unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

/// the scene from lon 7.3 to 7.5, lat 46.8 to 47.0, as a web mercator
/// source at 100 m cells
fn mercator_src() -> RasterSrc {
    let fwd = projicio_core::Transform::new("EPSG:4326", "EPSG:3857").unwrap();
    let inv = projicio_core::Transform::new("EPSG:3857", "EPSG:4326").unwrap();
    let (x0, y0) = fwd.convert(7.3, 47.0).unwrap();
    let (x1, y1) = fwd.convert(7.5, 46.8).unwrap();
    let cell = 100.0;
    let cols = ((x1 - x0) / cell).ceil() as usize;
    let rows = ((y0 - y1) / cell).ceil() as usize;
    let centers: Vec<(f64, f64)> = (0..rows)
        .flat_map(|r| {
            (0..cols).map(move |c| (x0 + (c as f64 + 0.5) * cell, y0 - (r as f64 + 0.5) * cell))
        })
        .collect();
    let lonlat = inv.convert_batch(&centers).unwrap();
    let data: Vec<f64> = lonlat
        .iter()
        .map(|&(lon, lat)| elevation(lon, lat))
        .collect();
    RasterSrc::new(
        BandedRaster::new(vec![
            Raster::from_vec(cols, rows, data, cell, f64::NAN).unwrap(),
        ])
        .unwrap(),
        x0,
        y0,
        Crs::WEB_MERCATOR,
    )
}

/// every finite pixel whose center lies inside `within` must match the
/// analytic elevation at that center, mapped to lon/lat by `to_lonlat`
fn assert_matches_analytic(
    chunk: &RasterChunk,
    within: &Bbox,
    to_lonlat: impl Fn(f64, f64) -> (f64, f64),
    tol: f64,
) {
    let band = chunk.bands.band(0).unwrap();
    let res = chunk.resolution;
    let mut checked = 0;
    for row in 0..band.height() {
        for col in 0..band.width() {
            let x = chunk.bbox.min_x + (col as f64 + 0.5) * res;
            let y = chunk.bbox.max_y - (row as f64 + 0.5) * res;
            if x < within.min_x || x > within.max_x || y < within.min_y || y > within.max_y {
                continue;
            }
            let (lon, lat) = to_lonlat(x, y);
            let expected = elevation(lon, lat);
            let got = band.data()[row * band.width() + col];
            assert!(
                (got - expected).abs() < tol,
                "({col},{row}): {got} vs {expected}"
            );
            checked += 1;
        }
    }
    assert!(checked > 1000, "only {checked} pixels inside the window");
}

#[tokio::test]
async fn mixed_crs_fanin_autoplugs_a_reproject() {
    let mut g = Graph::new();
    let a = g.add_source(Box::new(wgs84_src(300)));
    let b = g.add_source(Box::new(mercator_src()));
    let m = g.add_fanin(&[a, b], Box::new(Mosaic));
    let engine = Engine::new(g, 64 << 20).unwrap();

    // the fanin lands on the first parent's crs, sources keep their own
    assert_eq!(engine.caps(m).raster().crs, Crs::WGS84);
    assert_eq!(engine.caps(a).raster().crs, Crs::WGS84);
    assert_eq!(engine.caps(b).raster().crs, Crs::WEB_MERCATOR);

    // a window only the mercator source covers
    let req = WindowReq {
        bbox: Bbox {
            min_x: 7.32,
            max_x: 7.48,
            max_y: 46.98,
            min_y: 46.82,
        },
        resolution: CELL,
    };
    let got = engine.pull(m, req).await.unwrap().into_raster().unwrap();
    assert_matches_analytic(&got, &req.bbox, |x, y| (x, y), 2.0);
}

/// identity transform that insists on web mercator input
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

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        Ok(Chunk::Raster(input.raster()?.crop_to(&out.bbox)))
    }
}

#[tokio::test]
async fn crs_demanding_transform_autoplugs_upstream() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(wgs84_src(W)));
    let t = g.add_transform(src, Box::new(MercatorOnly));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(t).raster().crs, Crs::WEB_MERCATOR);
    assert_eq!(engine.caps(src).raster().crs, Crs::WGS84);

    let fwd = projicio_core::Transform::new("EPSG:4326", "EPSG:3857").unwrap();
    let inv = projicio_core::Transform::new("EPSG:3857", "EPSG:4326").unwrap();
    let (x0, y0) = fwd.convert(7.1, 46.95).unwrap();
    let (x1, y1) = fwd.convert(7.4, 46.75).unwrap();
    let req = WindowReq {
        bbox: Bbox {
            min_x: x0,
            max_x: x1,
            max_y: y0,
            min_y: y1,
        },
        resolution: 200.0,
    };
    let got = engine.pull(t, req).await.unwrap().into_raster().unwrap();
    assert_matches_analytic(&got, &req.bbox, |x, y| inv.convert(x, y).unwrap(), 2.0);
}

/// user-defined adapter: keeps only the first band. proves the solver is
/// generic over adapters rather than special-casing any caps field
struct BandPick;

impl Transform for BandPick {
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::any_raster(),
            passthrough: FieldMask {
                bands: false,
                ..FieldMask::ALL
            },
            output: CapsPattern::Raster(RasterPattern {
                bands: SetField::one(1),
                ..RasterPattern::default()
            }),
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let first = input.bands.band(0).expect("at least one band").clone();
        Ok(Chunk::Raster(
            RasterChunk {
                bands: BandedRaster::new(vec![first]).expect("single band"),
                bbox: input.bbox,
                resolution: input.resolution,
                crs: input.crs,
            }
            .crop_to(&out.bbox),
        ))
    }
}

fn band_pick_adapter() -> Adapter {
    Adapter {
        template: Constraint::Derived {
            input: CapsSet::any_raster(),
            passthrough: FieldMask {
                bands: false,
                ..FieldMask::ALL
            },
            output: CapsPattern::Raster(RasterPattern::default()),
        },
        build: |target| {
            let CapsPattern::Raster(target) = target else {
                return None;
            };
            match &target.bands {
                SetField::OneOf(v) if v.contains(&1) => Some(Box::new(BandPick)),
                _ => None,
            }
        },
    }
}

fn two_band_graph() -> (Graph, geoplumb::NodeId) {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(elevation(lon, lat));
        }
    }
    let dem = Raster::from_vec(W, H, data.clone(), CELL, f64::NAN).unwrap();
    let noise =
        Raster::from_vec(W, H, data.iter().map(|v| v * 2.0).collect(), CELL, f64::NAN).unwrap();
    let src = RasterSrc::new(
        BandedRaster::new(vec![dem, noise]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    );
    let mut g = Graph::new();
    let s = g.add_source(Box::new(src));
    let hs = g.add_transform(s, Box::new(Hillshade::new(315.0, 45.0)));
    (g, hs)
}

#[tokio::test]
async fn registered_adapter_bridges_a_bands_mismatch() {
    // without the adapter a two-band source into hillshade fails
    let (g, _) = two_band_graph();
    assert!(Engine::new(g, 64 << 20).is_err());

    let (mut g, hs) = two_band_graph();
    g.register_adapter(band_pick_adapter());
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(hs).raster().bands, 1);

    // the plugged graph matches a hand-built single-band pipeline
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(elevation(lon, lat));
        }
    }
    let dem = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
    let mut rg = Graph::new();
    let rs = rg.add_source(Box::new(RasterSrc::new(
        BandedRaster::new(vec![dem]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )));
    let rhs = rg.add_transform(rs, Box::new(Hillshade::new(315.0, 45.0)));
    let reference = Engine::new(rg, 64 << 20).unwrap();

    let req = WindowReq {
        bbox: Bbox {
            min_x: 7.05,
            max_x: 7.25,
            max_y: 46.95,
            min_y: 46.8,
        },
        resolution: CELL,
    };
    let a = engine.pull(hs, req).await.unwrap().into_raster().unwrap();
    let b = reference
        .pull(rhs, req)
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let (ba, bb) = (a.bands.band(0).unwrap(), b.bands.band(0).unwrap());
    for (i, (x, y)) in ba.data().iter().zip(bb.data()).enumerate() {
        assert!((x - y).abs() < 1e-12, "pixel {i}: {x} vs {y}");
    }
}

#[tokio::test]
async fn invalidation_crosses_the_spliced_plug() {
    let mut g = Graph::new();
    let a = g.add_source(Box::new(wgs84_src(300)));
    let b = g.add_source(Box::new(mercator_src()));
    let m = g.add_fanin(&[a, b], Box::new(Mosaic));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let mut events = engine.subscribe();

    let fwd = projicio_core::Transform::new("EPSG:4326", "EPSG:3857").unwrap();
    let (x0, y0) = fwd.convert(7.35, 46.95).unwrap();
    let (x1, y1) = fwd.convert(7.4, 46.9).unwrap();
    engine.invalidate(
        b,
        Bbox {
            min_x: x0,
            max_x: x1,
            max_y: y0,
            min_y: y1,
        },
    );
    let mut hit_fanin = false;
    while let Ok(ev) = events.try_recv() {
        if ev.node == m {
            hit_fanin = true;
            // the dirty window crossed the plug into wgs84 coordinates
            assert!(ev.bbox.min_x > 7.3 && ev.bbox.max_x < 7.5, "{:?}", ev.bbox);
        }
    }
    assert!(hit_fanin, "no invalidation event reached the fanin");
}
