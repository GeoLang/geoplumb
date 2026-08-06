//! fan-in: mosaic stitching, two-input algebra, backtracking caps
//! fixation across a diamond, and invalidation through a fanin node

use futures::future::BoxFuture;
use geoplumb::caps::{CapsPattern, CapsSet, Constraint, Dtype, RasterPattern, ResRange, SetField};
use geoplumb::element::Source;
use geoplumb::elements::{Combine, Hillshade, Mosaic, RasterSrc, Slope};
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, BinaryOp, Raster};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

fn elevation(lon: f64, lat: f64) -> f64 {
    500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos()
}

fn dem_patch(x0: usize, y0: usize, cols: usize, rows: usize) -> Raster {
    let mut data = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            let lon = ORIGIN_X + (x0 + col) as f64 * CELL + 0.5 * CELL;
            let lat = ORIGIN_Y - (y0 + row) as f64 * CELL - 0.5 * CELL;
            data.push(elevation(lon, lat));
        }
    }
    Raster::from_vec(cols, rows, data, CELL, f64::NAN).unwrap()
}

fn patch_src(x0: usize, cols: usize) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![dem_patch(x0, 0, cols, H)]).unwrap(),
        ORIGIN_X + x0 as f64 * CELL,
        ORIGIN_Y,
        Crs::WGS84,
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

fn assert_close(chunk: &RasterChunk, reference: &Raster, tol: f64) {
    let band = chunk.bands.band(0).unwrap();
    assert_eq!(band.width(), reference.width());
    assert_eq!(band.height(), reference.height());
    for (i, (a, b)) in band.data().iter().zip(reference.data()).enumerate() {
        assert!((a - b).abs() < tol, "pixel {i}: {a} vs {b}");
    }
}

#[tokio::test]
async fn mosaic_stitches_sources_with_offset_origins() {
    // left and right halves as separate sources anchored on their own
    // origins, 300 base pixels apart
    let mut g = Graph::new();
    let left = g.add_source(Box::new(patch_src(0, 300)));
    let right = g.add_source(Box::new(patch_src(300, 300)));
    let m = g.add_fanin(&[left, right], Box::new(Mosaic));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let req = WindowReq {
        bbox: window(0, 0, W, H),
        resolution: CELL,
        time: None,
    };
    let got = engine.pull(m, req).await.unwrap().into_raster().unwrap();
    assert_close(&got, &dem_patch(0, 0, W, H), 1e-9);
}

#[tokio::test]
async fn combine_subtracts_per_cell() {
    let mut g = Graph::new();
    let a = g.add_source(Box::new(patch_src(0, W)));
    let b = g.add_source(Box::new(patch_src(0, W)));
    let c = g.add_fanin(&[a, b], Box::new(Combine::new(BinaryOp::Subtract)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let req = WindowReq {
        bbox: window(10, 10, 500, 300),
        resolution: CELL,
        time: None,
    };
    let got = engine.pull(c, req).await.unwrap().into_raster().unwrap();
    let band = got.bands.band(0).unwrap();
    for (i, v) in band.data().iter().enumerate() {
        assert!(v.abs() < 1e-12, "pixel {i}: {v} should be zero");
    }
}

#[tokio::test]
async fn diamond_pull_combines_branches() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(patch_src(0, W)));
    let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let sl = g.add_transform(src, Box::new(Slope));
    let c = g.add_fanin(&[hs, sl], Box::new(Combine::new(BinaryOp::Add)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let req = WindowReq {
        bbox: window(20, 20, 340, 230),
        resolution: CELL,
        time: None,
    };
    let sum = engine.pull(c, req).await.unwrap().into_raster().unwrap();
    let h = engine.pull(hs, req).await.unwrap().into_raster().unwrap();
    let s = engine.pull(sl, req).await.unwrap().into_raster().unwrap();
    let (sb, hb, slb) = (
        sum.bands.band(0).unwrap(),
        h.bands.band(0).unwrap(),
        s.bands.band(0).unwrap(),
    );
    for i in 0..sb.data().len() {
        let expected = hb.data()[i] + slb.data()[i];
        let got = sb.data()[i];
        assert!(
            (got - expected).abs() < 1e-9,
            "pixel {i}: {got} vs {expected}"
        );
    }
}

/// source offering several crs alternatives in its own preference order
struct PrefSrc {
    crss: Vec<Crs>,
}

impl Source for PrefSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet {
            alternatives: self
                .crss
                .iter()
                .map(|&crs| {
                    CapsPattern::Raster(RasterPattern {
                        dtype: SetField::one(Dtype::F64),
                        bands: SetField::one(1),
                        crs: SetField::one(crs),
                        resolution: ResRange::at_least(CELL),
                        chunk_px: SetField::Any,
                    })
                })
                .collect(),
        })
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
        Box::pin(async move {
            let cols = (req.bbox.width() / req.resolution).round() as usize;
            let rows = (req.bbox.height() / req.resolution).round() as usize;
            let band =
                Raster::from_vec(cols, rows, vec![1.0; cols * rows], req.resolution, f64::NAN)
                    .unwrap();
            Ok(Chunk::Raster(RasterChunk {
                bands: BandedRaster::new(vec![band]).unwrap(),
                bbox: req.bbox,
                resolution: req.resolution,
                crs: self.crss[0],
            }))
        })
    }
}

#[test]
fn fanin_fixation_backtracks_across_disagreeing_preferences() {
    // greedy per-link fixation picks wgs84 for one parent and mercator for
    // the other, jointly impossible at the mosaic. backtracking must land
    // both on one crs
    let mut g = Graph::new();
    let a = g.add_source(Box::new(PrefSrc {
        crss: vec![Crs::WGS84, Crs::WEB_MERCATOR],
    }));
    let b = g.add_source(Box::new(PrefSrc {
        crss: vec![Crs::WEB_MERCATOR, Crs::WGS84],
    }));
    let m = g.add_fanin(&[a, b], Box::new(Mosaic));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let crs = engine.caps(m).raster().crs;
    assert_eq!(engine.caps(a).raster().crs, crs);
    assert_eq!(engine.caps(b).raster().crs, crs);
}

#[test]
fn combine_rejects_wrong_input_count() {
    let mut g = Graph::new();
    let a = g.add_source(Box::new(patch_src(0, W)));
    let b = g.add_source(Box::new(patch_src(0, W)));
    let c = g.add_source(Box::new(patch_src(0, W)));
    g.add_fanin(&[a, b, c], Box::new(Combine::new(BinaryOp::Add)));
    let err = Engine::new(g, 64 << 20).err().expect("three-input combine");
    assert!(err.to_string().contains("two inputs"), "{err}");
}

#[tokio::test]
async fn invalidation_crosses_the_fanin() {
    let mut g = Graph::new();
    let left = g.add_source(Box::new(patch_src(0, 300)));
    let right = g.add_source(Box::new(patch_src(300, 300)));
    let m = g.add_fanin(&[left, right], Box::new(Mosaic));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let mut events = engine.subscribe();

    let req = WindowReq {
        bbox: window(0, 0, 200, 200),
        resolution: CELL,
        time: None,
    };
    engine.pull(m, req).await.unwrap().into_raster().unwrap();

    engine.invalidate(left, window(0, 0, 50, 50));
    let ev = events.try_recv().expect("source event");
    assert_eq!(ev.node, left);
    let ev2 = events.try_recv().expect("fanin event");
    assert_eq!(ev2.node, m);
    assert!(events.try_recv().is_err(), "right source got an event");
}
