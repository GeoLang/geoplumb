use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use futures::future::BoxFuture;
use geoplumb::caps::Constraint;
use geoplumb::element::Source;
use geoplumb::elements::{Hillshade, RasterSrc, Reproject};
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Crs, Engine, Graph, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

fn elevation(lon: f64, lat: f64) -> f64 {
    500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos()
}

fn dem() -> Raster {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
            data.push(elevation(lon, lat));
        }
    }
    Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
}

fn dem_src() -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![dem()]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

struct CountingSrc {
    inner: RasterSrc,
    reads: Arc<AtomicUsize>,
    delay: Duration,
}

impl Source for CountingSrc {
    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> GridSpec {
        self.inner.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<geoplumb::Chunk>> {
        Box::pin(async move {
            self.reads.fetch_add(1, Ordering::SeqCst);
            if !self.delay.is_zero() {
                tokio::time::sleep(self.delay).await;
            }
            self.inner.read(req).await
        })
    }
}

fn window(px0: usize, py0: usize, px1: usize, py1: usize) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + px0 as f64 * CELL,
        max_x: ORIGIN_X + px1 as f64 * CELL,
        max_y: ORIGIN_Y - py0 as f64 * CELL,
        min_y: ORIGIN_Y - py1 as f64 * CELL,
    }
}

#[test]
fn negotiation_fixates_chain() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(dem_src()));
    let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let caps = engine.caps(hs).raster();
    assert_eq!(caps.crs, Crs::WGS84);
    assert_eq!(caps.bands, 1);
    assert_eq!(caps.chunk_px, 256);
}

#[test]
fn negotiation_failure_names_link() {
    let two_band = BandedRaster::new(vec![dem(), dem()]).unwrap();
    let mut g = Graph::new();
    let src = g.add_source(Box::new(RasterSrc::new(
        two_band,
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )));
    let _hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let err = Engine::new(g, 64 << 20)
        .err()
        .expect("two bands into hillshade");
    let msg = err.to_string();
    assert!(
        msg.contains("negotiation failed"),
        "unexpected error: {msg}"
    );
}

#[test]
fn reproject_retargets_crs_and_passes_bands() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(dem_src()));
    let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let rp = g.add_transform(hs, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(rp).raster().crs, Crs::WEB_MERCATOR);
    assert_eq!(engine.caps(rp).raster().bands, 1);
    assert_eq!(engine.caps(hs).raster().crs, Crs::WGS84);
}

#[tokio::test]
async fn hillshade_is_seam_free_across_chunks() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(dem_src()));
    let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    // spans the chunk border at pixel 256 in both axes
    let req = WindowReq {
        bbox: window(20, 20, 340, 230),
        resolution: CELL,
    };
    let got = engine.pull(hs, req).await.unwrap().into_raster().unwrap();

    // reference: hillshade over the padded window in one piece, pad cropped
    let pad = 4usize;
    let full = dem();
    let (x0, y0) = (20 - pad, 20 - pad);
    let (cols, rows) = (320 + 2 * pad, 210 + 2 * pad);
    let mut data = Vec::with_capacity(cols * rows);
    for r in 0..rows {
        for c in 0..cols {
            data.push(full.data()[(y0 + r) * W + (x0 + c)]);
        }
    }
    let patch = Raster::from_vec(cols, rows, data, CELL, f64::NAN).unwrap();
    let reference = terrano_core::hillshade(&patch, 315.0, 45.0);

    let band = got.bands.band(0).unwrap();
    assert_eq!(band.width(), 320);
    assert_eq!(band.height(), 210);
    for r in 0..210 {
        for c in 0..320 {
            let a = band.data()[r * 320 + c];
            let b = reference.data()[(r + pad) * cols + (c + pad)];
            assert!(
                (a - b).abs() < 1e-9,
                "seam at ({c},{r}): chunked {a} vs reference {b}"
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_pulls_coalesce() {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let src = g.add_source(Box::new(CountingSrc {
        inner: dem_src(),
        reads: reads.clone(),
        delay: Duration::from_millis(30),
    }));
    let engine = Arc::new(Engine::new(g, 64 << 20).unwrap());

    let req = WindowReq {
        bbox: window(0, 0, 300, 300),
        resolution: CELL,
    };
    let (a, b) = tokio::join!(
        tokio::spawn({
            let e = engine.clone();
            async move { e.pull(src, req).await }
        }),
        tokio::spawn({
            let e = engine.clone();
            async move { e.pull(src, req).await }
        }),
    );
    a.unwrap().unwrap();
    b.unwrap().unwrap();
    // window covers 2x2 chunks, both pulls together must read each once
    assert_eq!(reads.load(Ordering::SeqCst), 4);
}

#[tokio::test]
async fn cache_serves_repeat_pulls_and_invalidation_clears() {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let src = g.add_source(Box::new(CountingSrc {
        inner: dem_src(),
        reads: reads.clone(),
        delay: Duration::ZERO,
    }));
    let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let mut events = engine.subscribe();

    let req = WindowReq {
        bbox: window(10, 10, 200, 200),
        resolution: CELL,
    };
    engine.pull(hs, req).await.unwrap().into_raster().unwrap();
    let after_first = reads.load(Ordering::SeqCst);
    assert!(after_first > 0);

    engine.pull(hs, req).await.unwrap().into_raster().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        after_first,
        "cache miss on repeat"
    );

    // disjoint dirty window leaves the cache alone
    engine.invalidate(src, window(500, 300, 590, 390));
    engine.pull(hs, req).await.unwrap().into_raster().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        after_first,
        "disjoint invalidation recomputed"
    );

    // overlapping dirty window forces recompute and publishes events
    engine.invalidate(src, window(0, 0, 50, 50));
    let ev = events.try_recv().expect("invalidation event");
    assert_eq!(ev.node, src);
    let ev2 = events.try_recv().expect("downstream event");
    assert_eq!(ev2.node, hs);
    engine.pull(hs, req).await.unwrap().into_raster().unwrap();
    assert!(
        reads.load(Ordering::SeqCst) > after_first,
        "stale cache after invalidation"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancelled_pull_leaves_no_wedged_chunk() {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let src = g.add_source(Box::new(CountingSrc {
        inner: dem_src(),
        reads: reads.clone(),
        delay: Duration::from_millis(50),
    }));
    let engine = Arc::new(Engine::new(g, 64 << 20).unwrap());
    let req = WindowReq {
        bbox: window(0, 0, 100, 100),
        resolution: CELL,
    };

    let cancelled = tokio::time::timeout(Duration::from_millis(5), engine.pull(src, req)).await;
    assert!(
        cancelled.is_err(),
        "expected the first pull to be cancelled"
    );

    let chunk = tokio::time::timeout(Duration::from_secs(5), engine.pull(src, req))
        .await
        .expect("second pull wedged")
        .unwrap()
        .into_raster()
        .unwrap();
    assert_eq!(chunk.bands.band_count(), 1);
}

#[tokio::test]
async fn batch_materialize_walks_the_pyramid() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(dem_src()));
    let hs = g.add_transform(src, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let extent = window(0, 0, W, H);
    let mut seen = 0;
    let count = geoplumb::materialize(&engine, hs, extent, 2, |_k, chunk| {
        assert!(chunk.raster().unwrap().width() > 0);
        seen += 1;
    })
    .await
    .unwrap();
    assert_eq!(count, seen);
    // level 0 covers 3x2 chunks, level 1 covers 2x1, level 2 covers 1x1
    assert_eq!(count, 6 + 2 + 1);
}

#[tokio::test]
async fn reprojected_tile_matches_source_values() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(dem_src()));
    let rp = g.add_transform(src, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    // tile containing the dem center at z12
    let (lon, lat) = (
        ORIGIN_X + W as f64 * CELL / 2.0,
        ORIGIN_Y - H as f64 * CELL / 2.0,
    );
    let z = 12u8;
    let n = f64::from(1u32 << z);
    let x = ((lon + 180.0) / 360.0 * n).floor() as u32;
    let lat_rad = lat.to_radians();
    let y = ((1.0 - (lat_rad.tan() + 1.0 / lat_rad.cos()).ln() / std::f64::consts::PI) / 2.0 * n)
        .floor() as u32;
    let tile = geoplumb::tile::XyzTile { z, x, y };
    let chunk = geoplumb::tile::render_tile(&engine, rp, tile)
        .await
        .unwrap();
    assert_eq!(chunk.width(), geoplumb::tile::TILE_PX);

    // center pixel against the analytic elevation at the tile center
    let band = chunk.bands.band(0).unwrap();
    let center = band.data()[128 * 256 + 128];
    let b = tile.bbox();
    let (cx, cy) = ((b.min_x + b.max_x) / 2.0, (b.min_y + b.max_y) / 2.0);
    let inv = projicio_core::Transform::new("EPSG:3857", "EPSG:4326").unwrap();
    let (clon, clat) = inv.convert(cx, cy).unwrap();
    let expected = elevation(clon, clat);
    assert!(
        (center - expected).abs() < 2.0,
        "tile center {center} vs analytic {expected}"
    );
}

#[tokio::test]
async fn an_absurd_pull_fails_instead_of_enumerating() {
    let cell = 1e-12;
    let px = Raster::from_vec(1, 1, vec![0.0], cell, f64::NAN).unwrap();
    let src = RasterSrc::new(BandedRaster::new(vec![px]).unwrap(), 0.0, 1.0, Crs::WGS84);
    let mut g = Graph::new();
    let s = g.add_source(Box::new(src));
    let engine = Engine::new(g, 16 << 20).unwrap();
    let err = engine
        .pull(
            s,
            WindowReq {
                bbox: Bbox::new(0.0, 0.0, 1.0, 1.0),
                resolution: cell,
            },
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, geoplumb::Error::PullTooLarge { .. }),
        "unexpected error: {err}"
    );
}
