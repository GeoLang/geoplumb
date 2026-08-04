//! tensor chunks end to end: negotiation across both kind boundaries, the
//! chunk size demand a model input size arrives as, seam-free convolution,
//! spill reload

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use geoplumb::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField, TensorPattern};
use geoplumb::element::{Source, Transform};
use geoplumb::elements::{Hillshade, RasterSrc, TensorConv, ToRaster, ToTensor};
use geoplumb::{Bbox, Chunk, Crs, Engine, Error, Graph, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 320;
const H: usize = 256;
// binary exact, so pixel-aligned windows survive outward alignment
const CELL: f64 = 1.0 / 1024.0;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;
const CHANNELS: usize = 3;

const SCALES: [f32; CHANNELS] = [1.0, 2.0, 0.5];
const OFFSETS: [f32; CHANNELS] = [0.0, -1.0, 0.25];
const KERNEL: [[f32; 3]; 3] = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]];

/// modest magnitudes so f32 sums stay well inside single precision
fn band_value(band: usize, col: usize, row: usize) -> f64 {
    let (x, y) = (col as f64, row as f64);
    match band {
        0 => 0.5 + 0.25 * (x * 0.11).sin() * (y * 0.07).cos(),
        1 => 0.2 + 0.1 * ((x + y) * 0.05).sin(),
        _ => 0.9 - 0.3 * (x * 0.03).cos(),
    }
}

fn src() -> RasterSrc {
    let bands: Vec<Raster> = (0..CHANNELS)
        .map(|b| {
            let mut data = Vec::with_capacity(W * H);
            for row in 0..H {
                for col in 0..W {
                    data.push(band_value(b, col, row));
                }
            }
            Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
        })
        .collect();
    RasterSrc::new(
        BandedRaster::new(bands).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

fn one_band_src() -> RasterSrc {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            data.push(band_value(0, col, row));
        }
    }
    let band = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
    RasterSrc::new(
        BandedRaster::new(vec![band]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

fn to_tensor() -> ToTensor {
    ToTensor {
        scales: SCALES.to_vec(),
        offsets: OFFSETS.to_vec(),
    }
}

fn conv() -> TensorConv {
    TensorConv { kernel: KERNEL }
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
fn tensor_chain_negotiates_across_both_kind_boundaries() {
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(src()));
    let tt = g.add_transform(raster, Box::new(to_tensor()));
    let cv = g.add_transform(tt, Box::new(conv()));
    let tr = g.add_transform(cv, Box::new(ToRaster));
    let engine = Engine::new(g, 64 << 20).unwrap();

    assert_eq!(engine.caps(raster).raster().bands, CHANNELS as u16);
    let tensor_caps = engine.caps(tt).tensor();
    assert_eq!(tensor_caps.crs, Crs::WGS84, "crs passes into the tensor");
    assert_eq!(tensor_caps.channels, CHANNELS as u16);
    assert_eq!(tensor_caps.dtype, geoplumb::caps::TensorDtype::F32);
    assert_eq!(engine.caps(cv).tensor().channels, CHANNELS as u16);
    let out = engine.caps(tr).raster();
    assert_eq!(out.crs, Crs::WGS84, "crs passes back out of the tensor");
    assert_eq!(
        out.bands, CHANNELS as u16,
        "channels couple back to bands through the plane count"
    );
}

#[test]
fn single_channel_chain_satisfies_a_single_band_consumer() {
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(one_band_src()));
    let tt = g.add_transform(
        raster,
        Box::new(ToTensor {
            scales: vec![2.0],
            offsets: vec![0.5],
        }),
    );
    let cv = g.add_transform(tt, Box::new(conv()));
    let tr = g.add_transform(cv, Box::new(ToRaster));
    let hs = g.add_transform(tr, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(tt).tensor().channels, 1);
    assert_eq!(engine.caps(tr).raster().bands, 1);
    assert_eq!(engine.caps(hs).raster().bands, 1);
}

#[test]
fn multi_channel_tensor_cannot_satisfy_a_single_band_consumer() {
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(src()));
    let tt = g.add_transform(raster, Box::new(to_tensor()));
    let tr = g.add_transform(tt, Box::new(ToRaster));
    g.add_transform(tr, Box::new(Hillshade::new(315.0, 45.0)));
    match Engine::new(g, 64 << 20) {
        Err(Error::EmptyLink { .. }) => {}
        Err(other) => panic!("expected EmptyLink, got {other:?}"),
        Ok(_) => panic!("a band demand must narrow the channel count and fail"),
    }
}

#[test]
fn raster_source_cannot_feed_a_tensor_element_directly() {
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(src()));
    g.add_transform(raster, Box::new(conv()));
    match Engine::new(g, 64 << 20) {
        Err(Error::EmptyLink { .. }) => {}
        Err(other) => panic!("expected EmptyLink, got {other:?}"),
        Ok(_) => panic!("kind mismatch must fail negotiation"),
    }
}

#[test]
fn tensor_cannot_feed_a_raster_consumer_directly() {
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(src()));
    let tt = g.add_transform(raster, Box::new(to_tensor()));
    g.add_transform(tt, Box::new(Hillshade::new(315.0, 45.0)));
    match Engine::new(g, 64 << 20) {
        Err(Error::EmptyLink { .. }) => {}
        Err(other) => panic!("expected EmptyLink, got {other:?}"),
        Ok(_) => panic!("kind mismatch must fail negotiation"),
    }
}

/// raster identity demanding small tiles, the shape a fixed model input
/// size takes once it is back on the raster side
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

/// the same demand on the tensor side, so the conv runs on 16 px tiles
struct SmallTensorChunks;

impl Transform for SmallTensorChunks {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Tensor(TensorPattern {
            chunk_px: SetField::one(16),
            ..TensorPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        Ok(Chunk::Tensor(input.tensor()?.crop_to(&out.bbox)))
    }
}

#[test]
fn chunk_px_demand_narrows_the_tensor_links() {
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(src()));
    let tt = g.add_transform(raster, Box::new(to_tensor()));
    let cv = g.add_transform(tt, Box::new(conv()));
    let tr = g.add_transform(cv, Box::new(ToRaster));
    g.add_transform(tr, Box::new(SmallChunks));
    let engine = Engine::new(g, 64 << 20).unwrap();

    assert_eq!(engine.caps(tr).raster().chunk_px, 16);
    assert_eq!(
        engine.caps(cv).tensor().chunk_px,
        16,
        "chunk size passes back through the tensor to raster boundary"
    );
    assert_eq!(engine.caps(tt).tensor().chunk_px, 16);
    assert_eq!(
        engine.caps(raster).raster().chunk_px,
        16,
        "and on through the raster to tensor boundary"
    );
}

#[tokio::test]
async fn chunked_conv_matches_the_whole_window_reference() {
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(src()));
    let tt = g.add_transform(raster, Box::new(to_tensor()));
    let cv = g.add_transform(tt, Box::new(conv()));
    g.add_transform(cv, Box::new(SmallTensorChunks));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(cv).tensor().chunk_px, 16);

    // 5x4 tiles of 16 px, so most cells sit next to a seam the halo has to
    // cross
    let (px0, py0) = (32usize, 32usize);
    let (cols, rows) = (80usize, 64usize);
    let got = engine
        .pull(
            cv,
            WindowReq {
                bbox: window(px0, py0, px0 + cols, py0 + rows),
                resolution: CELL,
            },
        )
        .await
        .unwrap()
        .into_tensor()
        .unwrap();
    assert_eq!((got.width(), got.height()), (cols, rows));
    assert_eq!(got.channels, CHANNELS as u16);

    let value = |c: usize, col: usize, row: usize| -> f32 {
        band_value(c, col, row) as f32 * SCALES[c] + OFFSETS[c]
    };
    for c in 0..CHANNELS {
        for row in 0..rows {
            for col in 0..cols {
                let mut want = 0.0f32;
                for (ky, krow) in KERNEL.iter().enumerate() {
                    for (kx, k) in krow.iter().enumerate() {
                        let (sx, sy) = (px0 + col + kx - 1, py0 + row + ky - 1);
                        want += value(c, sx, sy) * k;
                    }
                }
                let a = got.channel(c)[row * cols + col];
                assert!(
                    (a - want).abs() < 1e-5 || (a.is_nan() && want.is_nan()),
                    "channel {c} cell ({col},{row}): chunked {a} vs whole-window {want}"
                );
            }
        }
    }
}

/// counts source reads so the spill test can tell a disk reload from a
/// recompute
struct CountingSrc {
    inner: RasterSrc,
    reads: Arc<AtomicUsize>,
}

impl Source for CountingSrc {
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
async fn tensor_chunks_spill_to_disk_and_reload() {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let raster = g.add_source(Box::new(CountingSrc {
        inner: src(),
        reads: reads.clone(),
    }));
    let tt = g.add_transform(raster, Box::new(to_tensor()));
    // exactly one level-0 tensor chunk, no slack: the coarse pull must
    // evict it, and any slack would keep it resident and prove nothing
    let budget = CHANNELS * 256 * 256 * size_of::<f32>();
    let engine = Engine::with_disk_cache(g, budget, std::env::temp_dir(), 64 << 20).unwrap();

    let bbox = window(0, 0, 256, 256);
    let fine = WindowReq {
        bbox,
        resolution: CELL,
    };
    let coarse = WindowReq {
        bbox,
        resolution: CELL * 2.0,
    };

    let first = engine.pull(tt, fine).await.unwrap().into_tensor().unwrap();
    assert_eq!(first.byte_size(), budget);
    engine.pull(tt, coarse).await.unwrap();
    let after = reads.load(Ordering::SeqCst);
    let again = engine.pull(tt, fine).await.unwrap().into_tensor().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        after,
        "spilled chunk must reload from disk, not recompute"
    );

    let bits = |t: &geoplumb::TensorChunk| t.data.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
    assert_eq!(
        bits(&first),
        bits(&again),
        "reloaded tensor differs from the computed one"
    );
    assert_eq!(again.channels, CHANNELS as u16);
}
