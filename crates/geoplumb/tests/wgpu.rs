#![cfg(feature = "wgpu")]

use std::path::Path;
use std::sync::Arc;

use geoplumb::caps::{CapsPattern, CapsSet, Constraint, Domain, SetField, TensorPattern};
use geoplumb::element::Transform;
use geoplumb::elements::{RasterSrc, TensorConv, ToRaster, ToTensor, WgpuConvolve};
use geoplumb::gpu::{DISPATCHES_PER_SUBMIT, GpuContext};
use geoplumb::{Bbox, Chunk, Crs, Engine, Error, Graph, NodeId, TensorChunk, WindowReq};
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

fn context(budget_bytes: usize) -> Option<Arc<GpuContext>> {
    match GpuContext::request(budget_bytes) {
        Ok(context) => Some(Arc::new(context)),
        Err(e) if std::env::var("GEOPLUMB_REQUIRE_GPU").is_ok_and(|value| value == "1") => {
            panic!("GEOPLUMB_REQUIRE_GPU is set: {e}")
        }
        Err(_) => None,
    }
}

// modest magnitudes so f32 sums stay well inside single precision
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

fn holed_src() -> RasterSrc {
    let bands: Vec<Raster> = (0..CHANNELS)
        .map(|b| {
            let mut data = Vec::with_capacity(W * H);
            for row in 0..H {
                for col in 0..W {
                    let hole = (col + 3 * row) % 41 == 0;
                    data.push(if hole {
                        f64::NAN
                    } else {
                        band_value(b, col, row)
                    });
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

fn to_tensor() -> ToTensor {
    ToTensor {
        scales: SCALES.to_vec(),
        offsets: OFFSETS.to_vec(),
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

// pins 16 px tensor tiles so one pull spans several convolve chunks
struct SmallTensorChunks;

impl Transform for SmallTensorChunks {
    fn identity(&self) -> String {
        "SmallTensorChunks".into()
    }

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
        Ok(Chunk::Tensor(input.tensor()?.crop_to(&out.bbox)?))
    }
}

struct Chain {
    tensor: NodeId,
    convolve: NodeId,
    raster: NodeId,
}

fn chain(graph: &mut Graph, source: RasterSrc, convolve: Box<dyn Transform>) -> Chain {
    let raster_source = graph.add_source(Box::new(source));
    let tensor = graph.add_transform(raster_source, Box::new(to_tensor()));
    let convolve = graph.add_transform(tensor, convolve);
    let raster = graph.add_transform(convolve, Box::new(ToRaster));
    Chain {
        tensor,
        convolve,
        raster,
    }
}

fn small_tile_chain(graph: &mut Graph, source: RasterSrc, convolve: Box<dyn Transform>) -> Chain {
    let raster_source = graph.add_source(Box::new(source));
    let tensor = graph.add_transform(raster_source, Box::new(to_tensor()));
    let convolve = graph.add_transform(tensor, convolve);
    let small = graph.add_transform(convolve, Box::new(SmallTensorChunks));
    let raster = graph.add_transform(small, Box::new(ToRaster));
    Chain {
        tensor,
        convolve,
        raster,
    }
}

// 5x4 tiles of 16 px, so most cells sit next to a seam the halo has to cross
const SEAM_WINDOW: (usize, usize, usize, usize) = (32, 32, 80, 64);

fn seam_window_request() -> WindowReq {
    let (px0, py0, cols, rows) = SEAM_WINDOW;
    WindowReq {
        bbox: window(px0, py0, px0 + cols, py0 + rows),
        resolution: CELL,
        time: None,
    }
}

#[test]
fn solve_splices_upload_and_download() {
    let Some(context) = context(64 << 20) else {
        return;
    };
    let mut g = Graph::with_gpu(context);
    let nodes = chain(&mut g, src(), Box::new(WgpuConvolve { kernel: KERNEL }));
    let engine = Engine::new(g, 64 << 20).unwrap();

    assert_eq!(
        engine.node_count(),
        6,
        "the solve splices one upload and one download into the four nodes"
    );
    assert_eq!(engine.caps(nodes.tensor).tensor().domain, Domain::System);
    assert_eq!(engine.caps(nodes.convolve).tensor().domain, Domain::Wgpu);
    assert_eq!(engine.caps(nodes.raster).raster().bands, CHANNELS as u16);
}

#[tokio::test]
async fn wgpu_convolve_matches_the_cpu_convolve() {
    let Some(context) = context(64 << 20) else {
        return;
    };
    let mut g = Graph::new();
    let cpu_nodes = small_tile_chain(&mut g, holed_src(), Box::new(TensorConv { kernel: KERNEL }));
    let cpu = Engine::new(g, 64 << 20).unwrap();
    let mut g = Graph::with_gpu(context);
    let gpu_nodes = small_tile_chain(
        &mut g,
        holed_src(),
        Box::new(WgpuConvolve { kernel: KERNEL }),
    );
    let gpu = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(gpu.caps(gpu_nodes.convolve).tensor().chunk_px, 16);

    let (_, _, cols, rows) = SEAM_WINDOW;
    let req = seam_window_request();
    let want = cpu
        .pull(cpu_nodes.raster, req)
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let got = gpu
        .pull(gpu_nodes.raster, req)
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    assert_eq!((got.width(), got.height()), (cols, rows));
    assert_eq!(got.bands.band_count(), CHANNELS);

    let mut nan_cells = 0;
    for c in 0..CHANNELS {
        let (a, b) = (got.bands.band(c).unwrap(), want.bands.band(c).unwrap());
        for (i, (x, y)) in a.data().iter().zip(b.data()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5 || (x.is_nan() && y.is_nan()),
                "channel {c} cell {i}: wgpu {x} vs cpu {y}"
            );
            nan_cells += usize::from(y.is_nan());
        }
    }
    assert!(nan_cells > 0, "the holes never reached the window");
}

#[tokio::test]
async fn a_pull_submits_once_per_batch() {
    let Some(context) = context(64 << 20) else {
        return;
    };
    let mut g = Graph::with_gpu(context.clone());
    let nodes = small_tile_chain(&mut g, src(), Box::new(WgpuConvolve { kernel: KERNEL }));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let (_, _, cols, rows) = SEAM_WINDOW;
    let download_chunks = ((cols / 16) * (rows / 16)) as u64;

    let (submissions_before, dispatches_before) = (context.submissions(), context.dispatches());
    engine
        .pull(nodes.raster, seam_window_request())
        .await
        .unwrap();
    let submitted = context.submissions() - submissions_before;
    let dispatched = context.dispatches() - dispatches_before;

    assert!(
        submitted <= 2 * download_chunks + dispatched / DISPATCHES_PER_SUBMIT as u64,
        "{submitted} submissions for {dispatched} dispatches over {download_chunks} downloads"
    );
    assert!(
        submitted < dispatched,
        "{submitted} submissions for {dispatched} dispatches, nothing was batched"
    );
}

// crops a chunk claiming one more workgroup along x than wgpu allows per dimension
struct OversizedCrop;

impl Transform for OversizedCrop {
    fn identity(&self) -> String {
        "OversizedCrop".into()
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Tensor(TensorPattern {
            domain: SetField::one(Domain::Wgpu),
            ..TensorPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, _out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        let too_wide = TensorChunk {
            channels: 1,
            bbox: Bbox {
                min_x: ORIGIN_X,
                max_x: ORIGIN_X + (8 * 65_536) as f64 * CELL,
                max_y: ORIGIN_Y,
                min_y: ORIGIN_Y - CELL,
            },
            ..input.tensor()?.clone()
        };
        Ok(Chunk::Tensor(too_wide.crop_to(&too_wide.bbox)?))
    }
}

#[tokio::test]
async fn an_oversized_dispatch_fails_its_own_pull() {
    let Some(context) = context(64 << 20) else {
        return;
    };
    let mut g = Graph::new();
    let cpu_nodes = small_tile_chain(&mut g, src(), Box::new(TensorConv { kernel: KERNEL }));
    let cpu = Engine::new(g, 64 << 20).unwrap();
    let mut g = Graph::with_gpu(context);
    let gpu_nodes = small_tile_chain(&mut g, src(), Box::new(WgpuConvolve { kernel: KERNEL }));
    let oversized = g.add_transform(gpu_nodes.convolve, Box::new(OversizedCrop));
    let gpu = Engine::new(g, 64 << 20).unwrap();

    match gpu.pull(oversized, seam_window_request()).await {
        Err(Error::Gpu(message)) => assert!(
            message.contains("exceed the device limit"),
            "the gpu error does not name the limit: {message}"
        ),
        Err(other) => panic!("expected a gpu error from the oversized dispatch, got {other:?}"),
        Ok(_) => panic!("a dispatch over the workgroup limit was recorded"),
    }

    let want = cpu
        .pull(cpu_nodes.raster, seam_window_request())
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let got = gpu
        .pull(gpu_nodes.raster, seam_window_request())
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    for c in 0..CHANNELS {
        let (a, b) = (got.bands.band(c).unwrap(), want.bands.band(c).unwrap());
        for (i, (x, y)) in a.data().iter().zip(b.data()).enumerate() {
            assert!(
                (x - y).abs() < 1e-5,
                "channel {c} cell {i}: wgpu {x} vs cpu {y}"
            );
        }
    }
}

fn spilled_files(dir: &Path) -> usize {
    let mut count = 0;
    for store in std::fs::read_dir(dir).unwrap() {
        count += std::fs::read_dir(store.unwrap().path()).unwrap().count();
    }
    count
}

#[tokio::test]
async fn gpu_chunks_count_against_their_own_budget_and_never_spill() {
    let chunk_bytes = CHANNELS * 256 * 256 * size_of::<f32>();
    let Some(context) = context(chunk_bytes) else {
        return;
    };
    // the spill dir persists, so each run gets its own base and removes it
    let spill_base =
        std::env::temp_dir().join(format!("geoplumb-test-{}-wgpu", std::process::id()));
    let (cpu_dir, gpu_dir) = (spill_base.join("cpu"), spill_base.join("gpu"));
    let req = WindowReq {
        bbox: window(200, 100, 300, 200),
        resolution: CELL,
        time: None,
    };

    let mut g = Graph::new();
    let cpu_nodes = chain(&mut g, src(), Box::new(TensorConv { kernel: KERNEL }));
    let cpu = Engine::with_disk_cache(g, 1 << 30, &cpu_dir, 1 << 30).unwrap();
    cpu.pull(cpu_nodes.raster, req).await.unwrap();

    let mut g = Graph::with_gpu(context);
    let gpu_nodes = chain(&mut g, src(), Box::new(WgpuConvolve { kernel: KERNEL }));
    let gpu = Engine::with_disk_cache(g, 1 << 30, &gpu_dir, 1 << 30).unwrap();
    gpu.pull(gpu_nodes.raster, req).await.unwrap();

    let gpu_bytes = gpu.gpu_bytes();
    let system_bytes = (gpu.system_bytes(), cpu.system_bytes());
    let files = (spilled_files(&gpu_dir), spilled_files(&cpu_dir));
    drop((cpu, gpu));
    let _ = std::fs::remove_dir_all(&spill_base);

    assert!(
        gpu_bytes > 0 && gpu_bytes <= chunk_bytes,
        "{gpu_bytes} gpu bytes cached against a {chunk_bytes} byte budget"
    );
    assert_eq!(
        system_bytes.0, system_bytes.1,
        "the system budget holds exactly the chunks the cpu chain holds"
    );
    assert!(files.1 > 0, "the cpu chain spilled nothing");
    assert_eq!(
        files.0, files.1,
        "the gpu chain spilled more files than its system chunks"
    );
}

#[test]
fn wgpu_demand_without_a_context_fails_the_solve() {
    let mut g = Graph::new();
    chain(&mut g, src(), Box::new(WgpuConvolve { kernel: KERNEL }));
    match Engine::new(g, 64 << 20) {
        Err(Error::EmptyLink { .. }) => {}
        Err(other) => panic!("expected EmptyLink, got {other:?}"),
        Ok(_) => panic!("a wgpu consumer must not solve without a context"),
    }
}
