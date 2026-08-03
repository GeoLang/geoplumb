//! the pull runtime. one map per engine serves as both cache and coalescing
//! table: a chunk is Ready (cached) or Pending (in flight), so concurrent
//! pulls of one chunk share a single computation. a cancelled computer's
//! guard removes its Pending entry and wakes waiters, one of which takes
//! over, so cancellation never wedges a chunk

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use tokio::sync::{broadcast, oneshot};

use crate::caps::Caps;
use crate::chunk::RasterChunk;
use crate::element::{Fanin, Source, Transform};
use crate::error::Result;
use crate::graph::{Graph, Node, NodeId};
use crate::solver;
use crate::window::{Bbox, ChunkKey, GridSpec, WindowReq};
use terrano_core::{BandedRaster, Raster};

enum RtElem {
    Source(Arc<dyn Source>),
    Transform {
        parent: usize,
        element: Arc<dyn Transform>,
    },
    Fanin {
        parents: Vec<usize>,
        element: Arc<dyn Fanin>,
    },
}

struct RtNode {
    elem: RtElem,
    caps: Caps,
    grid: GridSpec,
}

#[derive(Debug, Clone, Copy)]
pub struct Invalidation {
    pub node: NodeId,
    pub bbox: Bbox,
}

enum Entry {
    Ready {
        chunk: Arc<RasterChunk>,
        bytes: usize,
        last_used: u64,
    },
    Pending {
        waiters: Vec<oneshot::Sender<()>>,
    },
}

#[derive(Default)]
struct CacheState {
    entries: HashMap<(usize, ChunkKey), Entry>,
    bytes: usize,
    tick: u64,
}

pub struct Engine {
    nodes: Vec<RtNode>,
    topo: Vec<usize>,
    cache: Mutex<CacheState>,
    events: broadcast::Sender<Invalidation>,
    budget_bytes: usize,
}

impl Engine {
    /// solve the graph (splicing reprojects onto crs-mismatched edges),
    /// configure every transform with its fixated caps, and derive each
    /// node's grid from its source up. spliced nodes sit after their
    /// consumers in index order, so construction walks the topo order
    pub fn new(graph: Graph, budget_bytes: usize) -> Result<Engine> {
        let mut graph = graph;
        let caps = solver::solve(&mut graph)?;
        let topo = graph.topo_order();
        let mut boxes: Vec<Option<Node>> = graph.nodes.into_iter().map(Some).collect();
        let mut nodes: Vec<Option<RtNode>> = (0..boxes.len()).map(|_| None).collect();
        for &i in &topo {
            let built = |p: usize| -> &RtNode { nodes[p].as_ref().expect("parents built first") };
            let (elem, mut grid) = match boxes[i].take().expect("topo visits each node once") {
                Node::Source(s) => {
                    let grid = s.grid();
                    (RtElem::Source(Arc::from(s)), grid)
                }
                Node::Transform {
                    parent,
                    mut element,
                } => {
                    element.configure(&caps[parent.0], &caps[i])?;
                    let grid = element.output_grid(&built(parent.0).grid);
                    (
                        RtElem::Transform {
                            parent: parent.0,
                            element: Arc::from(element),
                        },
                        grid,
                    )
                }
                Node::Fanin {
                    parents,
                    mut element,
                } => {
                    let input_caps: Vec<Caps> = parents.iter().map(|p| caps[p.0].clone()).collect();
                    element.configure(&input_caps, &caps[i])?;
                    let grids: Vec<GridSpec> = parents.iter().map(|p| built(p.0).grid).collect();
                    let grid = element.output_grid(&grids);
                    (
                        RtElem::Fanin {
                            parents: parents.iter().map(|p| p.0).collect(),
                            element: Arc::from(element),
                        },
                        grid,
                    )
                }
            };
            grid.chunk_px = caps[i].raster().chunk_px;
            nodes[i] = Some(RtNode {
                elem,
                caps: caps[i].clone(),
                grid,
            });
        }
        let (events, _) = broadcast::channel(64);
        Ok(Engine {
            nodes: nodes.into_iter().map(|n| n.expect("all built")).collect(),
            topo,
            cache: Mutex::new(CacheState::default()),
            events,
            budget_bytes,
        })
    }

    pub fn caps(&self, node: NodeId) -> &Caps {
        &self.nodes[node.0].caps
    }

    pub fn grid(&self, node: NodeId) -> &GridSpec {
        &self.nodes[node.0].grid
    }

    pub fn subscribe(&self) -> broadcast::Receiver<Invalidation> {
        self.events.subscribe()
    }

    /// pull a window from a node: snap to the node's ladder, pull the
    /// covering chunks concurrently, mosaic, crop to the aligned window
    pub async fn pull(&self, node: NodeId, req: WindowReq) -> Result<RasterChunk> {
        self.pull_assembled(node.0, req).await
    }

    /// declare a window of a node dirty: drop overlapping cache entries at
    /// the node and its descendants, publish one event per affected node
    pub fn invalidate(&self, node: NodeId, bbox: Bbox) {
        let mut dirty: Vec<Option<Bbox>> = vec![None; self.nodes.len()];
        dirty[node.0] = Some(bbox);
        for &j in &self.topo {
            let n = &self.nodes[j];
            let from_parents: Vec<Bbox> = match &n.elem {
                RtElem::Source(_) => Vec::new(),
                RtElem::Transform { parent, .. } => dirty[*parent].into_iter().collect(),
                RtElem::Fanin { parents, .. } => parents.iter().filter_map(|p| dirty[*p]).collect(),
            };
            if !from_parents.is_empty() {
                // spread at the coarsest cached level so a coarse chunk's
                // wider halo is still covered
                let max_level = {
                    let state = self.cache.lock().unwrap();
                    state
                        .entries
                        .keys()
                        .filter(|(idx, _)| *idx == j)
                        .map(|(_, k)| k.level)
                        .max()
                        .unwrap_or(0)
                };
                let res = n.grid.resolution_at(max_level);
                for d in from_parents {
                    let spread = match &n.elem {
                        RtElem::Source(_) => unreachable!("sources have no parents"),
                        RtElem::Transform { element, .. } => element.spread(&d, res),
                        RtElem::Fanin { element, .. } => element.spread(&d, res),
                    };
                    dirty[j] = Some(match dirty[j] {
                        None => spread,
                        Some(prev) => union(prev, spread),
                    });
                }
            }
            let Some(d) = dirty[j] else { continue };
            let grid = n.grid;
            let mut state = self.cache.lock().unwrap();
            let mut freed = 0usize;
            state.entries.retain(|(n_idx, key), entry| {
                if *n_idx != j {
                    return true;
                }
                let keep = !grid.chunk_bbox(*key).intersects(&d);
                if !keep {
                    if let Entry::Ready { bytes, .. } = entry {
                        freed += *bytes;
                    }
                }
                keep
            });
            state_bytes_sub(&mut state.bytes, freed);
            drop(state);
            let _ = self.events.send(Invalidation {
                node: NodeId(j),
                bbox: d,
            });
        }
    }

    fn pull_assembled<'a>(
        &'a self,
        node: usize,
        req: WindowReq,
    ) -> BoxFuture<'a, Result<RasterChunk>> {
        Box::pin(async move {
            let grid = self.nodes[node].grid;
            let level = grid.snap_level(req.resolution);
            let res = grid.resolution_at(level);
            let aligned = align_outward(&req.bbox, &grid, res);
            let keys = grid.cover(&aligned, level);
            let chunks =
                futures::future::try_join_all(keys.iter().map(|k| self.chunk(node, *k))).await?;
            Ok(assemble(
                &grid,
                &keys,
                &chunks,
                &aligned,
                res,
                &self.nodes[node].caps,
            ))
        })
    }

    async fn chunk(&self, node: usize, key: ChunkKey) -> Result<Arc<RasterChunk>> {
        loop {
            let wait = {
                let mut state = self.cache.lock().unwrap();
                state.tick += 1;
                let tick = state.tick;
                match state.entries.get_mut(&(node, key)) {
                    Some(Entry::Ready {
                        chunk, last_used, ..
                    }) => {
                        *last_used = tick;
                        return Ok(chunk.clone());
                    }
                    Some(Entry::Pending { waiters }) => {
                        let (tx, rx) = oneshot::channel();
                        waiters.push(tx);
                        Some(rx)
                    }
                    None => {
                        state.entries.insert(
                            (node, key),
                            Entry::Pending {
                                waiters: Vec::new(),
                            },
                        );
                        None
                    }
                }
            };
            match wait {
                Some(rx) => {
                    // ok = computed, err = computer cancelled, retry either way
                    let _ = rx.await;
                }
                None => {
                    let guard = PendingGuard {
                        cache: &self.cache,
                        key: (node, key),
                        done: false,
                    };
                    let result = self.compute_chunk(node, key).await;
                    return self.finish_chunk(guard, result);
                }
            }
        }
    }

    fn finish_chunk(
        &self,
        mut guard: PendingGuard<'_>,
        result: Result<RasterChunk>,
    ) -> Result<Arc<RasterChunk>> {
        let chunk = Arc::new(result?);
        guard.done = true;
        let bytes = chunk.byte_size();
        let mut state = self.cache.lock().unwrap();
        state.tick += 1;
        let tick = state.tick;
        if let Some(Entry::Pending { waiters }) = state.entries.insert(
            guard.key,
            Entry::Ready {
                chunk: chunk.clone(),
                bytes,
                last_used: tick,
            },
        ) {
            for w in waiters {
                let _ = w.send(());
            }
        }
        state.bytes += bytes;
        evict_over_budget(&mut state, self.budget_bytes);
        Ok(chunk)
    }

    async fn compute_chunk(&self, node: usize, key: ChunkKey) -> Result<RasterChunk> {
        let grid = self.nodes[node].grid;
        let out = WindowReq {
            bbox: grid.chunk_bbox(key),
            resolution: grid.resolution_at(key.level),
        };
        match &self.nodes[node].elem {
            RtElem::Source(s) => s.read(&out).await,
            RtElem::Transform { parent, element } => {
                let in_req = element.plan(&out);
                let input = self.pull_assembled(*parent, in_req).await?;
                let element = element.clone();
                offload(move || element.compute(&out, &input)).await
            }
            RtElem::Fanin { parents, element } => {
                let inputs = futures::future::try_join_all(
                    parents
                        .iter()
                        .enumerate()
                        .map(|(k, p)| self.pull_assembled(*p, element.plan(&out, k))),
                )
                .await?;
                let element = element.clone();
                offload(move || element.compute(&out, &inputs)).await
            }
        }
    }
}

struct PendingGuard<'a> {
    cache: &'a Mutex<CacheState>,
    key: (usize, ChunkKey),
    done: bool,
}

impl Drop for PendingGuard<'_> {
    fn drop(&mut self) {
        if self.done {
            return;
        }
        let mut state = self.cache.lock().unwrap();
        if let Some(Entry::Pending { waiters }) = state.entries.remove(&self.key) {
            drop(waiters);
        }
    }
}

pub(crate) async fn offload<T: Send + 'static>(f: impl FnOnce() -> T + Send + 'static) -> T {
    match tokio::runtime::Handle::try_current() {
        Ok(h) => h.spawn_blocking(f).await.expect("compute task panicked"),
        Err(_) => f(),
    }
}

fn evict_over_budget(state: &mut CacheState, budget: usize) {
    while state.bytes > budget {
        let oldest = state
            .entries
            .iter()
            .filter_map(|(k, e)| match e {
                Entry::Ready { last_used, .. } => Some((*last_used, *k)),
                Entry::Pending { .. } => None,
            })
            .min();
        let Some((_, key)) = oldest else { break };
        if let Some(Entry::Ready { bytes, .. }) = state.entries.remove(&key) {
            state_bytes_sub(&mut state.bytes, bytes);
        }
    }
}

fn state_bytes_sub(total: &mut usize, bytes: usize) {
    *total = total.saturating_sub(bytes);
}

fn union(a: Bbox, b: Bbox) -> Bbox {
    Bbox {
        min_x: a.min_x.min(b.min_x),
        min_y: a.min_y.min(b.min_y),
        max_x: a.max_x.max(b.max_x),
        max_y: a.max_y.max(b.max_y),
    }
}

/// widen a bbox outward onto the node's pixel grid at the given resolution
fn align_outward(bbox: &Bbox, grid: &GridSpec, res: f64) -> Bbox {
    let eps = res * 1e-9;
    let min_x = grid.origin_x + (((bbox.min_x - grid.origin_x) / res + eps).floor()) * res;
    let max_x = grid.origin_x + (((bbox.max_x - grid.origin_x) / res - eps).ceil()) * res;
    let max_y = grid.origin_y - (((grid.origin_y - bbox.max_y) / res + eps).floor()) * res;
    let min_y = grid.origin_y - (((grid.origin_y - bbox.min_y) / res - eps).ceil()) * res;
    Bbox {
        min_x,
        min_y,
        max_x,
        max_y,
    }
}

/// mosaic same-level chunks into one raster covering `window`
fn assemble(
    grid: &GridSpec,
    keys: &[ChunkKey],
    chunks: &[Arc<RasterChunk>],
    window: &Bbox,
    res: f64,
    caps: &Caps,
) -> RasterChunk {
    let cols = (window.width() / res).round() as usize;
    let rows = (window.height() / res).round() as usize;
    let band_count = chunks
        .first()
        .map(|c| c.bands.band_count())
        .unwrap_or(usize::from(caps.raster().bands));
    let nodata = chunks
        .first()
        .and_then(|c| c.bands.band(0).map(|b| b.nodata))
        .unwrap_or(f64::NAN);
    let mut bands: Vec<Vec<f64>> = vec![vec![nodata; cols * rows]; band_count];
    for (key, chunk) in keys.iter().zip(chunks) {
        let cb = grid.chunk_bbox(*key);
        for (bi, band) in chunk.bands.bands().iter().enumerate() {
            let w = chunk.width();
            for row in 0..chunk.height() {
                let y = cb.max_y - (row as f64 + 0.5) * res;
                if y > window.max_y || y < window.min_y {
                    continue;
                }
                let out_row = ((window.max_y - y) / res).floor() as usize;
                if out_row >= rows {
                    continue;
                }
                for col in 0..w {
                    let x = cb.min_x + (col as f64 + 0.5) * res;
                    if x < window.min_x || x > window.max_x {
                        continue;
                    }
                    let out_col = ((x - window.min_x) / res).floor() as usize;
                    if out_col < cols {
                        bands[bi][out_row * cols + out_col] = band.data()[row * w + col];
                    }
                }
            }
        }
    }
    let rasters: Vec<Raster> = bands
        .into_iter()
        .map(|d| Raster::from_vec(cols, rows, d, res, nodata).expect("assemble dims"))
        .collect();
    RasterChunk {
        bands: BandedRaster::new(rasters).expect("assemble bands uniform"),
        bbox: *window,
        resolution: res,
        crs: caps.raster().crs,
    }
}

/// batch driver: pull every chunk of `node` covering `extent` at each ladder
/// level up to `max_level`, the full-extent materialization loop
pub async fn materialize(
    engine: &Engine,
    node: NodeId,
    extent: Bbox,
    max_level: u8,
    mut per_chunk: impl FnMut(ChunkKey, &RasterChunk),
) -> Result<usize> {
    let grid = *engine.grid(node);
    let mut count = 0;
    for level in 0..=max_level {
        let res = grid.resolution_at(level);
        for key in grid.cover(&extent, level) {
            let chunk = engine
                .pull(
                    node,
                    WindowReq {
                        bbox: grid.chunk_bbox(key),
                        resolution: res,
                    },
                )
                .await?;
            per_chunk(key, &chunk);
            count += 1;
        }
    }
    Ok(count)
}
