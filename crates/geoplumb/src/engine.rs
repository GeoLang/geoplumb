//! the pull runtime. one map per engine serves as both cache and coalescing
//! table: a chunk is Ready (cached) or Pending (in flight), so concurrent
//! pulls of one chunk share a single computation. a cancelled computer's
//! guard removes its Pending entry and wakes waiters, one of which takes
//! over, so cancellation never wedges a chunk

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use tokio::sync::{broadcast, oneshot};

use crate::caps::{Caps, TensorCaps};
use crate::chunk::{
    Chunk, PointChunk, RasterChunk, TensorChunk, VectorChunk, VectorFeature, clip_geometry,
    tile_contains,
};
use crate::element::{Fanin, Source, Transform};
use crate::error::Result;
use crate::graph::{Graph, Node, NodeId};
use crate::solver;
use crate::spill::{self, SpillStore};
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
        chunk: Arc<Chunk>,
        bytes: usize,
        last_used: u64,
        /// a copy of this chunk sits in the spill store, so memory
        /// eviction demotes to `Spilled` instead of dropping
        spilled: bool,
    },
    /// on disk only, reloaded through the pending machinery on a hit
    Spilled {
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
    disk_bytes: usize,
    tick: u64,
}

struct SpillState {
    store: SpillStore,
    budget_bytes: usize,
}

pub struct Engine {
    nodes: Vec<RtNode>,
    topo: Vec<usize>,
    cache: Mutex<CacheState>,
    events: broadcast::Sender<Invalidation>,
    budget_bytes: usize,
    spill: Option<SpillState>,
}

impl Engine {
    /// solve the graph (splicing reprojects onto crs-mismatched edges),
    /// configure every transform with its fixated caps, and derive each
    /// node's grid from its source up. spliced nodes sit after their
    /// consumers in index order, so construction walks the topo order
    pub fn new(graph: Graph, budget_bytes: usize) -> Result<Engine> {
        Engine::build(graph, budget_bytes, None)
    }

    /// like `new`, with evicted chunks demoted to a disk tier instead of
    /// dropped. the engine owns a fresh subdir of `dir` and removes it on
    /// drop, entries never outlive the engine that wrote them
    pub fn with_disk_cache(
        graph: Graph,
        budget_bytes: usize,
        dir: impl AsRef<std::path::Path>,
        disk_budget_bytes: usize,
    ) -> Result<Engine> {
        let store = SpillStore::create(dir.as_ref())?;
        Engine::build(
            graph,
            budget_bytes,
            Some(SpillState {
                store,
                budget_bytes: disk_budget_bytes,
            }),
        )
    }

    fn build(graph: Graph, budget_bytes: usize, spill: Option<SpillState>) -> Result<Engine> {
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
            grid.chunk_px = caps[i].chunk_px();
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
            spill,
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
    pub async fn pull(&self, node: NodeId, req: WindowReq) -> Result<Chunk> {
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
            let mut freed_disk = 0usize;
            let mut deletions = Vec::new();
            state.entries.retain(|(n_idx, key), entry| {
                if *n_idx != j {
                    return true;
                }
                let keep = !grid.chunk_bbox(*key).intersects(&d);
                if !keep {
                    match entry {
                        Entry::Ready { bytes, spilled, .. } => {
                            freed += *bytes;
                            if *spilled {
                                deletions.push((*n_idx, *key));
                            }
                        }
                        Entry::Spilled { bytes, .. } => {
                            freed_disk += *bytes;
                            deletions.push((*n_idx, *key));
                        }
                        Entry::Pending { .. } => {}
                    }
                }
                keep
            });
            state_bytes_sub(&mut state.bytes, freed);
            state_bytes_sub(&mut state.disk_bytes, freed_disk);
            drop(state);
            self.delete_spill_files(&deletions);
            let _ = self.events.send(Invalidation {
                node: NodeId(j),
                bbox: d,
            });
        }
    }

    fn pull_assembled<'a>(&'a self, node: usize, req: WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        Box::pin(async move {
            let grid = self.nodes[node].grid;
            let level = grid.snap_level(req.resolution);
            let res = grid.resolution_at(level);
            let aligned = align_outward(&req.bbox, &grid, res);
            let keys = grid.cover(&aligned, level);
            let chunks =
                futures::future::try_join_all(keys.iter().map(|k| self.chunk(node, *k))).await?;
            assemble(&grid, &keys, &chunks, &aligned, res, &self.nodes[node].caps)
        })
    }

    async fn chunk(&self, node: usize, key: ChunkKey) -> Result<Arc<Chunk>> {
        enum Action {
            Wait(oneshot::Receiver<()>),
            Compute,
            Load,
        }
        loop {
            let action = {
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
                        Action::Wait(rx)
                    }
                    Some(Entry::Spilled { bytes, .. }) => {
                        let bytes = *bytes;
                        state.entries.insert(
                            (node, key),
                            Entry::Pending {
                                waiters: Vec::new(),
                            },
                        );
                        state_bytes_sub(&mut state.disk_bytes, bytes);
                        Action::Load
                    }
                    None => {
                        state.entries.insert(
                            (node, key),
                            Entry::Pending {
                                waiters: Vec::new(),
                            },
                        );
                        Action::Compute
                    }
                }
            };
            let guard = |done| PendingGuard {
                cache: &self.cache,
                key: (node, key),
                done,
            };
            match action {
                Action::Wait(rx) => {
                    // ok = computed, err = computer cancelled, retry either way
                    let _ = rx.await;
                }
                Action::Load => {
                    let guard = guard(false);
                    let store = &self
                        .spill
                        .as_ref()
                        .expect("spilled entries need a store")
                        .store;
                    let path = store.path(node, key);
                    let read = path.clone();
                    return match offload(move || spill::read_chunk(&read)).await {
                        Ok(chunk) => self.finish_chunk(guard, Ok(chunk), true),
                        Err(_) => {
                            let _ = std::fs::remove_file(&path);
                            let result = self.compute_chunk(node, key).await;
                            self.finish_and_spill(guard, result, node, key).await
                        }
                    };
                }
                Action::Compute => {
                    let guard = guard(false);
                    let result = self.compute_chunk(node, key).await;
                    return self.finish_and_spill(guard, result, node, key).await;
                }
            }
        }
    }

    /// cache a fresh chunk and write it through to the disk tier, marking
    /// the entry spilled only once the file is safely on disk
    async fn finish_and_spill(
        &self,
        guard: PendingGuard<'_>,
        result: Result<Chunk>,
        node: usize,
        key: ChunkKey,
    ) -> Result<Arc<Chunk>> {
        let chunk = self.finish_chunk(guard, result, false)?;
        if let Some(sp) = &self.spill {
            let path = sp.store.path(node, key);
            let c = chunk.clone();
            if offload(move || spill::write_chunk(&path, &c)).await.is_ok() {
                let mut state = self.cache.lock().unwrap();
                if let Some(Entry::Ready { spilled, .. }) = state.entries.get_mut(&(node, key)) {
                    *spilled = true;
                }
                // entry gone: the orphan file gets overwritten on the next
                // compute of this key, the store dir dies with the engine
            }
        }
        Ok(chunk)
    }

    fn finish_chunk(
        &self,
        mut guard: PendingGuard<'_>,
        result: Result<Chunk>,
        spilled: bool,
    ) -> Result<Arc<Chunk>> {
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
                spilled,
            },
        ) {
            for w in waiters {
                let _ = w.send(());
            }
        }
        state.bytes += bytes;
        let deletions = evict_over_budget(
            &mut state,
            self.budget_bytes,
            self.spill.as_ref().map(|s| s.budget_bytes),
        );
        drop(state);
        self.delete_spill_files(&deletions);
        Ok(chunk)
    }

    fn delete_spill_files(&self, keys: &[(usize, ChunkKey)]) {
        if let Some(sp) = &self.spill {
            for &(node, key) in keys {
                let _ = std::fs::remove_file(sp.store.path(node, key));
            }
        }
    }

    async fn compute_chunk(&self, node: usize, key: ChunkKey) -> Result<Chunk> {
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

/// returns the keys whose spill files must be deleted, for the caller to
/// remove outside the cache lock
fn evict_over_budget(
    state: &mut CacheState,
    budget: usize,
    disk_budget: Option<usize>,
) -> Vec<(usize, ChunkKey)> {
    let mut deletions = Vec::new();
    while state.bytes > budget {
        let oldest = state
            .entries
            .iter()
            .filter_map(|(k, e)| match e {
                Entry::Ready { last_used, .. } => Some((*last_used, *k)),
                Entry::Spilled { .. } | Entry::Pending { .. } => None,
            })
            .min();
        let Some((_, key)) = oldest else { break };
        let Some(Entry::Ready {
            bytes,
            last_used,
            spilled,
            ..
        }) = state.entries.remove(&key)
        else {
            unreachable!("picked from ready entries");
        };
        state_bytes_sub(&mut state.bytes, bytes);
        if spilled && disk_budget.is_some() {
            state
                .entries
                .insert(key, Entry::Spilled { bytes, last_used });
            state.disk_bytes += bytes;
        } else if spilled {
            deletions.push(key);
        }
    }
    if let Some(disk_budget) = disk_budget {
        while state.disk_bytes > disk_budget {
            let oldest = state
                .entries
                .iter()
                .filter_map(|(k, e)| match e {
                    Entry::Spilled { last_used, .. } => Some((*last_used, *k)),
                    Entry::Ready { .. } | Entry::Pending { .. } => None,
                })
                .min();
            let Some((_, key)) = oldest else { break };
            if let Some(Entry::Spilled { bytes, .. }) = state.entries.remove(&key) {
                state_bytes_sub(&mut state.disk_bytes, bytes);
                deletions.push(key);
            }
        }
    }
    deletions
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

/// stitch same-level chunks into one chunk covering `window`, per kind
fn assemble(
    grid: &GridSpec,
    keys: &[ChunkKey],
    chunks: &[Arc<Chunk>],
    window: &Bbox,
    res: f64,
    caps: &Caps,
) -> Result<Chunk> {
    match caps {
        Caps::Raster(_) => {
            let rasters: Vec<&RasterChunk> =
                chunks.iter().map(|c| c.raster()).collect::<Result<_>>()?;
            Ok(Chunk::Raster(assemble_raster(
                grid, keys, &rasters, window, res, caps,
            )))
        }
        Caps::PointCloud(p) => {
            let mut pts = Vec::new();
            for chunk in chunks {
                let pc = chunk.points()?;
                pts.extend(
                    pc.points
                        .points()
                        .iter()
                        .filter(|pt| tile_contains(window, pt.x, pt.y))
                        .copied(),
                );
            }
            Ok(Chunk::PointCloud(PointChunk {
                points: nubis_core::PointCloud::from_points(pts),
                bbox: *window,
                resolution: res,
                crs: p.crs,
            }))
        }
        Caps::Vector(v) => {
            let mut features = Vec::new();
            for chunk in chunks {
                let vc = chunk.vector()?;
                for f in &vc.features {
                    for geometry in clip_geometry(&f.geometry, window) {
                        features.push(VectorFeature {
                            id: f.id,
                            geometry,
                            properties: f.properties.clone(),
                        });
                    }
                }
            }
            // stable by id so burn order downstream matches source order
            // even when a feature's fragments arrive from different tiles
            features.sort_by_key(|f| f.id);
            Ok(Chunk::Vector(VectorChunk::new(
                features, *window, res, v.crs,
            )))
        }
        Caps::Tensor(t) => {
            let tensors: Vec<&TensorChunk> =
                chunks.iter().map(|c| c.tensor()).collect::<Result<_>>()?;
            Ok(Chunk::Tensor(assemble_tensor(
                grid, keys, &tensors, window, res, t,
            )))
        }
    }
}

/// mosaic same-level raster chunks into one raster covering `window`
fn assemble_raster(
    grid: &GridSpec,
    keys: &[ChunkKey],
    chunks: &[&RasterChunk],
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

/// mosaic same-level tensor chunks into one tensor covering `window`
fn assemble_tensor(
    grid: &GridSpec,
    keys: &[ChunkKey],
    chunks: &[&TensorChunk],
    window: &Bbox,
    res: f64,
    caps: &TensorCaps,
) -> TensorChunk {
    let cols = (window.width() / res).round() as usize;
    let rows = (window.height() / res).round() as usize;
    let channels = chunks.first().map_or(caps.channels, |c| c.channels) as usize;
    let mut data = vec![f32::NAN; channels * cols * rows];
    for (key, chunk) in keys.iter().zip(chunks) {
        let cb = grid.chunk_bbox(*key);
        let (w, h) = (chunk.width(), chunk.height());
        for row in 0..h {
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
                if out_col >= cols {
                    continue;
                }
                for c in 0..channels {
                    data[c * cols * rows + out_row * cols + out_col] =
                        chunk.data[c * w * h + row * w + col];
                }
            }
        }
    }
    TensorChunk {
        data,
        channels: channels as u16,
        bbox: *window,
        resolution: res,
        crs: caps.crs,
    }
}

/// batch driver: pull every chunk of `node` covering `extent` at each ladder
/// level up to `max_level`, the full-extent materialization loop
pub async fn materialize(
    engine: &Engine,
    node: NodeId,
    extent: Bbox,
    max_level: u8,
    mut per_chunk: impl FnMut(ChunkKey, &Chunk),
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
