//! windowed cog source. each pull fetches only the tiles it touches
//! through terrano's CogReader, served from the file overview nearest the
//! requested ladder level. when the file pyramid is shallower than the
//! request the remainder is block-averaged, matching RasterSrc semantics

use std::collections::HashMap;
use std::sync::{Arc, Condvar, LazyLock, Mutex};

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::error::{Error, Result};
use crate::window::{GridSpec, WindowReq};
use futures::future::BoxFuture;
use terrano_core::{BandedRaster, CogReader, RangeRead, Raster};

pub struct CogSrc<R: RangeRead + Send + 'static> {
    // read_window_bands needs &mut, so concurrent chunk reads serialize here
    reader: Arc<Mutex<CogReader<R>>>,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    bands: u16,
    crs: Crs,
}

impl<R: RangeRead + Send + 'static> CogSrc<R> {
    /// reads the file layout with blocking range requests, call it off
    /// the async runtime (`spawn_blocking`) when one is running
    pub fn open(source: R) -> Result<Self> {
        let reader = CogReader::open(source)?;
        let meta = reader.meta().clone();
        let bands = band_count(&reader)?;
        Ok(CogSrc {
            origin_x: meta.origin_x,
            origin_y: meta.origin_y,
            base_resolution: meta.pixel_width,
            bands,
            crs: Crs(u32::from(meta.epsg)),
            reader: Arc::new(Mutex::new(reader)),
        })
    }

    /// bands the file carries, every level alike
    pub fn bands(&self) -> u16 {
        self.bands
    }
}

/// samples per pixel at the base level, which every level shares
pub(crate) fn band_count<R: RangeRead>(reader: &CogReader<R>) -> Result<u16> {
    let samples = reader
        .levels()
        .first()
        .ok_or_else(|| Error::Source("cog has no levels".into()))?
        .samples;
    u16::try_from(samples).map_err(|_| Error::Source(format!("cog has {samples} bands")))
}

impl<R: RangeRead + Send + 'static> Source for CogSrc<R> {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtype: SetField::one(Dtype::F64),
            bands: SetField::one(self.bands),
            crs: SetField::one(self.crs),
            resolution: ResRange::at_least(self.base_resolution),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            base_resolution: self.base_resolution,
            chunk_px: 256,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let reader = self.reader.clone();
        let req = *req;
        let (origin_x, origin_y, crs) = (self.origin_x, self.origin_y, self.crs);
        Box::pin(async move {
            crate::engine::offload(move || {
                read_chunk(&mut reader.lock().unwrap(), &req, origin_x, origin_y, crs)
                    .map(Chunk::Raster)
            })
            .await
        })
    }
}

pub(crate) fn read_chunk<R: RangeRead>(
    reader: &mut CogReader<R>,
    req: &WindowReq,
    origin_x: f64,
    origin_y: f64,
    crs: Crs,
) -> Result<RasterChunk> {
    let level = reader.select_level(req.resolution);
    let lres = reader.levels()[level].pixel_width;
    let samples = reader.levels()[level].samples;
    let factor = (req.resolution / lres).round().max(1.0) as usize;
    let cols = (req.bbox.width() / req.resolution).round() as usize;
    let rows = (req.bbox.height() / req.resolution).round() as usize;
    let (fcols, frows) = (cols * factor, rows * factor);
    let col0 = ((req.bbox.min_x - origin_x) / lres).round() as i64;
    let row0 = ((origin_y - req.bbox.max_y) / lres).round() as i64;

    // read_window_bands pads right/bottom itself but cannot start left or
    // above the image, so clamp the start and copy at an offset
    let mut fine = vec![vec![f64::NAN; fcols * frows]; samples];
    let (skip_c, skip_r) = ((-col0).max(0) as usize, (-row0).max(0) as usize);
    if skip_c < fcols && skip_r < frows {
        let window = reader.read_window_bands(
            level,
            (col0 + skip_c as i64) as usize,
            (row0 + skip_r as i64) as usize,
            fcols - skip_c,
            frows - skip_r,
        )?;
        for (plane, band) in fine.iter_mut().zip(window.bands()) {
            let w = band.width();
            for r in 0..band.height() {
                let dst = (skip_r + r) * fcols + skip_c;
                plane[dst..dst + w].copy_from_slice(&band.data()[r * w..(r + 1) * w]);
            }
        }
    }

    let bands: Vec<Raster> = fine
        .into_iter()
        .map(|plane| {
            let data = if factor == 1 {
                plane
            } else {
                decimate(&plane, cols, rows, factor)
            };
            Raster::from_vec(cols, rows, data, req.resolution, f64::NAN).expect("window dims")
        })
        .collect();
    Ok(RasterChunk {
        bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
        bbox: req.bbox,
        resolution: req.resolution,
        crs,
    })
}

fn decimate(fine: &[f64], cols: usize, rows: usize, factor: usize) -> Vec<f64> {
    let fcols = cols * factor;
    let mut out = vec![f64::NAN; cols * rows];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            let mut n = 0usize;
            for rr in 0..factor {
                for cc in 0..factor {
                    let v = fine[(row * factor + rr) * fcols + col * factor + cc];
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out[row * cols + col] = sum / n as f64;
            }
        }
    }
    out
}

/// range-request transport for remote cogs. blocking to its caller, so
/// it must run off the async runtime, which `CogSrc::read` already does,
/// but fetches run multiplexed on a small dedicated runtime: a
/// `read_ranges` call fetches all its misses concurrently. transient
/// faults (send errors, 5xx, 429, short or failed bodies) retry with
/// backoff
pub struct HttpRange {
    url: String,
}

const RANGE_ATTEMPTS: u32 = 3;
const RANGE_BACKOFF: std::time::Duration = std::time::Duration::from_millis(200);

/// transfers in flight across the process, however many readers are
/// mid-read. queued transfers wait on a semaphore, not a thread
const MAX_PARALLEL_TRANSFERS: usize = 48;

static FETCH_RT: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .thread_name("range-fetch")
        .enable_all()
        .build()
        .expect("fetch runtime never fails to build")
});

static FETCH_CLIENT: LazyLock<reqwest::Client> = LazyLock::new(reqwest::Client::new);

static TRANSFER_PERMITS: LazyLock<tokio::sync::Semaphore> =
    LazyLock::new(|| tokio::sync::Semaphore::new(MAX_PARALLEL_TRANSFERS));

/// one fetch attempt, the bool says whether the failure is transient
async fn fetch_once(
    url: &str,
    offset: u64,
    len: u64,
) -> core::result::Result<Vec<u8>, (bool, String)> {
    let end = offset + len - 1;
    let resp = FETCH_CLIENT
        .get(url)
        .header(reqwest::header::RANGE, format!("bytes={offset}-{end}"))
        .send()
        .await
        .map_err(|e| (true, format!("range request failed: {e}")))?;
    let status = resp.status();
    if status.is_client_error() || status.is_server_error() {
        let transient =
            status.is_server_error() || status == reqwest::StatusCode::TOO_MANY_REQUESTS;
        return Err((transient, format!("range request failed: status {status}")));
    }
    if status != reqwest::StatusCode::PARTIAL_CONTENT {
        return Err((
            false,
            format!("server ignored the range header (status {status})"),
        ));
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| (true, format!("range body read failed: {e}")))?;
    if bytes.len() as u64 != len {
        return Err((
            true,
            format!("range returned {} bytes, wanted {len}", bytes.len()),
        ));
    }
    Ok(bytes.to_vec())
}

/// retried fetch under a transfer permit
async fn fetch_range(
    url: &str,
    offset: u64,
    len: u64,
) -> core::result::Result<Vec<u8>, terrano_core::Error> {
    let _permit = TRANSFER_PERMITS
        .acquire()
        .await
        .expect("semaphore is never closed");
    let mut attempts = 0;
    let mut delay = RANGE_BACKOFF;
    loop {
        match fetch_once(url, offset, len).await {
            Ok(bytes) => return Ok(bytes),
            Err((transient, detail)) => {
                attempts += 1;
                if !transient || attempts == RANGE_ATTEMPTS {
                    return Err(terrano_core::Error::Format(detail));
                }
                tokio::time::sleep(delay).await;
                delay *= 4;
            }
        }
    }
}

/// process-wide budget for cached range bytes, compressed as fetched
const RANGE_CACHE_BUDGET: usize = 256 << 20;

type RangeKey = (String, u64, u64);

enum RangeSlot {
    /// a fetch is in flight, waiters sleep on the condvar
    Pending,
    Ready {
        bytes: Arc<Vec<u8>>,
        used: u64,
    },
}

/// what `probe` found without waiting
enum Probe {
    Ready(Arc<Vec<u8>>),
    /// the key is now claimed pending, the caller must fetch and publish
    Claimed,
    /// another caller's fetch is in flight
    InFlight,
}

struct RangeState {
    slots: HashMap<RangeKey, RangeSlot>,
    bytes: usize,
    /// bumps per touch, so eviction can drop the least recently used
    tick: u64,
}

/// cog readers fetch an interior tile as the same exact byte range every
/// time, so the concurrent chunks of a pull that touch the same tile
/// dedup here instead of re-fetching it. misses are single-flight: the
/// first caller fetches while the rest wait, and an abandoned claim (the
/// fetch failed) wakes a waiter to become the fetcher
struct RangeCache {
    state: Mutex<RangeState>,
    done: Condvar,
}

static RANGE_CACHE: LazyLock<RangeCache> = LazyLock::new(|| RangeCache {
    state: Mutex::new(RangeState {
        slots: HashMap::new(),
        bytes: 0,
        tick: 0,
    }),
    done: Condvar::new(),
});

impl RangeCache {
    /// like `get_or_claim` but never waits on someone else's fetch, so a
    /// caller can collect its own claims without holding them while
    /// blocked, which is what would deadlock two callers over one url
    fn probe(&self, key: &RangeKey) -> Probe {
        let mut st = self.state.lock().unwrap();
        st.tick += 1;
        let tick = st.tick;
        match st.slots.get_mut(key) {
            Some(RangeSlot::Ready { bytes, used }) => {
                *used = tick;
                Probe::Ready(bytes.clone())
            }
            Some(RangeSlot::Pending) => Probe::InFlight,
            None => {
                st.slots.insert(key.clone(), RangeSlot::Pending);
                Probe::Claimed
            }
        }
    }

    /// cached bytes, or `None` with the key claimed pending: the caller
    /// must fetch and `publish` the result
    fn get_or_claim(&self, key: &RangeKey) -> Option<Arc<Vec<u8>>> {
        let mut st = self.state.lock().unwrap();
        loop {
            st.tick += 1;
            let tick = st.tick;
            match st.slots.get_mut(key) {
                Some(RangeSlot::Ready { bytes, used }) => {
                    *used = tick;
                    return Some(bytes.clone());
                }
                Some(RangeSlot::Pending) => st = self.done.wait(st).unwrap(),
                None => {
                    st.slots.insert(key.clone(), RangeSlot::Pending);
                    return None;
                }
            }
        }
    }

    fn publish(&self, key: &RangeKey, fetched: Option<Arc<Vec<u8>>>) {
        let mut st = self.state.lock().unwrap();
        match fetched {
            Some(bytes) => {
                st.bytes += bytes.len();
                st.tick += 1;
                let used = st.tick;
                st.slots
                    .insert(key.clone(), RangeSlot::Ready { bytes, used });
                while st.bytes > RANGE_CACHE_BUDGET {
                    let oldest = st
                        .slots
                        .iter()
                        .filter_map(|(k, s)| match s {
                            RangeSlot::Ready { bytes, used } => {
                                Some((*used, k.clone(), bytes.len()))
                            }
                            RangeSlot::Pending => None,
                        })
                        .min_by_key(|(used, _, _)| *used);
                    let Some((_, k, n)) = oldest else { break };
                    st.slots.remove(&k);
                    st.bytes -= n;
                }
            }
            None => {
                st.slots.remove(key);
            }
        }
        drop(st);
        self.done.notify_all();
    }
}

impl HttpRange {
    pub fn new(url: impl Into<String>) -> Self {
        HttpRange { url: url.into() }
    }

    /// fetch one claimed range on the fetch runtime and publish it
    fn fetch_claimed(&self, key: &RangeKey) -> core::result::Result<Vec<u8>, terrano_core::Error> {
        match FETCH_RT.block_on(fetch_range(&self.url, key.1, key.2)) {
            Ok(bytes) => {
                let bytes = Arc::new(bytes);
                RANGE_CACHE.publish(key, Some(bytes.clone()));
                Ok((*bytes).clone())
            }
            Err(e) => {
                RANGE_CACHE.publish(key, None);
                Err(e)
            }
        }
    }
}

impl RangeRead for HttpRange {
    fn read_range(
        &mut self,
        offset: u64,
        len: u64,
    ) -> core::result::Result<Vec<u8>, terrano_core::Error> {
        Ok(self.read_ranges(&[(offset, len)])?.remove(0))
    }

    /// misses fetch concurrently on the dedicated runtime. claims are
    /// held only while their fetches run, and waiting on another
    /// caller's in-flight fetch happens after every own claim has been
    /// published, so callers cannot deadlock on each other's claims
    fn read_ranges(
        &mut self,
        ranges: &[(u64, u64)],
    ) -> core::result::Result<Vec<Vec<u8>>, terrano_core::Error> {
        let mut out: Vec<Option<Vec<u8>>> = vec![None; ranges.len()];
        let mut mine = Vec::new();
        let mut theirs = Vec::new();
        for (i, &(offset, len)) in ranges.iter().enumerate() {
            let key = (self.url.clone(), offset, len);
            match RANGE_CACHE.probe(&key) {
                Probe::Ready(bytes) => out[i] = Some((*bytes).clone()),
                Probe::Claimed => mine.push(i),
                Probe::InFlight => theirs.push(i),
            }
        }

        let mut first_err = None;
        if !mine.is_empty() {
            let url = &self.url;
            let fetched = FETCH_RT.block_on(futures::future::join_all(mine.iter().map(|&i| {
                let (offset, len) = ranges[i];
                async move { (i, fetch_range(url, offset, len).await) }
            })));
            for (i, result) in fetched {
                let (offset, len) = ranges[i];
                let key = (self.url.clone(), offset, len);
                match result {
                    Ok(bytes) => {
                        let bytes = Arc::new(bytes);
                        RANGE_CACHE.publish(&key, Some(bytes.clone()));
                        out[i] = Some((*bytes).clone());
                    }
                    Err(e) => {
                        RANGE_CACHE.publish(&key, None);
                        first_err.get_or_insert(e);
                    }
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }

        for i in theirs {
            let (offset, len) = ranges[i];
            let key = (self.url.clone(), offset, len);
            out[i] = Some(match RANGE_CACHE.get_or_claim(&key) {
                Some(bytes) => (*bytes).clone(),
                // the fetch that was in flight failed, this caller takes over
                None => self.fetch_claimed(&key)?,
            });
        }

        Ok(out
            .into_iter()
            .map(|o| o.expect("every range resolved"))
            .collect())
    }
}
