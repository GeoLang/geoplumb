//! window requests, the resolution ladder, and chunk snapping.
//! a pull names a bbox and a target resolution. the engine snaps it to a
//! node's grid: a power-of-two resolution ladder over the node's base
//! resolution, tiled from the grid origin, so cache keys stay discrete.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bbox {
    pub min_x: f64,
    pub min_y: f64,
    pub max_x: f64,
    pub max_y: f64,
}

impl Bbox {
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Self {
        Bbox {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    pub fn width(&self) -> f64 {
        self.max_x - self.min_x
    }

    pub fn height(&self) -> f64 {
        self.max_y - self.min_y
    }

    pub fn expand(&self, margin: f64) -> Bbox {
        Bbox {
            min_x: self.min_x - margin,
            min_y: self.min_y - margin,
            max_x: self.max_x + margin,
            max_y: self.max_y + margin,
        }
    }

    pub fn intersects(&self, other: &Bbox) -> bool {
        self.min_x < other.max_x
            && other.min_x < self.max_x
            && self.min_y < other.max_y
            && other.min_y < self.max_y
    }
}

/// half-open utc instant range in epoch milliseconds: `start_ms` is in,
/// `end_ms` is out. the engine core holds only numbers, rfc 3339 parsing
/// and formatting live with the source that speaks it
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TimeInterval {
    pub start_ms: i64,
    pub end_ms: i64,
}

impl TimeInterval {
    pub fn new(start_ms: i64, end_ms: i64) -> TimeInterval {
        TimeInterval { start_ms, end_ms }
    }
}

/// a pull: give me this window at this ground resolution, optionally at
/// this instant
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowReq {
    pub bbox: Bbox,
    pub resolution: f64,
    /// `None` leaves a time-varying source on its own configured
    /// interval, `Some` overrides it for this pull
    pub time: Option<TimeInterval>,
}

impl WindowReq {
    /// the same pull over another window: how a `plan` rewrites its
    /// upstream request, so the pull's time rides along untouched
    pub fn with_window(&self, bbox: Bbox, resolution: f64) -> WindowReq {
        WindowReq {
            bbox,
            resolution,
            time: self.time,
        }
    }

    pub fn with_time(&self, time: Option<TimeInterval>) -> WindowReq {
        WindowReq { time, ..*self }
    }
}

/// a node's native pixel grid, anchor of its resolution ladder.
/// y grows downward from the origin, the raster convention
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridSpec {
    pub origin_x: f64,
    pub origin_y: f64,
    pub base_resolution: f64,
    pub chunk_px: u32,
}

/// cache-addressable unit: ladder level plus tile indices in the node
/// grid, plus the pull time at nodes whose data varies with it
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey {
    pub level: u8,
    pub ix: i64,
    pub iy: i64,
    /// `None` at a node nothing upstream resolves per pull time, so a
    /// time-invariant graph keeps one entry per tile however many
    /// instants are pulled
    pub time: Option<TimeInterval>,
}

impl ChunkKey {
    pub fn at_time(&self, time: Option<TimeInterval>) -> ChunkKey {
        ChunkKey { time, ..*self }
    }
}

pub const MAX_LEVEL: u8 = 31;

impl GridSpec {
    pub fn resolution_at(&self, level: u8) -> f64 {
        self.base_resolution * f64::from(1u32 << level)
    }

    /// finest level whose resolution is coarse enough to be sampled from,
    /// yet not coarser than the request: the finest level >= requested
    /// quality is level floor(log2(req/base)), clamped to the ladder
    pub fn snap_level(&self, resolution: f64) -> u8 {
        let ratio = resolution / self.base_resolution;
        if ratio <= 1.0 {
            return 0;
        }
        (ratio.log2().floor() as u32).min(u32::from(MAX_LEVEL)) as u8
    }

    pub fn chunk_ground_size(&self, level: u8) -> f64 {
        self.resolution_at(level) * f64::from(self.chunk_px)
    }

    pub fn chunk_bbox(&self, key: ChunkKey) -> Bbox {
        let size = self.chunk_ground_size(key.level);
        Bbox {
            min_x: self.origin_x + key.ix as f64 * size,
            max_x: self.origin_x + (key.ix + 1) as f64 * size,
            max_y: self.origin_y - key.iy as f64 * size,
            min_y: self.origin_y - (key.iy + 1) as f64 * size,
        }
    }

    /// tile index bounds covering the bbox at a level, inclusive
    pub(crate) fn tile_range(&self, bbox: &Bbox, level: u8) -> (i64, i64, i64, i64) {
        let size = self.chunk_ground_size(level);
        let eps = size * 1e-9;
        let ix0 = ((bbox.min_x - self.origin_x + eps) / size).floor() as i64;
        let ix1 = ((bbox.max_x - self.origin_x - eps) / size).floor() as i64;
        let iy0 = ((self.origin_y - bbox.max_y + eps) / size).floor() as i64;
        let iy1 = ((self.origin_y - bbox.min_y - eps) / size).floor() as i64;
        (ix0, ix1, iy0, iy1)
    }

    /// chunk keys covering the bbox at the given level, row-major, at the
    /// source's own time. the engine stamps the pull time onto them
    pub fn cover(&self, bbox: &Bbox, level: u8) -> Vec<ChunkKey> {
        let (ix0, ix1, iy0, iy1) = self.tile_range(bbox, level);
        let mut keys = Vec::new();
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                keys.push(ChunkKey {
                    level,
                    ix,
                    iy,
                    time: None,
                });
            }
        }
        keys
    }
}
