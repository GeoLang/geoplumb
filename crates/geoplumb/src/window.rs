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

/// a pull: give me this window at this ground resolution
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WindowReq {
    pub bbox: Bbox,
    pub resolution: f64,
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

/// cache-addressable unit: ladder level plus tile indices in the node grid
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ChunkKey {
    pub level: u8,
    pub ix: i64,
    pub iy: i64,
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

    /// chunk keys covering the bbox at the given level, row-major
    pub fn cover(&self, bbox: &Bbox, level: u8) -> Vec<ChunkKey> {
        let size = self.chunk_ground_size(level);
        let eps = size * 1e-9;
        let ix0 = ((bbox.min_x - self.origin_x + eps) / size).floor() as i64;
        let ix1 = ((bbox.max_x - self.origin_x - eps) / size).floor() as i64;
        let iy0 = ((self.origin_y - bbox.max_y + eps) / size).floor() as i64;
        let iy1 = ((self.origin_y - bbox.min_y - eps) / size).floor() as i64;
        let mut keys = Vec::new();
        for iy in iy0..=iy1 {
            for ix in ix0..=ix1 {
                keys.push(ChunkKey { level, ix, iy });
            }
        }
        keys
    }
}
