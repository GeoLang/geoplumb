//! xyz tile adapter: turns z/x/y into a web mercator window pull plus an
//! exact-grid resample, the sink-side face of the engine

use crate::chunk::RasterChunk;
use crate::engine::Engine;
use crate::error::Result;
use crate::graph::NodeId;
use crate::resample::resample_to_grid;
use crate::window::{Bbox, WindowReq};

pub const TILE_PX: usize = 256;
const WEB_MERCATOR_EXTENT: f64 = 20037508.342789244;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct XyzTile {
    pub z: u8,
    pub x: u32,
    pub y: u32,
}

impl XyzTile {
    pub fn bbox(&self) -> Bbox {
        let n = f64::from(1u32 << self.z);
        let size = 2.0 * WEB_MERCATOR_EXTENT / n;
        Bbox {
            min_x: -WEB_MERCATOR_EXTENT + f64::from(self.x) * size,
            max_x: -WEB_MERCATOR_EXTENT + f64::from(self.x + 1) * size,
            max_y: WEB_MERCATOR_EXTENT - f64::from(self.y) * size,
            min_y: WEB_MERCATOR_EXTENT - f64::from(self.y + 1) * size,
        }
    }

    pub fn resolution(&self) -> f64 {
        2.0 * WEB_MERCATOR_EXTENT / (f64::from(1u32 << self.z) * TILE_PX as f64)
    }
}

/// pull a node's window for a tile and resample onto the exact 256 px grid
pub async fn render_tile(engine: &Engine, node: NodeId, tile: XyzTile) -> Result<RasterChunk> {
    let bbox = tile.bbox();
    let pulled = engine
        .pull(
            node,
            WindowReq {
                bbox,
                resolution: tile.resolution(),
            },
        )
        .await?;
    Ok(resample_to_grid(&pulled, &bbox, TILE_PX, TILE_PX))
}
