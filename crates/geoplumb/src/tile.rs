//! xyz tile adapter: turns z/x/y into a web mercator window pull plus an
//! exact-grid resample, the sink-side face of the engine

use crate::chunk::RasterChunk;
use crate::engine::Engine;
use crate::error::Result;
use crate::graph::NodeId;
use crate::resample::resample_to_grid;
use crate::window::{Bbox, TimeInterval, WindowReq};

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

    pub fn covering(z: u8, lonlat: &Bbox) -> impl Iterator<Item = XyzTile> {
        let n = f64::from(1u32 << z);
        let last = (1u32 << z) - 1;
        let column = |lon: f64| (((lon + 180.0) / 360.0 * n).floor().max(0.0) as u32).min(last);
        let row = |lat: f64| {
            let r = lat.to_radians();
            let y = (1.0 - (r.tan() + 1.0 / r.cos()).ln() / std::f64::consts::PI) / 2.0 * n;
            (y.floor().max(0.0) as u32).min(last)
        };
        let (x0, x1) = (column(lonlat.min_x), column(lonlat.max_x));
        let (y0, y1) = (row(lonlat.max_y), row(lonlat.min_y));
        (x0..=x1).flat_map(move |x| (y0..=y1).map(move |y| XyzTile { z, x, y }))
    }
}

/// pull a node's window for a tile and resample onto the exact 256 px grid
pub async fn render_tile(engine: &Engine, node: NodeId, tile: XyzTile) -> Result<RasterChunk> {
    render_tile_at(engine, node, tile, None).await
}

/// the same tile at a pull time, overriding the interval a time-varying
/// source was configured with
pub async fn render_tile_at(
    engine: &Engine,
    node: NodeId,
    tile: XyzTile,
    time: Option<TimeInterval>,
) -> Result<RasterChunk> {
    let bbox = tile.bbox();
    let pulled = engine
        .pull(
            node,
            WindowReq {
                bbox,
                resolution: tile.resolution(),
                time,
            },
        )
        .await?;
    let raster = pulled.raster()?;
    let value_kinds = &engine.caps(node).raster().value_kinds;
    Ok(resample_to_grid(
        raster,
        &bbox,
        TILE_PX,
        TILE_PX,
        value_kinds,
    ))
}
