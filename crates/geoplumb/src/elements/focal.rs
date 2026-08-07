//! focal statistics over a square window, per band. the upstream plan widens
//! the window by the radius, compute runs terrano's focal_stats over the
//! wider input and crops the halo back off, which is what keeps chunk
//! borders seam-free. the widened request is already on the node's pixel
//! grid, so the engine's outward alignment adds nothing and radius cells of
//! halo is exactly enough

use crate::caps::{CapsSet, Constraint};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, WindowReq};
use terrano_core::{BandedRaster, FocalStat, Neighborhood, focal_stats};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocalOp {
    Mean,
    Median,
    Min,
    Max,
}

impl From<FocalOp> for FocalStat {
    fn from(op: FocalOp) -> FocalStat {
        match op {
            FocalOp::Mean => FocalStat::Mean,
            FocalOp::Median => FocalStat::Median,
            FocalOp::Min => FocalStat::Min,
            FocalOp::Max => FocalStat::Max,
        }
    }
}

/// one statistic per output pixel over the square window of side
/// `2 * radius + 1` centred on it, every band independently, band count
/// unchanged. a nodata neighbour drops out of the statistic, a nodata centre
/// stays nodata, and nodata is NaN in and out
pub struct Focal {
    pub op: FocalOp,
    pub radius: u32,
}

impl Focal {
    pub fn new(op: FocalOp, radius: u32) -> Focal {
        Focal { op, radius }
    }

    fn halo(&self, resolution: f64) -> f64 {
        f64::from(self.radius) * resolution
    }
}

impl Transform for Focal {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        out.with_window(out.bbox.expand(self.halo(out.resolution)), out.resolution)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        dirty.expand(self.halo(resolution))
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let bands = input
            .bands
            .bands()
            .iter()
            .map(|band| {
                focal_stats(
                    band,
                    self.radius as usize,
                    Neighborhood::Square,
                    self.op.into(),
                )
            })
            .collect();
        let full = RasterChunk {
            bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        };
        Ok(Chunk::Raster(full.crop_to(&out.bbox)))
    }
}
