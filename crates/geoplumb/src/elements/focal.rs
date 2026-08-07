//! focal statistics over a square window, per band. the upstream plan widens
//! the window by the radius and compute reads the output window back out of
//! the wider input, which is what keeps chunk borders seam-free. the widened
//! request is already on the node's pixel grid, so the engine's outward
//! alignment adds nothing and radius cells of halo is exactly enough

use crate::caps::{CapsSet, Constraint};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, WindowReq};
use terrano_core::{BandedRaster, Raster};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FocalOp {
    Mean,
    Median,
    Min,
    Max,
}

impl FocalOp {
    /// the statistic over a window's valid values, NaN over an empty set.
    /// `values` is the caller's scratch buffer, median sorts it in place. an
    /// even count takes the mean of the two middle values
    fn apply(self, values: &mut [f64]) -> f64 {
        match self {
            FocalOp::Mean => values.iter().sum::<f64>() / values.len() as f64,
            FocalOp::Median => {
                values.sort_by(f64::total_cmp);
                match values.len() {
                    0 => f64::NAN,
                    n if n % 2 == 1 => values[n / 2],
                    n => (values[n / 2 - 1] + values[n / 2]) / 2.0,
                }
            }
            // f64::min and f64::max return the other operand against NaN, so
            // folding from NaN gives the statistic and NaN over no values
            FocalOp::Min => values.iter().copied().fold(f64::NAN, f64::min),
            FocalOp::Max => values.iter().copied().fold(f64::NAN, f64::max),
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
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        let (in_cols, in_rows) = (input.width(), input.height());
        let col0 = ((out.bbox.min_x - input.bbox.min_x) / res).round() as isize;
        let row0 = ((input.bbox.max_y - out.bbox.max_y) / res).round() as isize;
        let radius = self.radius as isize;
        let side = 2 * self.radius as usize + 1;
        let mut window = Vec::with_capacity(side * side);
        let mut bands = Vec::with_capacity(input.bands.band_count());
        for band in input.bands.bands() {
            let tap = |row: isize, col: isize| -> f64 {
                if row < 0 || col < 0 || row >= in_rows as isize || col >= in_cols as isize {
                    return f64::NAN;
                }
                let v = band.data()[row as usize * in_cols + col as usize];
                if band.is_nodata(v) { f64::NAN } else { v }
            };
            let mut data = Vec::with_capacity(cols * rows);
            for row in 0..rows {
                for col in 0..cols {
                    let (centre_row, centre_col) = (row0 + row as isize, col0 + col as isize);
                    if tap(centre_row, centre_col).is_nan() {
                        data.push(f64::NAN);
                        continue;
                    }
                    window.clear();
                    for dy in -radius..=radius {
                        for dx in -radius..=radius {
                            let v = tap(centre_row + dy, centre_col + dx);
                            if !v.is_nan() {
                                window.push(v);
                            }
                        }
                    }
                    data.push(self.op.apply(&mut window));
                }
            }
            bands.push(Raster::from_vec(cols, rows, data, res, f64::NAN).expect("focal dims"));
        }
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: out.bbox,
            resolution: res,
            crs: input.crs,
        }))
    }
}
