//! self-describing chunks. a pull response carries its resolved grid rather
//! than trusting the request, since snapping may widen or align the window

use crate::caps::Crs;
use crate::window::Bbox;
use terrano_core::{BandedRaster, Raster};

#[derive(Debug, Clone)]
pub struct RasterChunk {
    pub bands: BandedRaster,
    pub bbox: Bbox,
    pub resolution: f64,
    pub crs: Crs,
}

/// vector and point cloud variants are reserved here
#[derive(Debug, Clone)]
pub enum Chunk {
    Raster(RasterChunk),
}

impl RasterChunk {
    pub fn width(&self) -> usize {
        self.bands.width()
    }

    pub fn height(&self) -> usize {
        self.bands.height()
    }

    pub fn byte_size(&self) -> usize {
        self.bands.band_count() * self.width() * self.height() * size_of::<f64>()
    }

    /// crop to a bbox that must lie on this chunk's pixel grid
    pub fn crop_to(&self, bbox: &Bbox) -> RasterChunk {
        let res = self.resolution;
        let col0 = ((bbox.min_x - self.bbox.min_x) / res).round().max(0.0) as usize;
        let row0 = ((self.bbox.max_y - bbox.max_y) / res).round().max(0.0) as usize;
        let cols = (bbox.width() / res).round() as usize;
        let rows = (bbox.height() / res).round() as usize;
        let cols = cols.min(self.width().saturating_sub(col0));
        let rows = rows.min(self.height().saturating_sub(row0));
        let bands = self
            .bands
            .bands()
            .iter()
            .map(|b| {
                let mut out = Vec::with_capacity(cols * rows);
                for r in 0..rows {
                    let src = (row0 + r) * self.width() + col0;
                    out.extend_from_slice(&b.data()[src..src + cols]);
                }
                Raster::from_vec(cols, rows, out, res, b.nodata).expect("crop dims consistent")
            })
            .collect();
        RasterChunk {
            bands: BandedRaster::new(bands).expect("uniform crop"),
            bbox: Bbox {
                min_x: self.bbox.min_x + col0 as f64 * res,
                max_y: self.bbox.max_y - row0 as f64 * res,
                max_x: self.bbox.min_x + (col0 + cols) as f64 * res,
                min_y: self.bbox.max_y - (row0 + rows) as f64 * res,
            },
            resolution: res,
            crs: self.crs,
        }
    }
}
