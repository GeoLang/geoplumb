//! self-describing chunks. a pull response carries its resolved grid rather
//! than trusting the request, since snapping may widen or align the window

use crate::caps::Crs;
use crate::error::{Error, Result};
use crate::window::Bbox;
use nubis_core::{Point3, PointCloud};
use terrano_core::{BandedRaster, Raster};

#[derive(Debug, Clone)]
pub struct RasterChunk {
    pub bands: BandedRaster,
    pub bbox: Bbox,
    pub resolution: f64,
    pub crs: Crs,
}

/// points inside one tile window. resolution is the ladder level the points
/// were thinned for, the point analogue of pixel size
#[derive(Debug, Clone)]
pub struct PointChunk {
    pub points: PointCloud,
    pub bbox: Bbox,
    pub resolution: f64,
    pub crs: Crs,
}

/// vector and tensor variants are reserved here
#[derive(Debug, Clone)]
pub enum Chunk {
    Raster(RasterChunk),
    PointCloud(PointChunk),
}

impl Chunk {
    pub fn raster(&self) -> Result<&RasterChunk> {
        match self {
            Chunk::Raster(r) => Ok(r),
            _ => Err(Error::Kind("raster")),
        }
    }

    pub fn into_raster(self) -> Result<RasterChunk> {
        match self {
            Chunk::Raster(r) => Ok(r),
            _ => Err(Error::Kind("raster")),
        }
    }

    pub fn points(&self) -> Result<&PointChunk> {
        match self {
            Chunk::PointCloud(p) => Ok(p),
            _ => Err(Error::Kind("point cloud")),
        }
    }

    pub fn into_points(self) -> Result<PointChunk> {
        match self {
            Chunk::PointCloud(p) => Ok(p),
            _ => Err(Error::Kind("point cloud")),
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Chunk::Raster(r) => r.byte_size(),
            Chunk::PointCloud(p) => p.byte_size(),
        }
    }
}

/// tile membership for points: x in [min, max), y in (min, max]. tiles step
/// down from the grid origin, so the top edge is the inclusive one. every
/// producer and consumer of point chunks must share this convention or
/// points on tile seams duplicate or vanish
pub fn tile_contains(bbox: &Bbox, x: f64, y: f64) -> bool {
    x >= bbox.min_x && x < bbox.max_x && y > bbox.min_y && y <= bbox.max_y
}

impl PointChunk {
    pub fn byte_size(&self) -> usize {
        self.points.len() * size_of::<Point3>()
    }

    /// keep the points inside `bbox` under the tile membership convention
    pub fn crop_to(&self, bbox: &Bbox) -> PointChunk {
        let kept: Vec<Point3> = self
            .points
            .points()
            .iter()
            .filter(|p| tile_contains(bbox, p.x, p.y))
            .copied()
            .collect();
        PointChunk {
            points: PointCloud::from_points(kept),
            bbox: *bbox,
            resolution: self.resolution,
            crs: self.crs,
        }
    }
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
