//! point cloud source and gridder over nubis, the PDAL slot. `LasSrc` holds
//! a cloud resident and serves tile windows with per-level voxel thinning,
//! `IdwGrid` is the cross-kind element turning point chunks into rasters

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, FieldMask, PointPattern, RasterPattern, ResRange,
    SetField,
};
use crate::chunk::{Chunk, PointChunk, RasterChunk, tile_contains};
use crate::element::{Source, Transform};
use crate::error::{Error, Result};
use crate::window::{Bbox, GridSpec, WindowReq};
use futures::future::BoxFuture;
use nubis_core::{GridWindow, PointCloud, idw_window, thin_voxel};
use terrano_core::{BandedRaster, Raster};

/// in-memory point cloud source. the ladder thins by voxel decimation: a
/// level's chunk keeps roughly one point per cell of that resolution, the
/// point analogue of block-averaged raster decimation. level 0 is the raw
/// cloud
pub struct LasSrc {
    cloud: PointCloud,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    crs: Crs,
}

impl LasSrc {
    /// errors on an empty cloud, there is no grid to anchor. the base
    /// resolution is the mean point spacing estimated from bounds and count
    pub fn new(cloud: PointCloud, crs: Crs) -> Result<Self> {
        let (min, max) = cloud
            .bounds()
            .ok_or(Error::Source("empty point cloud".into()))?;
        let area = (max.x - min.x) * (max.y - min.y);
        let base_resolution = if area > 0.0 {
            (area / cloud.len() as f64).sqrt()
        } else {
            1.0
        };
        Ok(LasSrc {
            cloud,
            origin_x: min.x,
            origin_y: max.y,
            base_resolution,
            crs,
        })
    }

    pub fn from_las(reader: &mut (impl std::io::Read + std::io::Seek), crs: Crs) -> Result<Self> {
        let cloud =
            nubis_core::read_las(reader).map_err(|e| Error::Source(format!("las read: {e}")))?;
        LasSrc::new(cloud, crs)
    }
}

impl Source for LasSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::PointCloud(PointPattern {
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
        Box::pin(async move {
            let tile: Vec<_> = self
                .cloud
                .points()
                .iter()
                .filter(|p| tile_contains(&req.bbox, p.x, p.y))
                .copied()
                .collect();
            let tile = PointCloud::from_points(tile);
            let points = if req.resolution > self.base_resolution {
                thin_voxel(&tile, req.resolution)
            } else {
                tile
            };
            Ok(Chunk::PointCloud(PointChunk {
                points,
                bbox: req.bbox,
                resolution: req.resolution,
                crs: self.crs,
            }))
        })
    }
}

/// inverse distance weighted gridding, points in and a single-band raster
/// out. the search radius is in output pixels so every ladder level greets
/// a comparable neighborhood
pub struct IdwGrid {
    pub power: f64,
    pub radius_px: f64,
    pub min_points: usize,
}

impl Default for IdwGrid {
    fn default() -> Self {
        IdwGrid {
            power: 2.0,
            radius_px: 4.0,
            min_points: 1,
        }
    }
}

impl Transform for IdwGrid {
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::one(CapsPattern::PointCloud(PointPattern::default())),
            passthrough: FieldMask {
                crs: true,
                chunk_px: true,
                dtype: false,
                bands: false,
                resolution: false,
            },
            output: CapsPattern::Raster(RasterPattern {
                dtype: SetField::one(Dtype::F64),
                bands: SetField::one(1),
                ..RasterPattern::default()
            }),
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        out.with_window(
            out.bbox.expand(self.radius_px * out.resolution),
            out.resolution,
        )
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        dirty.expand(self.radius_px * resolution)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.points()?;
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        // nubis grids bottom-up at cell nodes, so anchor the window at the
        // bottom-left cell center and flip rows into the raster order
        let window = GridWindow {
            origin_x: out.bbox.min_x + 0.5 * res,
            origin_y: out.bbox.min_y + 0.5 * res,
            width: cols,
            height: rows,
            cell_size: res,
        };
        let grid = idw_window(
            &input.points,
            &window,
            self.power,
            self.radius_px * res,
            self.min_points,
        )
        .ok_or(Error::Source("idw window rejected its parameters".into()))?;
        let mut data = vec![f64::NAN; cols * rows];
        for row in 0..rows {
            let src = (rows - 1 - row) * cols;
            for col in 0..cols {
                let v = grid.data[src + col];
                data[row * cols + col] = if v == grid.nodata { f64::NAN } else { v };
            }
        }
        let band = Raster::from_vec(cols, rows, data, res, f64::NAN).expect("idw dims");
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(vec![band]).expect("single band"),
            bbox: out.bbox,
            resolution: res,
            crs: input.crs,
        }))
    }
}
