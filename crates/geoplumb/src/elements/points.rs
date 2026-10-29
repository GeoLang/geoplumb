//! point cloud source and gridder over nubis, the PDAL slot. `LasSrc` holds
//! a cloud resident and serves tile windows with per-level voxel thinning,
//! `IdwGrid` is the cross-kind element turning point chunks into rasters

use std::collections::HashMap;

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, FieldMask, PointPattern, RasterPattern, ResRange,
    SetField,
};
use crate::chunk::{Chunk, PointChunk, RasterChunk, tile_contains};
use crate::element::{Source, Transform};
use crate::error::{Error, Result};
use crate::spill::Fnv1a;
use crate::window::{Bbox, GridSpec, WindowReq};
use futures::future::BoxFuture;
use nubis_core::{GridWindow, Point3, PointCloud, idw_window, thin_voxel};
use terrano_core::{BandedRaster, Raster};

const TILE_PX: u32 = 256;

// rows count down from the origin, as the ladder does
fn bin_of(x: f64, y: f64, origin_x: f64, origin_y: f64, bin_size: f64) -> (i64, i64) {
    (
        ((x - origin_x) / bin_size).floor() as i64,
        ((origin_y - y) / bin_size).floor() as i64,
    )
}

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
    bin_size: f64,
    bins: HashMap<(i64, i64), Vec<usize>>,
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
        let bin_size = f64::from(TILE_PX) * base_resolution;
        let mut bins: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        for (index, point) in cloud.points().iter().enumerate() {
            bins.entry(bin_of(point.x, point.y, min.x, max.y, bin_size))
                .or_default()
                .push(index);
        }
        Ok(LasSrc {
            cloud,
            origin_x: min.x,
            origin_y: max.y,
            base_resolution,
            crs,
            bin_size,
            bins,
        })
    }

    pub fn from_las(reader: &mut (impl std::io::Read + std::io::Seek), crs: Crs) -> Result<Self> {
        let cloud =
            nubis_core::read_las(reader).map_err(|e| Error::Source(format!("las read: {e}")))?;
        LasSrc::new(cloud, crs)
    }

    pub fn candidates(&self, bbox: &Bbox) -> impl Iterator<Item = &Point3> {
        let (first_column, first_row) = bin_of(
            bbox.min_x,
            bbox.max_y,
            self.origin_x,
            self.origin_y,
            self.bin_size,
        );
        let (last_column, last_row) = bin_of(
            bbox.max_x,
            bbox.min_y,
            self.origin_x,
            self.origin_y,
            self.bin_size,
        );
        let columns = (i128::from(last_column) - i128::from(first_column) + 1).max(0);
        let rows = (i128::from(last_row) - i128::from(first_row) + 1).max(0);
        let in_range = |(column, row): (i64, i64)| {
            (first_column..=last_column).contains(&column) && (first_row..=last_row).contains(&row)
        };
        let mut indices: Vec<usize> = if columns == 0 || rows == 0 {
            Vec::new()
        } else if columns.saturating_mul(rows) > self.bins.len() as i128 {
            self.bins
                .iter()
                .filter(|(key, _)| in_range(**key))
                .flat_map(|(_, indices)| indices.iter().copied())
                .collect()
        } else {
            (first_row..=last_row)
                .flat_map(|row| (first_column..=last_column).map(move |column| (column, row)))
                .filter_map(|key| self.bins.get(&key))
                .flat_map(|indices| indices.iter().copied())
                .collect()
        };
        // thin_voxel keeps the first point per voxel, so cloud order must survive
        indices.sort_unstable();
        let points = self.cloud.points();
        indices.into_iter().map(move |index| &points[index])
    }
}

impl Source for LasSrc {
    fn identity(&self) -> String {
        let mut digest = Fnv1a::new();
        for p in self.cloud.points() {
            for v in [p.x, p.y, p.z] {
                digest.update(&v.to_le_bytes());
            }
            digest.update(&p.intensity.to_le_bytes());
            digest.update(&[p.classification.to_u8()]);
        }
        format!(
            "LasSrc points={} origin_x={:?} origin_y={:?} base_resolution={:?} crs={} content={:016x}",
            self.cloud.len(),
            self.origin_x,
            self.origin_y,
            self.base_resolution,
            self.crs,
            digest.finish()
        )
    }

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
            chunk_px: TILE_PX,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        Box::pin(async move {
            let tile: Vec<_> = self
                .candidates(&req.bbox)
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
    fn identity(&self) -> String {
        format!(
            "IdwGrid power={:?} radius_px={:?} min_points={}",
            self.power, self.radius_px, self.min_points
        )
    }

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
