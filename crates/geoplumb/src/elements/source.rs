//! in-memory raster source. the whole dataset is held resident (a geotiff is
//! fetched or read once at open) and windows are served by block-averaged
//! decimation per ladder level. TODO: ranged cog reads once terrano grows a
//! windowed reader, today its cog support is write-side only

use crate::caps::{CapsPattern, CapsSet, Constraint, Crs, RasterPattern, ResRange, SetField};
use crate::chunk::RasterChunk;
use crate::element::Source;
use crate::error::Result;
use crate::window::{GridSpec, WindowReq};
use futures::future::BoxFuture;
use terrano_core::{BandedRaster, GeoTiffMetadata, Raster, read_geotiff_bands};

pub struct RasterSrc {
    data: BandedRaster,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    crs: Crs,
}

impl RasterSrc {
    pub fn new(data: BandedRaster, origin_x: f64, origin_y: f64, crs: Crs) -> Self {
        let base_resolution = data.cell_size();
        RasterSrc {
            data,
            origin_x,
            origin_y,
            base_resolution,
            crs,
        }
    }

    pub fn from_geotiff(bytes: &[u8]) -> Result<Self> {
        let (data, meta): (BandedRaster, GeoTiffMetadata) = read_geotiff_bands(bytes)?;
        Ok(RasterSrc::new(
            data,
            meta.origin_x,
            meta.origin_y,
            Crs(u32::from(meta.epsg)),
        ))
    }

    fn sample_level(&self, req: &WindowReq) -> RasterChunk {
        let factor = (req.resolution / self.base_resolution).round().max(1.0) as usize;
        let res = req.resolution;
        let cols = (req.bbox.width() / res).round() as usize;
        let rows = (req.bbox.height() / res).round() as usize;
        let bands: Vec<Raster> = self
            .data
            .bands()
            .iter()
            .map(|band| {
                let nodata = band.nodata;
                let mut out = vec![nodata; cols * rows];
                for row in 0..rows {
                    for col in 0..cols {
                        let x0 = req.bbox.min_x + col as f64 * res;
                        let y0 = req.bbox.max_y - row as f64 * res;
                        let src_col = ((x0 - self.origin_x) / self.base_resolution).round() as i64;
                        let src_row = ((self.origin_y - y0) / self.base_resolution).round() as i64;
                        // clamp the averaging block to the raster up front so
                        // far-outside windows cost nothing
                        let r0 = src_row.max(0) as usize;
                        let r1 = (src_row + factor as i64).clamp(0, band.height() as i64) as usize;
                        let c0 = src_col.max(0) as usize;
                        let c1 = (src_col + factor as i64).clamp(0, band.width() as i64) as usize;
                        let mut sum = 0.0;
                        let mut n = 0usize;
                        for rr in r0..r1 {
                            for cc in c0..c1 {
                                let v = band.data()[rr * band.width() + cc];
                                if !band.is_nodata(v) && v.is_finite() {
                                    sum += v;
                                    n += 1;
                                }
                            }
                        }
                        out[row * cols + col] = if n > 0 { sum / n as f64 } else { nodata };
                    }
                }
                Raster::from_vec(cols, rows, out, res, nodata).expect("level dims")
            })
            .collect();
        RasterChunk {
            bands: BandedRaster::new(bands).expect("uniform bands"),
            bbox: req.bbox,
            resolution: res,
            crs: self.crs,
        }
    }
}

impl Source for RasterSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtype: SetField::one(crate::caps::Dtype::F64),
            bands: SetField::one(self.data.band_count() as u16),
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

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<RasterChunk>> {
        Box::pin(async move { Ok(self.sample_level(req)) })
    }
}
