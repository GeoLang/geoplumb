//! first-wins mosaic: each output pixel takes the first input, wiring
//! order, that has a value there. inputs are sampled bilinearly onto the
//! output grid, exact when an input shares the output alignment

use crate::caps::{CapsSet, Constraint};
use crate::chunk::RasterChunk;
use crate::element::Fanin;
use crate::error::Result;
use crate::resample::sample_bilinear;
use crate::window::WindowReq;
use terrano_core::{BandedRaster, Raster};

pub struct Mosaic;

impl Fanin for Mosaic {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn compute(&self, out: &WindowReq, inputs: &[RasterChunk]) -> Result<RasterChunk> {
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        let band_count = inputs[0].bands.band_count();
        let bands: Vec<Raster> = (0..band_count)
            .map(|bi| {
                let nodata = inputs[0].bands.band(bi).expect("negotiated bands").nodata;
                let mut data = Vec::with_capacity(cols * rows);
                for row in 0..rows {
                    let y = out.bbox.max_y - (row as f64 + 0.5) * res;
                    for col in 0..cols {
                        let x = out.bbox.min_x + (col as f64 + 0.5) * res;
                        let v = inputs
                            .iter()
                            .find_map(|c| {
                                sample_bilinear(
                                    c.bands.band(bi).expect("negotiated bands"),
                                    c,
                                    x,
                                    y,
                                )
                            })
                            .unwrap_or(nodata);
                        data.push(v);
                    }
                }
                Raster::from_vec(cols, rows, data, res, nodata).expect("mosaic dims")
            })
            .collect();
        Ok(RasterChunk {
            bands: BandedRaster::new(bands).expect("uniform bands"),
            bbox: out.bbox,
            resolution: res,
            crs: inputs[0].crs,
        })
    }
}
