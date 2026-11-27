use std::borrow::Cow;

use crate::bands::{RasterBands, finite_value, gathered, narrowed_like};
use crate::caps::{BandKinds, Caps, CapsSet, Constraint, ValueKind};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Fanin;
use crate::error::Result;
use crate::resample::{nearest_index, peak_index, sample_point};
use crate::window::WindowReq;
use terrano_core::{Raster, TypedBand};

#[derive(Default)]
pub struct Mosaic {
    value_kinds: Option<BandKinds>,
}

impl Fanin for Mosaic {
    fn identity(&self) -> String {
        "Mosaic".into()
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn configure(&mut self, _inputs: &[Caps], output: &Caps) -> Result<()> {
        self.value_kinds = Some(output.raster().value_kinds.clone());
        Ok(())
    }

    fn compute(&self, out: &WindowReq, inputs: &[Chunk]) -> Result<Chunk> {
        let inputs: Vec<&RasterChunk> = inputs.iter().map(Chunk::raster).collect::<Result<_>>()?;
        let value_kinds = self.value_kinds.as_ref().expect("configured");
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        let center = |index: usize| {
            let (row, col) = (index / cols, index % cols);
            (
                out.bbox.min_x + (col as f64 + 0.5) * res,
                out.bbox.max_y - (row as f64 + 0.5) * res,
            )
        };
        let band_count = inputs[0].bands.band_count();
        let bands = (0..band_count)
            .map(|bi| {
                let band: Vec<&TypedBand> = inputs
                    .iter()
                    .map(|c| c.bands.band(bi).expect("negotiated bands"))
                    .collect();
                match value_kinds.band(bi) {
                    ValueKind::Categorical => gathered(&band, cols, rows, res, |index| {
                        let (x, y) = center(index);
                        inputs.iter().enumerate().find_map(|(k, c)| {
                            let at = nearest_index(c, x, y)?;
                            finite_value(band[k], at).map(|_| (k, at))
                        })
                    }),
                    ValueKind::Peak => gathered(&band, cols, rows, res, |index| {
                        let (x, y) = center(index);
                        inputs.iter().enumerate().find_map(|(k, c)| {
                            peak_index(c, x, y, res, |at| finite_value(band[k], at))
                                .map(|at| (k, at))
                        })
                    }),
                    ValueKind::Continuous => {
                        let rasters: Vec<Cow<'_, Raster>> =
                            band.iter().map(|band| band.to_raster()).collect();
                        let nodata = rasters[0].nodata;
                        let data = (0..cols * rows)
                            .map(|index| {
                                let (x, y) = center(index);
                                inputs
                                    .iter()
                                    .zip(&rasters)
                                    .find_map(|(c, raster)| {
                                        sample_point(ValueKind::Continuous, raster, c, x, y)
                                    })
                                    .unwrap_or(nodata)
                            })
                            .collect();
                        Ok(narrowed_like(band[0], cols, rows, res, data))
                    }
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Chunk::Raster(RasterChunk {
            bands: RasterBands::new(bands)?,
            bbox: out.bbox,
            resolution: res,
            crs: inputs[0].crs,
        }))
    }
}
