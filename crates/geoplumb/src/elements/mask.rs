//! masking by a quality band: sentinel-2 scl, landsat qa and their kin. the
//! codes in such a band are small integers, so a keep-list matches by exact
//! equality. window-local on an identity plan, so chunked output equals
//! whole-window output

use crate::bands::{RasterBands, finite_value, masked};
use crate::caps::{Caps, CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::WindowReq;
use terrano_core::TypedBand;

pub struct QualityMask {
    pub band: usize,
    pub valid_values: Vec<f64>,
}

impl QualityMask {
    pub fn new(band: usize, valid_values: Vec<f64>) -> QualityMask {
        QualityMask { band, valid_values }
    }
}

pub(crate) fn mask_by_quality(
    bands: &[TypedBand],
    quality: &TypedBand,
    valid_values: &[f64],
    resolution: f64,
) -> Result<Vec<TypedBand>> {
    let keep: Vec<bool> = (0..quality.width() * quality.height())
        .map(|index| {
            finite_value(quality, index).is_some_and(|value| valid_values.contains(&value))
        })
        .collect();
    bands
        .iter()
        .map(|band| masked(band, &keep, resolution))
        .collect()
}

impl Transform for QualityMask {
    fn identity(&self) -> String {
        format!(
            "QualityMask band={} valid_values={:?}",
            self.band, self.valid_values
        )
    }

    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Raster(RasterPattern {
            // a band index past u16 just demands more than any link carries
            bands: SetField::AtLeast(u16::try_from(self.band + 1).unwrap_or(u16::MAX)),
            ..RasterPattern::default()
        })))
    }

    fn configure(&mut self, input: &Caps, _output: &Caps) -> Result<()> {
        let input = input.raster();
        for band in 0..usize::from(input.bands) {
            let dtype = input.dtypes.band(band);
            if !dtype.marks_absent_cells() {
                return Err(Error::InvalidGraph(format!(
                    "quality mask needs a nodata value on band {band}, its dtype {dtype:?} has none"
                )));
            }
        }
        Ok(())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?.crop_to(&out.bbox);
        let quality = input.bands.band(self.band).expect("negotiated bands");
        let bands = mask_by_quality(
            input.bands.bands(),
            quality,
            &self.valid_values,
            input.resolution,
        )?;
        Ok(Chunk::Raster(RasterChunk {
            bands: RasterBands::new(bands)?,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        }))
    }
}
