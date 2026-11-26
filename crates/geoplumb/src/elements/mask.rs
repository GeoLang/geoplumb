//! masking by a quality band: sentinel-2 scl, landsat qa and their kin. the
//! codes in such a band are small integers, so a keep-list matches by exact
//! equality. window-local on an identity plan, so chunked output equals
//! whole-window output

use crate::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::WindowReq;
use terrano_core::{BandedRaster, Raster};

/// keeps a pixel where the quality band holds one of `valid_values` and sets
/// every band to NaN where it does not, the quality band included. band count
/// and band order are untouched, so downstream band indices stay put. nodata
/// in the quality band is already an invalid pixel, so it masks too
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
    bands: &[Raster],
    quality: &Raster,
    valid_values: &[f64],
    resolution: f64,
) -> Vec<Raster> {
    let keep: Vec<bool> = quality
        .data()
        .iter()
        .map(|v| !quality.is_nodata(*v) && valid_values.contains(v))
        .collect();
    bands
        .iter()
        .map(|band| {
            let data: Vec<f64> = band
                .data()
                .iter()
                .zip(&keep)
                .map(|(v, keep)| {
                    if *keep && !band.is_nodata(*v) {
                        *v
                    } else {
                        f64::NAN
                    }
                })
                .collect();
            Raster::from_vec(band.width(), band.height(), data, resolution, f64::NAN)
                .expect("mask dims")
        })
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
        );
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        }))
    }
}
