use crate::bands::{RasterBands, narrowed};
use crate::caps::{
    BandDtypes, Caps, CapsPattern, CapsSet, Constraint, Dtype, FieldMask, RasterPattern, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::{Adapter, Transform};
use crate::error::{Error, Result};
use crate::window::WindowReq;

pub struct Cast {
    to: BandDtypes,
}

fn cast_constraint(to: SetField<BandDtypes>) -> Constraint {
    Constraint::Derived {
        input: CapsSet::any_raster(),
        passthrough: FieldMask {
            dtypes: false,
            ..FieldMask::ALL
        },
        output: CapsPattern::Raster(RasterPattern {
            dtypes: to,
            ..RasterPattern::default()
        }),
    }
}

impl Cast {
    pub fn adapter() -> Adapter {
        Adapter {
            template: cast_constraint(SetField::Any),
            build: Box::new(|target| {
                let CapsPattern::Raster(raster) = target else {
                    return None;
                };
                let SetField::OneOf(dtypes) = &raster.dtypes else {
                    return None;
                };
                let to = dtypes.first()?.clone();
                Some(Box::new(Cast { to }))
            }),
        }
    }
}

impl Transform for Cast {
    fn identity(&self) -> String {
        format!("Cast to={:?}", self.to)
    }

    fn constraint(&self) -> Constraint {
        cast_constraint(SetField::one(self.to.clone()))
    }

    fn configure(&mut self, input: &Caps, _output: &Caps) -> Result<()> {
        let input = input.raster();
        for band in 0..usize::from(input.bands) {
            let (from, to) = (input.dtypes.band(band), self.to.band(band));
            if from != to && !from.widens_losslessly_to(to) {
                return Err(Error::InvalidGraph(format!(
                    "dtype of band {band} cannot go from {from:?} to {to:?} without losing values"
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
        let bands = input
            .bands
            .bands()
            .iter()
            .enumerate()
            .map(|(bi, band)| {
                let to = self.to.band(bi);
                if Dtype::of(band) == to {
                    return band.clone();
                }
                let widened = band.to_raster().into_owned().into_data();
                narrowed(to, band.width(), band.height(), band.cell_size(), widened)
            })
            .collect();
        Ok(Chunk::Raster(RasterChunk {
            bands: RasterBands::new(bands)?,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        }))
    }
}
