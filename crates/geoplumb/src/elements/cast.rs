use crate::bands::{RasterBands, narrowed_raster};
use crate::caps::{
    BandDtypes, Caps, CapsPattern, CapsSet, Constraint, Dtype, FieldMask, RasterPattern, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::{Adapter, Transform};
use crate::error::{Error, Result};
use crate::window::WindowReq;

pub struct Cast {
    to: BandDtypes,
    lossless_only: bool,
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
    pub fn new(to: BandDtypes) -> Cast {
        Cast {
            to,
            lossless_only: false,
        }
    }

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
                Some(Box::new(Cast {
                    to,
                    lossless_only: true,
                }))
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
            if self.lossless_only && from != to && !from.widens_losslessly_to(to) {
                return Err(Error::InvalidGraph(format!(
                    "dtype of band {band} cannot go from {from:?} to {to:?} without losing values"
                )));
            }
            if from.marks_absent_cells() && !to.marks_absent_cells() {
                return Err(Error::InvalidGraph(format!(
                    "a cast of band {band} from {from:?} to {to:?} has no nodata to mark absent cells"
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
                narrowed_raster(to, &band.to_raster())
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
