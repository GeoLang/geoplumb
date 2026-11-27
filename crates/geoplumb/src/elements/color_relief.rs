use crate::bands::{RasterBands, finite_value};
use crate::caps::{
    BandDtypes, BandKinds, CapsPattern, CapsSet, Constraint, Dtype, FieldMask, RasterPattern,
    SetField, ValueKind,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::WindowReq;
use jung_core::classification::ValueRamp;
pub use jung_core::classification::{ColorFile, ColorSelection, ValueRange};
use terrano_core::{SampleGrid, TypedBand};

pub struct ColorRelief {
    color_file: ColorFile,
    selection: ColorSelection,
    range: Option<ValueRange>,
}

fn nodata_value(band: &TypedBand) -> Option<f64> {
    match band {
        TypedBand::F64(raster) => Some(raster.nodata),
        TypedBand::F32(_) => Some(f64::NAN),
        integer => integer.integer_nodata(),
    }
}

impl ColorRelief {
    pub fn new(
        color_file: ColorFile,
        selection: ColorSelection,
        range: Option<ValueRange>,
    ) -> Result<ColorRelief> {
        if color_file.has_percent_stops() && range.is_none() {
            return Err(Error::InvalidGraph(
                "percent stops in a colour file need the raster's min and max".into(),
            ));
        }
        Ok(ColorRelief {
            color_file,
            selection,
            range,
        })
    }

    pub fn rgba(&self, band: &TypedBand) -> Result<Vec<[u8; 4]>> {
        let ramp = ValueRamp::new(
            &self.color_file,
            self.selection,
            nodata_value(band),
            self.range,
        )
        .map_err(|e| Error::InvalidGraph(e.to_string()))?;
        let pixels = (0..band.width() * band.height())
            .map(|index| {
                let color = match finite_value(band, index) {
                    Some(value) => ramp.color(value),
                    None => ramp.nodata_color(),
                };
                [color.r, color.g, color.b, color.a]
            })
            .collect();
        Ok(pixels)
    }
}

impl Transform for ColorRelief {
    fn identity(&self) -> String {
        format!(
            "ColorRelief entries={:?} selection={:?} range={:?}",
            self.color_file.entries, self.selection, self.range
        )
    }

    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::one(CapsPattern::Raster(RasterPattern {
                bands: SetField::one(1),
                ..RasterPattern::default()
            })),
            passthrough: FieldMask {
                dtypes: false,
                bands: false,
                value_kinds: false,
                ..FieldMask::ALL
            },
            output: CapsPattern::Raster(RasterPattern {
                dtypes: SetField::one(BandDtypes::from(Dtype::U8 { nodata: None })),
                bands: SetField::one(4),
                // bilinear resampling weighs all four channels alike
                value_kinds: SetField::one(BandKinds::from(ValueKind::Continuous)),
                ..RasterPattern::default()
            }),
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?.crop_to(&out.bbox);
        let band = input
            .bands
            .band(0)
            .ok_or(Error::Source("empty input chunk".into()))?;
        let pixels = self.rgba(band)?;
        let channel = |index: usize| -> Result<TypedBand> {
            let values = pixels.iter().map(|pixel| pixel[index]).collect();
            let samples = SampleGrid::new(input.width(), input.height(), input.resolution, values)?;
            Ok(TypedBand::U8 {
                samples,
                nodata: None,
            })
        };
        let bands = (0..4).map(channel).collect::<Result<Vec<_>>>()?;
        Ok(Chunk::Raster(RasterChunk {
            bands: RasterBands::new(bands)?,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        }))
    }
}
