use crate::bands::{RasterBands, narrowed_raster};
use crate::caps::{BandKinds, Caps, Constraint, Dtype, ValueKind};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::elements::raster_vector::one_band;
use crate::error::{Error, Result};
use crate::resample::resample_to_grid;
use crate::window::{Bbox, WindowReq};
use terrano_core::viewshed;

// terrano's viewshed codes are 0 hidden and 1 visible
const VIEWSHED_DTYPE: Dtype = Dtype::U8 { nodata: Some(255) };

pub struct Viewshed {
    pub observer_x: f64,
    pub observer_y: f64,
    pub observer_height: f64,
    pub radius: f64,
}

impl Transform for Viewshed {
    fn identity(&self) -> String {
        format!(
            "Viewshed observer_x={:?} observer_y={:?} observer_height={:?} radius={:?}",
            self.observer_x, self.observer_y, self.observer_height, self.radius
        )
    }

    fn constraint(&self) -> Constraint {
        Constraint::pinned_value_kind_and_dtype(
            one_band(ValueKind::Continuous),
            ValueKind::Categorical,
            VIEWSHED_DTYPE,
        )
    }

    fn configure(&mut self, _input: &Caps, _output: &Caps) -> Result<()> {
        let observer = [self.observer_x, self.observer_y, self.observer_height];
        if !observer.iter().all(|v| v.is_finite()) {
            return Err(Error::InvalidGraph(format!(
                "a viewshed needs a finite observer, got {observer:?}"
            )));
        }
        if !(self.radius.is_finite() && self.radius > 0.0) {
            return Err(Error::InvalidGraph(format!(
                "a viewshed needs a positive radius, got {}",
                self.radius
            )));
        }
        Ok(())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn region(&self) -> Option<Bbox> {
        Some(Bbox::new(
            self.observer_x - self.radius,
            self.observer_y - self.radius,
            self.observer_x + self.radius,
            self.observer_y + self.radius,
        ))
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let dem = input
            .bands
            .band(0)
            .ok_or(Error::Source("empty input chunk".into()))?
            .to_raster();
        let columns = (out.bbox.width() / out.resolution).round() as usize;
        let rows = (out.bbox.height() / out.resolution).round() as usize;
        let column = ((self.observer_x - input.bbox.min_x) / input.resolution).floor() as usize;
        let row = ((input.bbox.max_y - self.observer_y) / input.resolution).floor() as usize;
        let seen = viewshed(&dem, row, column, self.observer_height, self.radius);
        let disc = RasterChunk {
            bands: RasterBands::new(vec![narrowed_raster(VIEWSHED_DTYPE, &seen)])?,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        };
        Ok(Chunk::Raster(resample_to_grid(
            &disc,
            &out.bbox,
            columns,
            rows,
            &BandKinds::from(ValueKind::Categorical),
        )))
    }
}
