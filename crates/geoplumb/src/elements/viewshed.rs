use crate::caps::{Caps, Constraint, ValueKind};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::elements::raster_vector::one_band;
use crate::error::{Error, Result};
use crate::resample::resample_to_grid;
use crate::window::{Bbox, WindowReq};
use terrano_core::{BandedRaster, Raster, viewshed};

pub struct Viewshed {
    pub observer_x: f64,
    pub observer_y: f64,
    pub observer_height: f64,
    pub radius: f64,
}

impl Viewshed {
    fn disc(&self) -> Bbox {
        Bbox::new(
            self.observer_x - self.radius,
            self.observer_y - self.radius,
            self.observer_x + self.radius,
            self.observer_y + self.radius,
        )
    }
}

impl Transform for Viewshed {
    fn identity(&self) -> String {
        format!(
            "Viewshed observer_x={:?} observer_y={:?} observer_height={:?} radius={:?}",
            self.observer_x, self.observer_y, self.observer_height, self.radius
        )
    }

    fn constraint(&self) -> Constraint {
        Constraint::pinned_value_kind(one_band(ValueKind::Continuous), ValueKind::Categorical)
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
        out.with_window(self.disc(), out.resolution)
    }

    fn output_without_input(&self, out: &WindowReq, output: &Caps) -> Option<Chunk> {
        if out.bbox.intersects(&self.disc()) {
            return None;
        }
        let columns = (out.bbox.width() / out.resolution).round() as usize;
        let rows = (out.bbox.height() / out.resolution).round() as usize;
        let outside = Raster::new(columns, rows, out.resolution, f64::NAN);
        Some(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(vec![outside]).expect("one band"),
            bbox: out.bbox,
            resolution: out.resolution,
            crs: output.crs(),
        }))
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        let disc = self.disc().expand(resolution);
        if dirty.intersects(&disc) {
            disc
        } else {
            *dirty
        }
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let dem = input
            .bands
            .band(0)
            .ok_or(Error::Source("empty input chunk".into()))?;
        let columns = (out.bbox.width() / out.resolution).round() as usize;
        let rows = (out.bbox.height() / out.resolution).round() as usize;
        let column = ((self.observer_x - input.bbox.min_x) / input.resolution).floor() as usize;
        let row = ((input.bbox.max_y - self.observer_y) / input.resolution).floor() as usize;
        let seen = viewshed(dem, row, column, self.observer_height, self.radius);
        let disc = RasterChunk {
            bands: BandedRaster::new(vec![seen]).expect("one band"),
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        };
        Ok(Chunk::Raster(resample_to_grid(
            &disc,
            &out.bbox,
            columns,
            rows,
            ValueKind::Categorical,
        )))
    }
}
