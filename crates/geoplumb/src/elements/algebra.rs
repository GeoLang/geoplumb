//! per-cell map algebra and reclassification over terrano ops

use crate::caps::{CapsSet, Constraint};
use crate::chunk::RasterChunk;
use crate::element::Transform;
use crate::error::Result;
use crate::window::WindowReq;
use terrano_core::{BandedRaster, UnaryOp, reclassify};

pub enum AlgebraOp {
    Unary(UnaryOp),
    /// (min, max, class) ranges as terrano's reclassify takes them
    Reclassify(Vec<(f64, f64, f64)>),
}

pub struct MapAlgebra {
    op: AlgebraOp,
}

impl MapAlgebra {
    pub fn new(op: AlgebraOp) -> Self {
        MapAlgebra { op }
    }
}

impl Transform for MapAlgebra {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &RasterChunk) -> Result<RasterChunk> {
        let bands = input
            .bands
            .bands()
            .iter()
            .map(|b| match &self.op {
                AlgebraOp::Unary(op) => b.apply_unary(op),
                AlgebraOp::Reclassify(classes) => reclassify(b, classes),
            })
            .collect();
        let full = RasterChunk {
            bands: BandedRaster::new(bands).expect("uniform bands"),
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        };
        Ok(full.crop_to(&out.bbox))
    }
}
