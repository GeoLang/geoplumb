//! per-cell map algebra and reclassification over terrano ops

use crate::bands::RasterBands;
use crate::caps::{BandKinds, Caps, CapsSet, Constraint, Dtype, ValueKind};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::{Fanin, Transform};
use crate::error::{Error, Result};
use crate::resample::sample_point;
use crate::window::WindowReq;
use terrano_core::{BinaryOp, Raster, UnaryOp, reclassify};

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
    fn identity(&self) -> String {
        let op = match &self.op {
            AlgebraOp::Unary(UnaryOp::Add(v)) => format!("add({v:?})"),
            AlgebraOp::Unary(UnaryOp::Multiply(v)) => format!("multiply({v:?})"),
            AlgebraOp::Unary(UnaryOp::Sqrt) => "sqrt".into(),
            AlgebraOp::Unary(UnaryOp::Abs) => "abs".into(),
            AlgebraOp::Unary(UnaryOp::Log) => "log".into(),
            AlgebraOp::Reclassify(classes) => format!("reclassify({classes:?})"),
        };
        format!("MapAlgebra op={op}")
    }

    fn constraint(&self) -> Constraint {
        let value_kind = match self.op {
            AlgebraOp::Unary(_) => ValueKind::Continuous,
            AlgebraOp::Reclassify(_) => ValueKind::Categorical,
        };
        Constraint::pinned_value_kind_and_dtype(CapsSet::any_raster(), value_kind, Dtype::F64)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let bands = input
            .bands
            .bands()
            .iter()
            .map(|band| {
                let band = band.to_raster();
                match &self.op {
                    AlgebraOp::Unary(op) => band.apply_unary(op),
                    AlgebraOp::Reclassify(classes) => reclassify(&band, classes),
                }
            })
            .collect();
        let full = RasterChunk {
            bands: RasterBands::from_rasters(bands).expect("uniform bands"),
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        };
        Ok(Chunk::Raster(full.crop_to(&out.bbox)))
    }
}

pub struct Combine {
    op: BinaryOp,
    input_value_kinds: Option<[BandKinds; 2]>,
}

impl Combine {
    pub fn new(op: BinaryOp) -> Self {
        Combine {
            op,
            input_value_kinds: None,
        }
    }
}

impl Fanin for Combine {
    fn identity(&self) -> String {
        let op = match self.op {
            BinaryOp::Add => "add",
            BinaryOp::Subtract => "subtract",
            BinaryOp::Multiply => "multiply",
            BinaryOp::Divide => "divide",
            BinaryOp::Min => "min",
            BinaryOp::Max => "max",
        };
        format!("Combine op={op}")
    }

    fn constraint(&self) -> Constraint {
        Constraint::pinned_value_kind_and_dtype(
            CapsSet::any_raster(),
            ValueKind::Continuous,
            Dtype::F64,
        )
    }

    fn configure(&mut self, inputs: &[Caps], _output: &Caps) -> Result<()> {
        let [first, second] = inputs else {
            return Err(Error::InvalidGraph(format!(
                "combine needs exactly two inputs, wired with {}",
                inputs.len()
            )));
        };
        self.input_value_kinds = Some([
            first.raster().value_kinds.clone(),
            second.raster().value_kinds.clone(),
        ]);
        Ok(())
    }

    fn compute(&self, out: &WindowReq, inputs: &[Chunk]) -> Result<Chunk> {
        let inputs: Vec<&RasterChunk> = inputs.iter().map(Chunk::raster).collect::<Result<_>>()?;
        let [first_kinds, second_kinds] = self.input_value_kinds.as_ref().expect("configured");
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        let on_grid = |chunk: &RasterChunk, value_kind: ValueKind, bi: usize| -> Raster {
            let band = chunk.bands.band(bi).expect("negotiated bands").to_raster();
            let mut data = Vec::with_capacity(cols * rows);
            for row in 0..rows {
                let y = out.bbox.max_y - (row as f64 + 0.5) * res;
                for col in 0..cols {
                    let x = out.bbox.min_x + (col as f64 + 0.5) * res;
                    data.push(sample_point(value_kind, &band, chunk, x, y).unwrap_or(band.nodata));
                }
            }
            Raster::from_vec(cols, rows, data, res, band.nodata).expect("combine dims")
        };
        let bands: Vec<Raster> = (0..inputs[0].bands.band_count())
            .map(|bi| {
                on_grid(inputs[0], first_kinds.band(bi), bi)
                    .apply_binary(&on_grid(inputs[1], second_kinds.band(bi), bi), &self.op)
                    .expect("equal dims by construction")
            })
            .collect();
        Ok(Chunk::Raster(RasterChunk {
            bands: RasterBands::from_rasters(bands).expect("uniform bands"),
            bbox: out.bbox,
            resolution: res,
            crs: inputs[0].crs,
        }))
    }
}
