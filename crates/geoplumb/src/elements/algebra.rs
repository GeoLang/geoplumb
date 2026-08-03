//! per-cell map algebra and reclassification over terrano ops

use crate::caps::{Caps, CapsSet, Constraint};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::{Fanin, Transform};
use crate::error::{Error, Result};
use crate::resample::sample_bilinear;
use crate::window::WindowReq;
use terrano_core::{BandedRaster, BinaryOp, Raster, UnaryOp, reclassify};

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

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
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
        Ok(Chunk::Raster(full.crop_to(&out.bbox)))
    }
}

/// two-input per-cell algebra: both inputs are sampled onto the output
/// grid, then terrano's binary op runs band by band
pub struct Combine {
    op: BinaryOp,
}

impl Combine {
    pub fn new(op: BinaryOp) -> Self {
        Combine { op }
    }
}

impl Fanin for Combine {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::any_raster())
    }

    fn configure(&mut self, inputs: &[Caps], _output: &Caps) -> Result<()> {
        if inputs.len() != 2 {
            return Err(Error::InvalidGraph(format!(
                "combine needs exactly two inputs, wired with {}",
                inputs.len()
            )));
        }
        Ok(())
    }

    fn compute(&self, out: &WindowReq, inputs: &[Chunk]) -> Result<Chunk> {
        let inputs: Vec<&RasterChunk> = inputs.iter().map(Chunk::raster).collect::<Result<_>>()?;
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        let on_grid = |chunk: &RasterChunk, bi: usize| -> Raster {
            let band = chunk.bands.band(bi).expect("negotiated bands");
            let mut data = Vec::with_capacity(cols * rows);
            for row in 0..rows {
                let y = out.bbox.max_y - (row as f64 + 0.5) * res;
                for col in 0..cols {
                    let x = out.bbox.min_x + (col as f64 + 0.5) * res;
                    data.push(sample_bilinear(band, chunk, x, y).unwrap_or(band.nodata));
                }
            }
            Raster::from_vec(cols, rows, data, res, band.nodata).expect("combine dims")
        };
        let bands: Vec<Raster> = (0..inputs[0].bands.band_count())
            .map(|bi| {
                on_grid(inputs[0], bi)
                    .apply_binary(&on_grid(inputs[1], bi), &self.op)
                    .expect("equal dims by construction")
            })
            .collect();
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(bands).expect("uniform bands"),
            bbox: out.bbox,
            resolution: res,
            crs: inputs[0].crs,
        }))
    }
}
