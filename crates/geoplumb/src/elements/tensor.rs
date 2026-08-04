//! tensor elements, pure rust. `ToTensor` scales raster bands into CHW f32
//! channels, `TensorConv` runs a 3x3 kernel per channel with one pixel of
//! halo, `ToRaster` hands channels back as raster bands. inference itself
//! lives outside this repo, these are the windowed pre- and post-processing
//! ends of it

use crate::caps::{
    Caps, CapsPattern, CapsSet, Constraint, Dtype, FieldMask, RasterPattern, SetField, TensorDtype,
    TensorPattern,
};
use crate::chunk::{Chunk, RasterChunk, TensorChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, WindowReq};
use terrano_core::{BandedRaster, Raster};

/// fields a raster to tensor transform carries over: the common fields plus
/// the plane count, which couples raster bands to tensor channels so a band
/// demand narrows a channel count and vice versa
fn cross_kind_passthrough() -> FieldMask {
    FieldMask {
        crs: true,
        chunk_px: true,
        bands: true,
        dtype: false,
        resolution: false,
    }
}

/// raster bands to tensor channels: channel `i` is band `i` scaled and
/// offset, the usual model normalization. the number of scales fixes the
/// output channel count, so the input must have exactly that many bands
pub struct ToTensor {
    pub scales: Vec<f32>,
    pub offsets: Vec<f32>,
}

impl Transform for ToTensor {
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::one(CapsPattern::Raster(RasterPattern::default())),
            passthrough: cross_kind_passthrough(),
            output: CapsPattern::Tensor(TensorPattern {
                dtype: SetField::one(TensorDtype::F32),
                channels: SetField::one(self.scales.len() as u16),
                ..TensorPattern::default()
            }),
        }
    }

    fn configure(&mut self, _input: &Caps, _output: &Caps) -> Result<()> {
        if self.scales.len() != self.offsets.len() {
            return Err(Error::Source(format!(
                "to tensor: {} scales against {} offsets",
                self.scales.len(),
                self.offsets.len()
            )));
        }
        Ok(())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?.crop_to(&out.bbox);
        let channels = self.scales.len();
        if input.bands.band_count() != channels {
            return Err(Error::Source(format!(
                "to tensor: {} bands into {channels} channels",
                input.bands.band_count()
            )));
        }
        let plane = input.width() * input.height();
        let mut data = vec![f32::NAN; channels * plane];
        for (c, band) in input.bands.bands().iter().enumerate() {
            let (scale, offset) = (self.scales[c], self.offsets[c]);
            for (i, v) in band.data().iter().enumerate() {
                data[c * plane + i] = *v as f32 * scale + offset;
            }
        }
        Ok(Chunk::Tensor(TensorChunk {
            data,
            channels: channels as u16,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        }))
    }
}

/// 3x3 convolution per channel, kernel rows top to bottom. a tap on a NaN
/// or off the input window contributes nothing, zero padding, but a NaN
/// center leaves the output cell NaN
pub struct TensorConv {
    pub kernel: [[f32; 3]; 3],
}

impl Transform for TensorConv {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Tensor(TensorPattern::default())))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        WindowReq {
            bbox: out.bbox.expand(out.resolution),
            resolution: out.resolution,
        }
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        dirty.expand(resolution)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.tensor()?;
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        let (in_cols, in_rows) = (input.width(), input.height());
        let col0 = ((out.bbox.min_x - input.bbox.min_x) / res).round() as isize;
        let row0 = ((input.bbox.max_y - out.bbox.max_y) / res).round() as isize;
        let channels = input.channels as usize;
        let mut data = vec![f32::NAN; channels * cols * rows];
        for c in 0..channels {
            let plane = c * in_cols * in_rows;
            let tap = |r: isize, col: isize| -> f32 {
                if r < 0 || col < 0 || r >= in_rows as isize || col >= in_cols as isize {
                    return f32::NAN;
                }
                input.data[plane + r as usize * in_cols + col as usize]
            };
            for row in 0..rows {
                for col in 0..cols {
                    let (r, cl) = (row0 + row as isize, col0 + col as isize);
                    if tap(r, cl).is_nan() {
                        continue;
                    }
                    let mut sum = 0.0f32;
                    for ky in 0..3 {
                        for kx in 0..3 {
                            let v = tap(r + ky as isize - 1, cl + kx as isize - 1);
                            if !v.is_nan() {
                                sum += v * self.kernel[ky][kx];
                            }
                        }
                    }
                    data[c * cols * rows + row * cols + col] = sum;
                }
            }
        }
        Ok(Chunk::Tensor(TensorChunk {
            data,
            channels: input.channels,
            bbox: out.bbox,
            resolution: res,
            crs: input.crs,
        }))
    }
}

/// tensor channels back to raster bands, one band per channel, nodata NaN
pub struct ToRaster;

impl Transform for ToRaster {
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::one(CapsPattern::Tensor(TensorPattern::default())),
            passthrough: cross_kind_passthrough(),
            output: CapsPattern::Raster(RasterPattern {
                dtype: SetField::one(Dtype::F64),
                ..RasterPattern::default()
            }),
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.tensor()?.crop_to(&out.bbox);
        let (cols, rows) = (input.width(), input.height());
        let bands: Vec<Raster> = (0..input.channels as usize)
            .map(|c| {
                let data: Vec<f64> = input.channel(c).iter().map(|v| f64::from(*v)).collect();
                Raster::from_vec(cols, rows, data, input.resolution, f64::NAN)
                    .expect("to raster dims")
            })
            .collect();
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: input.bbox,
            resolution: input.resolution,
            crs: input.crs,
        }))
    }
}
