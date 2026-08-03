//! driver-side bilinear resample onto an exact output grid. the engine
//! returns chunk-aligned windows at ladder resolutions, a tile wants an
//! exact 256 px grid, this bridges the two

use crate::chunk::RasterChunk;
use crate::window::Bbox;
use terrano_core::{BandedRaster, Raster};

pub fn resample_to_grid(src: &RasterChunk, bbox: &Bbox, cols: usize, rows: usize) -> RasterChunk {
    let out_res_x = bbox.width() / cols as f64;
    let out_res_y = bbox.height() / rows as f64;
    let bands: Vec<Raster> = src
        .bands
        .bands()
        .iter()
        .map(|band| {
            let nodata = band.nodata;
            let mut data = Vec::with_capacity(cols * rows);
            for row in 0..rows {
                let y = bbox.max_y - (row as f64 + 0.5) * out_res_y;
                for col in 0..cols {
                    let x = bbox.min_x + (col as f64 + 0.5) * out_res_x;
                    data.push(sample_bilinear(band, src, x, y).unwrap_or(nodata));
                }
            }
            Raster::from_vec(cols, rows, data, out_res_x, nodata).expect("resample dims")
        })
        .collect();
    RasterChunk {
        bands: BandedRaster::new(bands).expect("uniform bands"),
        bbox: *bbox,
        resolution: out_res_x,
        crs: src.crs,
    }
}

pub(crate) fn sample_bilinear(band: &Raster, chunk: &RasterChunk, x: f64, y: f64) -> Option<f64> {
    let res = chunk.resolution;
    let fx = (x - chunk.bbox.min_x) / res - 0.5;
    let fy = (chunk.bbox.max_y - y) / res - 0.5;
    let c0 = fx.floor();
    let r0 = fy.floor();
    let tx = fx - c0;
    let ty = fy - r0;
    let sample = |r: f64, c: f64| -> Option<f64> {
        if r < 0.0 || c < 0.0 {
            return None;
        }
        let (r, c) = (r as usize, c as usize);
        if r >= band.height() || c >= band.width() {
            return None;
        }
        let v = band.data()[r * band.width() + c];
        (!band.is_nodata(v) && v.is_finite()).then_some(v)
    };
    let v00 = sample(r0, c0);
    let v01 = sample(r0, c0 + 1.0);
    let v10 = sample(r0 + 1.0, c0);
    let v11 = sample(r0 + 1.0, c0 + 1.0);
    match (v00, v01, v10, v11) {
        (Some(a), Some(b), Some(c), Some(d)) => Some(
            a * (1.0 - tx) * (1.0 - ty) + b * tx * (1.0 - ty) + c * (1.0 - tx) * ty + d * tx * ty,
        ),
        _ => v00.or(v01).or(v10).or(v11),
    }
}
