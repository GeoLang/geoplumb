//! chunk encoders for the demo drivers: grayscale png with alpha for
//! nodata, and geotiff via terrano

use crate::caps::Crs;
use crate::chunk::RasterChunk;
use crate::error::{Error, Result};
use terrano_core::{GeoTiffMetadata, SampleFormat, write_geotiff_bands};

/// band 0 scaled from [min, max] to u8 gray, nodata rendered transparent
pub fn png_gray(chunk: &RasterChunk, min: f64, max: f64) -> Result<Vec<u8>> {
    let band = chunk
        .bands
        .band(0)
        .ok_or(Error::Source("empty chunk".into()))?;
    let (w, h) = (chunk.width(), chunk.height());
    let mut pixels = Vec::with_capacity(w * h * 2);
    let span = (max - min).max(f64::EPSILON);
    for &v in band.data() {
        if band.is_nodata(v) || !v.is_finite() {
            pixels.extend_from_slice(&[0, 0]);
        } else {
            let g = (((v - min) / span).clamp(0.0, 1.0) * 255.0).round() as u8;
            pixels.extend_from_slice(&[g, 255]);
        }
    }
    let mut out = Vec::new();
    {
        let mut enc = png::Encoder::new(&mut out, w as u32, h as u32);
        enc.set_color(png::ColorType::GrayscaleAlpha);
        enc.set_depth(png::BitDepth::Eight);
        let mut writer = enc
            .write_header()
            .map_err(|e| Error::Source(format!("png header: {e}")))?;
        writer
            .write_image_data(&pixels)
            .map_err(|e| Error::Source(format!("png data: {e}")))?;
    }
    Ok(out)
}

pub fn geotiff(chunk: &RasterChunk) -> Result<Vec<u8>> {
    let Crs(epsg) = chunk.crs;
    let meta = GeoTiffMetadata {
        origin_x: chunk.bbox.min_x,
        origin_y: chunk.bbox.max_y,
        pixel_width: chunk.resolution,
        pixel_height: chunk.resolution,
        epsg: epsg as u16,
    };
    let mut out = Vec::new();
    write_geotiff_bands(&chunk.bands, &meta, SampleFormat::F64, &mut out)?;
    Ok(out)
}
