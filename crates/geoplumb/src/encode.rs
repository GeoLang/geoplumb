//! chunk encoders for the demo drivers: grayscale png with alpha for
//! nodata, and geotiff via terrano

use crate::bands::finite_value;
use crate::caps::Crs;
use crate::chunk::RasterChunk;
use crate::error::{Error, Result};
use terrano_core::{
    BandedRaster, GeoTiffMetadata, Raster, SampleFormat, TypedBand, write_geotiff_bands,
};

pub fn png_gray(chunk: &RasterChunk, min: f64, max: f64) -> Result<Vec<u8>> {
    let band = chunk
        .bands
        .band(0)
        .ok_or(Error::Source("empty chunk".into()))?;
    let (w, h) = (chunk.width(), chunk.height());
    let mut pixels = Vec::with_capacity(w * h * 2);
    let span = (max - min).max(f64::EPSILON);
    for index in 0..w * h {
        match finite_value(band, index) {
            None => pixels.extend_from_slice(&[0, 0]),
            Some(v) => {
                let g = (((v - min) / span).clamp(0.0, 1.0) * 255.0).round() as u8;
                pixels.extend_from_slice(&[g, 255]);
            }
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

fn with_sentinel(band: &TypedBand) -> Raster {
    let mut raster = band.to_raster().into_owned();
    if let Some(sentinel) = band.integer_nodata() {
        for value in raster.data_mut() {
            if value.is_nan() {
                *value = sentinel;
            }
        }
    }
    raster
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
    let bands = chunk.bands.bands();
    let first = bands[0].format();
    let format = if bands.iter().all(|band| band.format() == first) {
        first
    } else {
        // write_geotiff_bands writes one sample format for every band
        SampleFormat::F64
    };
    let rasters = bands
        .iter()
        .map(|band| match format {
            SampleFormat::F64 => band.to_raster().into_owned(),
            _ => with_sentinel(band),
        })
        .collect();
    let mut out = Vec::new();
    write_geotiff_bands(&BandedRaster::new(rasters)?, &meta, format, &mut out)?;
    Ok(out)
}
