use std::io::Cursor;

use crate::bands::finite_value;
use crate::caps::{Crs, Dtype};
use crate::chunk::RasterChunk;
use crate::error::{Error, Result};
use terrano_core::{CogParams, TypedBand, typed_tile_bytes, write_cog_tiles};

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

pub fn geotiff(chunk: &RasterChunk, palette: Option<&[[u16; 3]]>) -> Result<Vec<u8>> {
    let Crs(epsg) = chunk.crs;
    let bands = chunk.bands.bands();
    let first = Dtype::of(&bands[0]);
    let one_dtype = bands.iter().all(|band| Dtype::of(band) == first);
    // a tiff has one sample format and one nodata value for every band
    let dtype = if one_dtype { first } else { Dtype::F64 };
    let bands: Vec<TypedBand> = match dtype {
        Dtype::F64 => chunk
            .bands
            .clone()
            .into_rasters()
            .into_iter()
            .map(TypedBand::F64)
            .collect(),
        _ => bands.to_vec(),
    };
    let (width, height) = (chunk.width(), chunk.height());
    let params = CogParams {
        // tiff 6.0 tile sides are multiples of 16
        tile_width: width.next_multiple_of(16) as u32,
        tile_height: height.next_multiple_of(16) as u32,
        overview_levels: 0,
        epsg: epsg as u16,
        origin_x: chunk.bbox.min_x,
        origin_y: chunk.bbox.max_y,
        pixel_width: chunk.resolution,
        pixel_height: chunk.resolution,
        deflate: false,
        format: dtype.format(),
        nodata: dtype.integer_nodata(),
        palette: palette.map(<[[u16; 3]]>::to_vec),
    };
    let mut out = Cursor::new(Vec::new());
    write_cog_tiles(
        width,
        height,
        bands.len(),
        &params,
        |_, tile_col, tile_row| typed_tile_bytes(&bands, tile_col, tile_row, &params),
        &mut out,
    )?;
    Ok(out.into_inner())
}
