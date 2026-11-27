use std::io::{Seek, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use terrano_core::{CogParams, TypedBand, typed_tile_bytes, write_cog_tiles};

use crate::caps::{Caps, Dtype};
use crate::chunk::RasterChunk;
use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::graph::NodeId;
use crate::resample::resample_to_grid;
use crate::window::{Bbox, TimeInterval, WindowReq};

pub const COG_TILE_PX: usize = 512;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExportGrid {
    pub origin_x: f64,
    pub origin_y: f64,
    pub pixel_size: f64,
    pub cols: usize,
    pub rows: usize,
}

impl ExportGrid {
    pub fn bbox(&self) -> Bbox {
        Bbox::new(
            self.origin_x,
            self.origin_y - self.rows as f64 * self.pixel_size,
            self.origin_x + self.cols as f64 * self.pixel_size,
            self.origin_y,
        )
    }

    pub fn level(&self, level: usize) -> ExportGrid {
        let factor = 1usize << level;
        ExportGrid {
            pixel_size: self.pixel_size * factor as f64,
            cols: self.cols.div_ceil(factor),
            rows: self.rows.div_ceil(factor),
            ..*self
        }
    }

    fn tile(&self, tile_col: usize, tile_row: usize) -> ExportGrid {
        let first_col = tile_col * COG_TILE_PX;
        let first_row = tile_row * COG_TILE_PX;
        ExportGrid {
            origin_x: self.origin_x + first_col as f64 * self.pixel_size,
            origin_y: self.origin_y - first_row as f64 * self.pixel_size,
            pixel_size: self.pixel_size,
            cols: (self.cols - first_col).min(COG_TILE_PX),
            rows: (self.rows - first_row).min(COG_TILE_PX),
        }
    }
}

const MAX_OVERVIEW_LEVELS: f64 = 5.0;

pub fn overview_levels(cols: usize, rows: usize) -> u32 {
    let max_side = cols.max(rows) as f64;
    (max_side / COG_TILE_PX as f64)
        .log2()
        .ceil()
        .clamp(0.0, MAX_OVERVIEW_LEVELS) as u32
}

struct SetWhenDropped(Arc<AtomicBool>);

impl Drop for SetWhenDropped {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

pub async fn write_cog<W: Write + Seek + Send + 'static>(
    engine: Arc<Engine>,
    node: NodeId,
    grid: ExportGrid,
    time: Option<TimeInterval>,
    mut writer: W,
) -> Result<W> {
    let Caps::Raster(caps) = engine.caps(node) else {
        return Err(Error::Kind("raster"));
    };
    let epsg = u16::try_from(caps.crs.0).map_err(|_| {
        terrano_core::Error::InvalidInput(format!(
            "{} does not fit the 16 bit epsg key of a geotiff",
            caps.crs
        ))
    })?;
    let bands = usize::from(caps.bands);
    let value_kinds = caps.value_kinds.clone();
    let first = caps.dtypes.band(0);
    let one_dtype = (1..bands).all(|band| caps.dtypes.band(band) == first);
    // a cog level has one sample format and one nodata value for every band
    let dtype = if one_dtype { first } else { Dtype::F64 };
    let params = CogParams {
        tile_width: COG_TILE_PX as u32,
        tile_height: COG_TILE_PX as u32,
        overview_levels: overview_levels(grid.cols, grid.rows),
        epsg,
        origin_x: grid.origin_x,
        origin_y: grid.origin_y,
        pixel_width: grid.pixel_size,
        pixel_height: grid.pixel_size,
        deflate: true,
        format: dtype.format(),
        nodata: dtype.integer_nodata(),
        palette: engine.palette(node).map(<[[u16; 3]]>::to_vec),
    };
    let runtime = tokio::runtime::Handle::current();
    let dropped = Arc::new(AtomicBool::new(false));
    let _set_when_dropped = SetWhenDropped(Arc::clone(&dropped));
    tokio::task::spawn_blocking(move || -> Result<W> {
        let mut failed_pull = None;
        let written = write_cog_tiles(
            grid.cols,
            grid.rows,
            bands,
            &params,
            |level, tile_col, tile_row| {
                if dropped.load(Ordering::Relaxed) {
                    return Err(terrano_core::Error::InvalidInput(
                        "the export was dropped before its last tile".into(),
                    ));
                }
                let tile = grid.level(level).tile(tile_col, tile_row);
                let bbox = tile.bbox();
                let request = WindowReq {
                    bbox,
                    resolution: tile.pixel_size,
                    time,
                };
                let exact = runtime
                    .block_on(engine.pull(node, request))
                    .and_then(|chunk| {
                        let raster = chunk.raster()?;
                        Ok(resample_to_grid(
                            raster,
                            &bbox,
                            tile.cols,
                            tile.rows,
                            &value_kinds,
                        ))
                    });
                match exact {
                    Ok(exact) => tile_bytes(exact, dtype, &params),
                    Err(error) => {
                        let message = error.to_string();
                        failed_pull = Some(error);
                        Err(terrano_core::Error::InvalidInput(message))
                    }
                }
            },
            &mut writer,
        );
        if let Some(error) = failed_pull {
            return Err(error);
        }
        written?;
        Ok(writer)
    })
    .await
    .expect("cog export task panicked")
}

fn tile_bytes(
    exact: RasterChunk,
    dtype: Dtype,
    params: &CogParams,
) -> core::result::Result<Vec<u8>, terrano_core::Error> {
    let bands: Vec<TypedBand> = match dtype {
        Dtype::F64 => exact
            .bands
            .into_rasters()
            .into_iter()
            .map(TypedBand::F64)
            .collect(),
        _ => exact.bands.into_bands(),
    };
    // typed_tile_bytes fills cells past the band edge with nodata, or nan for floats
    typed_tile_bytes(&bands, 0, 0, params)
}
