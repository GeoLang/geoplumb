use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use terrano_core::grib::{GribField, GridTemplate, decode_grib_field, scan_grib};

use crate::caps::{Crs, ValueKind};
use crate::elements::RasterSrc;
use crate::elements::stored_grid::{StoredGrid, north_up_source};
use crate::error::{Error, Result};

impl RasterSrc {
    pub fn from_grib(
        path: impl AsRef<Path>,
        message: usize,
        value_kind: ValueKind,
    ) -> Result<RasterSrc> {
        let path = path.as_ref();
        let named = |detail: String| {
            Error::Source(format!("{} message {message}: {detail}", path.display()))
        };
        if message == 0 {
            return Err(named("messages count from 1".into()));
        }
        let file = File::open(path).map_err(|e| named(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let fields = scan_grib(&mut reader).map_err(|e| named(e.to_string()))?;
        let field = fields
            .get(message - 1)
            .ok_or_else(|| named(format!("the file holds {} messages", fields.len())))?;
        let grid = regular_lat_lon(field).map_err(named)?;
        let values = decode_grib_field(&mut reader, field).map_err(|e| named(e.to_string()))?;
        north_up_source(&values, &grid, Crs::WGS84, value_kind)
    }
}

fn regular_lat_lon(field: &GribField) -> core::result::Result<StoredGrid, String> {
    let definition = &field.grid_definition;
    let Some(grid) = definition.lat_lon else {
        return Err(match definition.template {
            GridTemplate::LatLon => {
                "the lat/lon grid lists its row lengths or gives no increments".into()
            }
            other => format!(
                "grid template 3.{} is not read, only 3.0 regular lat/lon",
                other.number()
            ),
        });
    };
    if grid.di != grid.dj {
        return Err(format!(
            "cells are {} by {} degrees, the engine takes square cells",
            grid.di, grid.dj
        ));
    }
    // wmo flag table 3.4: 0x80 rows run west, 0x40 rows stack north, the
    // lower bits are column-major, alternating and staggered layouts
    if grid.scan_mode & 0x3F != 0 {
        return Err(format!(
            "scanning mode {:08b} is not read, only the 0x80 and 0x40 flags",
            grid.scan_mode
        ));
    }
    Ok(StoredGrid {
        columns: grid.ni as usize,
        rows: grid.nj as usize,
        cell_size: grid.di,
        first_column_x: grid.lon_first,
        first_row_y: grid.lat_first,
        columns_run_west: grid.scan_mode & 0x80 != 0,
        rows_run_north: grid.scan_mode & 0x40 != 0,
        x_is_longitude: true,
    })
}
