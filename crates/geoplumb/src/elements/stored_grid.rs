use terrano_core::{BandedRaster, Raster};

use crate::caps::{Crs, ValueKind};
use crate::elements::RasterSrc;
use crate::error::{Error, Result};

pub(crate) const CELL_TOLERANCE: f64 = 1e-3;

#[derive(Debug, Clone, Copy)]
pub(crate) struct StoredGrid {
    pub columns: usize,
    pub rows: usize,
    pub cell_size: f64,
    pub first_column_x: f64,
    pub first_row_y: f64,
    pub columns_run_west: bool,
    pub rows_run_north: bool,
    pub x_is_longitude: bool,
}

pub(crate) fn north_up_source(
    values: &[f64],
    grid: &StoredGrid,
    crs: Crs,
    value_kind: ValueKind,
) -> Result<RasterSrc> {
    let StoredGrid {
        columns,
        rows,
        cell_size,
        ..
    } = *grid;
    if columns < 2 || rows < 2 {
        return Err(Error::Source(format!(
            "a {columns} by {rows} grid has no cell spacing on one axis"
        )));
    }
    if values.len() != columns * rows {
        return Err(Error::Source(format!(
            "{} values for a {columns} by {rows} grid",
            values.len()
        )));
    }
    let west_x = match grid.columns_run_west {
        true => grid.first_column_x - (columns - 1) as f64 * cell_size,
        false => grid.first_column_x,
    };
    let north_y = match grid.rows_run_north {
        true => grid.first_row_y + (rows - 1) as f64 * cell_size,
        false => grid.first_row_y,
    };
    let (first_column, west_x) = match grid.x_is_longitude {
        true => antimeridian_split(west_x, cell_size, columns)?,
        false => (0, west_x),
    };
    let (west_to_east_columns, west_x) = match grid.x_is_longitude {
        true => antimeridian_wrap(west_x, cell_size, columns),
        false => ((0..columns).collect(), west_x),
    };
    let mut north_up = Vec::with_capacity(rows * west_to_east_columns.len());
    for row in 0..rows {
        let stored_row = if grid.rows_run_north {
            rows - 1 - row
        } else {
            row
        };
        for column in &west_to_east_columns {
            let west_to_east = (first_column + column) % columns;
            let stored_column = match grid.columns_run_west {
                true => columns - 1 - west_to_east,
                false => west_to_east,
            };
            north_up.push(values[stored_row * columns + stored_column]);
        }
    }
    let raster = Raster::from_vec(
        west_to_east_columns.len(),
        rows,
        north_up,
        cell_size,
        f64::NAN,
    )?;
    Ok(RasterSrc::new(
        BandedRaster::new(vec![raster])?,
        west_x - cell_size / 2.0,
        north_y + cell_size / 2.0,
        crs,
        value_kind,
    ))
}

fn antimeridian_wrap(west_x: f64, cell_size: f64, columns: usize) -> (Vec<usize>, f64) {
    let tolerance = CELL_TOLERANCE * cell_size;
    let mut west_to_east: Vec<usize> = (0..columns).collect();
    let global = (columns as f64 * cell_size - 360.0).abs() <= tolerance;
    let west_edge = west_x - cell_size / 2.0;
    // without the repeated column a pixel between the last centre and 180 reads its neighbour cell
    if global && west_edge < -180.0 - tolerance {
        west_to_east.push(0);
        return (west_to_east, west_x);
    }
    if global && west_edge > -180.0 + tolerance {
        west_to_east.insert(0, columns - 1);
        return (west_to_east, west_x - cell_size);
    }
    (west_to_east, west_x)
}

fn antimeridian_split(west_x: f64, cell_size: f64, columns: usize) -> Result<(usize, f64)> {
    let tolerance = CELL_TOLERANCE * cell_size;
    let east_x = west_x + (columns - 1) as f64 * cell_size;
    if east_x <= 180.0 + tolerance {
        return Ok((0, west_x));
    }
    if west_x >= 180.0 - tolerance {
        return Ok((0, west_x - 360.0));
    }
    let span = columns as f64 * cell_size;
    if (span - 360.0).abs() > tolerance {
        return Err(Error::Source(format!(
            "longitudes {west_x} to {east_x} cross 180 without covering the globe"
        )));
    }
    let first_column = ((180.0 - west_x) / cell_size - CELL_TOLERANCE).ceil() as usize;
    Ok((
        first_column,
        west_x + first_column as f64 * cell_size - 360.0,
    ))
}
