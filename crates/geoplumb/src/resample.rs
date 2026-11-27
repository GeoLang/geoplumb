use std::ops::Range;

use terrano_core::{Raster, TypedBand};

use crate::bands::{RasterBands, finite_value, narrowed_like, picked};
use crate::caps::{BandKinds, ValueKind};
use crate::chunk::RasterChunk;
use crate::window::Bbox;

pub fn resample_to_grid(
    src: &RasterChunk,
    bbox: &Bbox,
    cols: usize,
    rows: usize,
    value_kinds: &BandKinds,
) -> RasterChunk {
    let out_res_x = bbox.width() / cols as f64;
    let out_res_y = bbox.height() / rows as f64;
    let center = |index: usize| {
        let (row, col) = (index / cols, index % cols);
        (
            bbox.min_x + (col as f64 + 0.5) * out_res_x,
            bbox.max_y - (row as f64 + 0.5) * out_res_y,
        )
    };
    let grid = OutputGrid {
        cols,
        rows,
        cell_size: out_res_x,
        cell_size_in_input_crs: out_res_x,
    };
    let bands = src
        .bands
        .bands()
        .iter()
        .enumerate()
        .map(|(bi, band)| resampled_band(band, value_kinds.band(bi), src, &grid, center))
        .collect();
    RasterChunk {
        bands: RasterBands::new(bands).expect("uniform bands"),
        bbox: *bbox,
        resolution: out_res_x,
        crs: src.crs,
    }
}

pub(crate) struct OutputGrid {
    pub(crate) cols: usize,
    pub(crate) rows: usize,
    pub(crate) cell_size: f64,
    pub(crate) cell_size_in_input_crs: f64,
}

pub(crate) fn resampled_band(
    band: &TypedBand,
    value_kind: ValueKind,
    chunk: &RasterChunk,
    grid: &OutputGrid,
    point: impl Fn(usize) -> (f64, f64),
) -> TypedBand {
    match value_kind {
        ValueKind::Categorical => picked(band, grid.cols, grid.rows, grid.cell_size, |index| {
            let (x, y) = point(index);
            nearest_index(chunk, x, y).filter(|&at| finite_value(band, at).is_some())
        }),
        ValueKind::Peak => picked(band, grid.cols, grid.rows, grid.cell_size, |index| {
            let (x, y) = point(index);
            peak_index(chunk, x, y, grid.cell_size_in_input_crs, |at| {
                finite_value(band, at)
            })
        }),
        ValueKind::Continuous => {
            let raster = band.to_raster();
            let values = (0..grid.cols * grid.rows)
                .map(|index| {
                    let (x, y) = point(index);
                    sample_bilinear(&raster, chunk, x, y).unwrap_or(raster.nodata)
                })
                .collect();
            narrowed_like(band, grid.cols, grid.rows, grid.cell_size, values)
        }
    }
}

pub(crate) fn sample_point(
    value_kind: ValueKind,
    band: &Raster,
    chunk: &RasterChunk,
    x: f64,
    y: f64,
) -> Option<f64> {
    match value_kind {
        ValueKind::Continuous => sample_bilinear(band, chunk, x, y),
        ValueKind::Categorical | ValueKind::Peak => sample_nearest(band, chunk, x, y),
    }
}

fn sample_bilinear(band: &Raster, chunk: &RasterChunk, x: f64, y: f64) -> Option<f64> {
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
    let w = [
        (1.0 - tx) * (1.0 - ty),
        tx * (1.0 - ty),
        (1.0 - tx) * ty,
        tx * ty,
    ];
    match (v00, v01, v10, v11) {
        (Some(a), Some(b), Some(c), Some(d)) => Some(a * w[0] + b * w[1] + c * w[2] + d * w[3]),
        // some neighbors missing: nearest present one, ignoring neighbors
        // whose weight is float jitter so a point on a pixel center never
        // takes a value a whole pixel away
        _ => [v00, v01, v10, v11]
            .iter()
            .zip(w)
            .filter(|(_, w)| *w > 1e-6)
            .filter_map(|(v, w)| v.map(|v| (w, v)))
            .max_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, v)| v),
    }
}

pub(crate) fn nearest_index(chunk: &RasterChunk, x: f64, y: f64) -> Option<usize> {
    let col = ((x - chunk.bbox.min_x) / chunk.resolution).floor();
    let row = ((chunk.bbox.max_y - y) / chunk.resolution).floor();
    if col.is_nan() || row.is_nan() || col < 0.0 || row < 0.0 {
        return None;
    }
    let (row, col) = (row as usize, col as usize);
    if row >= chunk.height() || col >= chunk.width() {
        return None;
    }
    Some(row * chunk.width() + col)
}

fn centers_within(offset_cells: f64, half_width_cells: f64, len: usize) -> Range<usize> {
    let first = (offset_cells - half_width_cells - 0.5).ceil().max(0.0) as usize;
    let end = (offset_cells + half_width_cells - 0.5).ceil().max(0.0) as usize;
    first.min(len)..end.min(len)
}

pub(crate) fn peak_index(
    chunk: &RasterChunk,
    x: f64,
    y: f64,
    cell_size_in_input_crs: f64,
    value: impl Fn(usize) -> Option<f64>,
) -> Option<usize> {
    let half_width_cells = cell_size_in_input_crs / chunk.resolution / 2.0;
    let rows = centers_within(
        (chunk.bbox.max_y - y) / chunk.resolution,
        half_width_cells,
        chunk.height(),
    );
    let cols = centers_within(
        (x - chunk.bbox.min_x) / chunk.resolution,
        half_width_cells,
        chunk.width(),
    );
    let width = chunk.width();
    let highest_covered = rows
        .flat_map(|row| cols.clone().map(move |col| row * width + col))
        .filter_map(|at| value(at).map(|v| (at, v)))
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(at, _)| at);
    highest_covered.or_else(|| nearest_index(chunk, x, y).filter(|&at| value(at).is_some()))
}

fn sample_nearest(band: &Raster, chunk: &RasterChunk, x: f64, y: f64) -> Option<f64> {
    let v = band.data()[nearest_index(chunk, x, y)?];
    (!band.is_nodata(v) && v.is_finite()).then_some(v)
}

pub(crate) fn reduce_block(value_kind: ValueKind, values: &mut [f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    match value_kind {
        // std's f64 sum starts from negative zero
        ValueKind::Continuous => {
            Some(values.iter().fold(0.0, |sum, v| sum + v) / values.len() as f64)
        }
        ValueKind::Categorical => {
            values.sort_by(f64::total_cmp);
            let (mut mode, mut mode_count) = (values[0], 0);
            for run in values.chunk_by(|a, b| a == b) {
                if run.len() > mode_count {
                    (mode, mode_count) = (run[0], run.len());
                }
            }
            Some(mode)
        }
        ValueKind::Peak => values.iter().copied().max_by(f64::total_cmp),
    }
}
