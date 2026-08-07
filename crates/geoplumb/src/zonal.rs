//! zonal statistics and per-step time series: pull drivers, not graph
//! elements. every transform computes one tile of its own grid and only ever
//! sees that tile, so aggregating across tiles has to sit above the graph
//! the way `materialize` and `VectorChunk::dissolve` do

use std::collections::HashMap;

use crate::chunk::{RasterChunk, VectorFeature};
use crate::engine::{Engine, align_outward};
use crate::error::{Error, Result};
use crate::graph::NodeId;
use crate::window::{Bbox, TimeInterval, WindowReq};
use terrano_core::Raster;
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{GridWindow, rasterize};

/// which number to read off a reduction. every one of these merges across
/// tiles from the same four accumulators, which is why median is absent
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Statistic {
    Mean,
    Minimum,
    Maximum,
    Sum,
    Count,
}

/// the reduction of a set of pixels, mergeable across tiles: sum and count
/// carry the mean, minimum and maximum run. nan pixels never enter it
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PixelStatistics {
    pub count: usize,
    pub sum: f64,
    /// nan while `count` is zero
    pub minimum: f64,
    pub maximum: f64,
}

impl Default for PixelStatistics {
    fn default() -> PixelStatistics {
        PixelStatistics {
            count: 0,
            sum: 0.0,
            minimum: f64::NAN,
            maximum: f64::NAN,
        }
    }
}

impl PixelStatistics {
    pub fn mean(&self) -> f64 {
        if self.count == 0 {
            f64::NAN
        } else {
            self.sum / self.count as f64
        }
    }

    pub fn value(&self, statistic: Statistic) -> f64 {
        match statistic {
            Statistic::Mean => self.mean(),
            Statistic::Minimum => self.minimum,
            Statistic::Maximum => self.maximum,
            Statistic::Sum => self.sum,
            Statistic::Count => self.count as f64,
        }
    }

    /// fold in another tile's partial. f64 min and max ignore a nan side, so
    /// an empty partial merges without special casing
    pub fn merge(&mut self, other: &PixelStatistics) {
        self.count += other.count;
        self.sum += other.sum;
        self.minimum = self.minimum.min(other.minimum);
        self.maximum = self.maximum.max(other.maximum);
    }

    /// fold in one pixel the caller has already checked for nodata
    fn add(&mut self, value: f64) {
        self.count += 1;
        self.sum += value;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
    }
}

/// one feature's reduction, keyed by the id every fragment of that feature
/// shares
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureStatistics {
    pub feature_id: u64,
    pub statistics: PixelStatistics,
}

/// reduce band 0 of `node` over `window`, grouped by the feature covering
/// each pixel. one row per feature that caught at least one valid pixel,
/// ordered by id. tiles are pulled one at a time and their partials merged,
/// so a feature spanning many tiles still gets one exact result.
///
/// features overlap by burn order, last feature of the slice wins a shared
/// pixel, the rule `Rasterize` already follows
pub async fn zonal_statistics(
    engine: &Engine,
    node: NodeId,
    features: &[VectorFeature],
    window: WindowReq,
) -> Result<Vec<FeatureStatistics>> {
    let shapes = zone_shapes(features);
    let mut merged: HashMap<u64, PixelStatistics> = HashMap::new();
    for_each_tile(engine, node, &window, |tile| {
        accumulate_zones(&shapes, tile, &mut merged)
    })
    .await?;
    let mut rows: Vec<FeatureStatistics> = merged
        .into_iter()
        .map(|(feature_id, statistics)| FeatureStatistics {
            feature_id,
            statistics,
        })
        .collect();
    rows.sort_by_key(|row| row.feature_id);
    Ok(rows)
}

/// reduce band 0 of `node` over `window` as a single zone, no features
pub async fn window_statistics(
    engine: &Engine,
    node: NodeId,
    window: WindowReq,
) -> Result<PixelStatistics> {
    let mut merged = PixelStatistics::default();
    for_each_tile(engine, node, &window, |tile| {
        merged.merge(&reduce_band(band_zero(tile)?));
        Ok(())
    })
    .await?;
    Ok(merged)
}

/// one pull per step, steps in the order given, each yielding the per
/// feature reduction of that step's window.
///
/// every step is a distinct pull interval, so a source holding per interval
/// state holds one more set of it per step
pub async fn zonal_time_series(
    engine: &Engine,
    node: NodeId,
    features: &[VectorFeature],
    window: WindowReq,
    steps: &[TimeInterval],
) -> Result<Vec<(TimeInterval, Vec<FeatureStatistics>)>> {
    let mut series = Vec::with_capacity(steps.len());
    for step in steps {
        let rows = zonal_statistics(engine, node, features, window.with_time(Some(*step))).await?;
        series.push((*step, rows));
    }
    Ok(series)
}

/// `zonal_time_series` with the whole window as one zone
pub async fn window_time_series(
    engine: &Engine,
    node: NodeId,
    window: WindowReq,
    steps: &[TimeInterval],
) -> Result<Vec<(TimeInterval, PixelStatistics)>> {
    let mut series = Vec::with_capacity(steps.len());
    for step in steps {
        let statistics = window_statistics(engine, node, window.with_time(Some(*step))).await?;
        series.push((*step, statistics));
    }
    Ok(series)
}

fn zone_shapes(features: &[VectorFeature]) -> Vec<(FeatureGeometry, f64)> {
    // the id rides as a zone label, exact for every id below 2^53
    features
        .iter()
        .map(|f| (f.geometry.clone(), f.id as f64))
        .collect()
}

fn band_zero(tile: &RasterChunk) -> Result<&Raster> {
    tile.bands.band(0).ok_or(Error::Kind("raster with a band"))
}

fn reduce_band(band: &Raster) -> PixelStatistics {
    let mut stats = PixelStatistics::default();
    for value in band.data() {
        if band.is_nodata(*value) {
            continue;
        }
        stats.add(*value);
    }
    stats
}

/// burn the feature ids over this tile and fold its pixels straight into the
/// running partials, so nothing per tile outlives the tile
fn accumulate_zones(
    shapes: &[(FeatureGeometry, f64)],
    tile: &RasterChunk,
    merged: &mut HashMap<u64, PixelStatistics>,
) -> Result<()> {
    let values = band_zero(tile)?;
    let labels = zone_labels(shapes, tile);
    for (value, label) in values.data().iter().zip(&labels) {
        if label.is_nan() || values.is_nodata(*value) {
            continue;
        }
        merged.entry(*label as u64).or_default().add(*value);
    }
    Ok(())
}

/// feature ids as zone labels over one tile window, nan where nothing
/// burned. the geometry goes in unclipped: topoi's coverage rules burn a
/// sub-window as exactly that slice of the whole burn, so tiled aggregation
/// lands on the same pixels a single pass would
fn zone_labels(shapes: &[(FeatureGeometry, f64)], tile: &RasterChunk) -> Vec<f64> {
    let (cols, rows) = (tile.width(), tile.height());
    let window = GridWindow {
        origin_x: tile.bbox.min_x,
        origin_y: tile.bbox.min_y,
        width: cols,
        height: rows,
        cell_size: tile.resolution,
    };
    let burned = rasterize(shapes, &window);
    // topoi burns rows upward from min_y, flip into raster order
    let mut labels = vec![f64::NAN; cols * rows];
    for row in 0..rows {
        let source = (rows - 1 - row) * cols;
        labels[row * cols..(row + 1) * cols].copy_from_slice(&burned[source..source + cols]);
    }
    labels
}

/// pull `window` one chunk at a time, in the node's own chunking, so peak
/// memory is one tile however wide the window is. the chunks partition the
/// window snapped outward onto the node's pixel grid, so every pixel of it
/// reaches `per_tile` exactly once
async fn for_each_tile(
    engine: &Engine,
    node: NodeId,
    window: &WindowReq,
    mut per_tile: impl FnMut(&RasterChunk) -> Result<()>,
) -> Result<()> {
    let grid = *engine.grid(node);
    let level = grid.snap_level(window.resolution);
    let resolution = grid.resolution_at(level);
    let aligned = align_outward(&window.bbox, &grid, resolution);
    for key in grid.cover(&aligned, level) {
        let Some(bbox) = intersection(&grid.chunk_bbox(key), &aligned) else {
            continue;
        };
        let chunk = engine
            .pull(
                node,
                WindowReq {
                    bbox,
                    resolution,
                    time: window.time,
                },
            )
            .await?;
        per_tile(chunk.raster()?)?;
    }
    Ok(())
}

fn intersection(a: &Bbox, b: &Bbox) -> Option<Bbox> {
    let out = Bbox {
        min_x: a.min_x.max(b.min_x),
        min_y: a.min_y.max(b.min_y),
        max_x: a.max_x.min(b.max_x),
        max_y: a.max_y.min(b.max_y),
    };
    (out.width() > 0.0 && out.height() > 0.0).then_some(out)
}
