use std::collections::{BTreeMap, HashMap};

use crate::caps::{
    Caps, CapsPattern, CapsSet, Constraint, FieldMask, RasterPattern, SetField, ValueKind,
    VectorPattern,
};
use crate::chunk::{Chunk, VectorChunk, VectorFeature, clip_geometry};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, WindowReq};
use serde_json::Value;
use terrano_core::{contours, polygonize};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{Coord, LineString, MultiLineString, MultiPolygon, Polygon, Ring};

pub(crate) fn one_band(value_kind: ValueKind) -> CapsSet {
    CapsSet::one(CapsPattern::Raster(RasterPattern {
        bands: SetField::one(1),
        value_kind: SetField::one(value_kind),
        ..RasterPattern::default()
    }))
}

fn raster_to_features(value_kind: ValueKind) -> Constraint {
    Constraint::Derived {
        input: one_band(value_kind),
        passthrough: FieldMask {
            crs: true,
            resolution: true,
            chunk_px: true,
            dtype: false,
            bands: false,
            value_kind: false,
        },
        output: CapsPattern::Vector(VectorPattern::default()),
    }
}

// flipping the sign bit makes the u64 ids sort like the i64 indices
fn ordered_feature_id(index: i64) -> u64 {
    (index as u64) ^ (1 << 63)
}

struct TileAxis {
    start: f64,
    end: f64,
    cells: i64,
    step: f64,
}

impl TileAxis {
    fn x(out: &WindowReq) -> TileAxis {
        TileAxis {
            start: out.bbox.min_x,
            end: out.bbox.max_x,
            cells: (out.bbox.width() / out.resolution).round() as i64,
            step: out.resolution,
        }
    }

    fn y(out: &WindowReq) -> TileAxis {
        TileAxis {
            start: out.bbox.max_y,
            end: out.bbox.min_y,
            cells: (out.bbox.height() / out.resolution).round() as i64,
            step: -out.resolution,
        }
    }

    fn position(&self, cell: i64, within: f64) -> f64 {
        // a seam is the same f64 in both tiles beside it
        if 2 * cell < self.cells {
            self.start + (cell as f64 + within) * self.step
        } else {
            self.end - ((self.cells - cell) as f64 - within) * self.step
        }
    }
}

pub struct Contours {
    pub interval: f64,
    pub base: f64,
}

impl Transform for Contours {
    fn identity(&self) -> String {
        format!("Contours interval={:?} base={:?}", self.interval, self.base)
    }

    fn constraint(&self) -> Constraint {
        raster_to_features(ValueKind::Continuous)
    }

    fn configure(&mut self, _input: &Caps, _output: &Caps) -> Result<()> {
        if !(self.interval.is_finite() && self.interval > 0.0) {
            return Err(Error::InvalidGraph(format!(
                "contours need a positive interval, got {}",
                self.interval
            )));
        }
        if !self.base.is_finite() {
            return Err(Error::InvalidGraph(format!(
                "contours need a finite base, got {}",
                self.base
            )));
        }
        Ok(())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        out.with_window(out.bbox.expand(out.resolution), out.resolution)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        dirty.expand(resolution)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let band = input
            .bands
            .band(0)
            .ok_or(Error::Source("empty input chunk".into()))?;
        let (x_axis, y_axis) = (TileAxis::x(out), TileAxis::y(out));
        let halo_columns = ((out.bbox.min_x - input.bbox.min_x) / out.resolution).round() as i64;
        let halo_rows = ((input.bbox.max_y - out.bbox.max_y) / out.resolution).round() as i64;
        let mut levels: BTreeMap<i64, (f64, Vec<LineString>)> = BTreeMap::new();
        for line in contours(band, self.interval, self.base) {
            let coords = line
                .vertices
                .iter()
                .map(|v| {
                    Coord::new(
                        x_axis.position(v.col as i64 - halo_columns, 0.5 + v.col_fraction),
                        y_axis.position(v.row as i64 - halo_rows, 0.5 + v.row_fraction),
                    )
                })
                .collect();
            levels
                .entry(line.level_index)
                .or_insert_with(|| (line.level, Vec::new()))
                .1
                .push(LineString::new(coords));
        }
        let features = levels
            .into_iter()
            .filter_map(|(index, (elevation, lines))| {
                let lines = FeatureGeometry::MultiLineString(MultiLineString::new(lines));
                let geometry = clip_geometry(&lines, &out.bbox).into_iter().next()?;
                Some(VectorFeature {
                    id: ordered_feature_id(index),
                    geometry,
                    properties: HashMap::from([("elevation".to_string(), Value::from(elevation))]),
                })
            })
            .collect();
        Ok(Chunk::Vector(VectorChunk::new(
            features,
            out.bbox,
            out.resolution,
            input.crs,
        )))
    }
}

// f64 represents every integer up to this one and not all above it
const LARGEST_EXACT_INTEGER: f64 = 9_007_199_254_740_992.0;

fn class_code(value: f64) -> Result<i64> {
    if value.fract() != 0.0 || value.abs() > LARGEST_EXACT_INTEGER {
        return Err(Error::Source(format!(
            "polygonize takes integer class codes, the raster holds {value}"
        )));
    }
    Ok(value as i64)
}

pub struct Polygonize;

impl Transform for Polygonize {
    fn identity(&self) -> String {
        "Polygonize".into()
    }

    fn constraint(&self) -> Constraint {
        raster_to_features(ValueKind::Categorical)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?.crop_to(&out.bbox);
        let band = input
            .bands
            .band(0)
            .ok_or(Error::Source("empty input chunk".into()))?;
        let (x_axis, y_axis) = (TileAxis::x(out), TileAxis::y(out));
        let mut cells = band.clone();
        cells.cell_size = 1.0;
        let mut classes: BTreeMap<i64, Vec<Polygon>> = BTreeMap::new();
        for region in polygonize(&cells) {
            let code = class_code(region.value)?;
            // terrano's ring vertices are whole cell corners, column then row
            let mut rings = region.rings.iter().map(|ring| {
                Ring::new(
                    ring.iter()
                        .map(|&(column, row)| {
                            Coord::new(
                                x_axis.position(column as i64, 0.0),
                                y_axis.position(row as i64, 0.0),
                            )
                        })
                        .collect(),
                )
            });
            let exterior = rings.next().expect("terrano puts the exterior first");
            classes
                .entry(code)
                .or_default()
                .push(Polygon::new(exterior, rings.collect()));
        }
        let features = classes
            .into_iter()
            .map(|(code, polygons)| VectorFeature {
                id: ordered_feature_id(code),
                geometry: FeatureGeometry::MultiPolygon(MultiPolygon::new(polygons)),
                properties: HashMap::from([("value".to_string(), Value::from(code))]),
            })
            .collect();
        Ok(Chunk::Vector(VectorChunk::new(
            features,
            out.bbox,
            out.resolution,
            input.crs,
        )))
    }
}
