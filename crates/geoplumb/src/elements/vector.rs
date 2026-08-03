//! vector source and rasterizer over topoi. `VecSrc` holds a feature
//! collection resident and serves tile windows with per-level douglas-peucker
//! simplification, `Rasterize` is the cross-kind element burning features
//! into rasters

use std::collections::HashMap;

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, FieldMask, RasterPattern, ResRange, SetField,
    VectorPattern,
};
use crate::chunk::{Chunk, RasterChunk, VectorChunk, VectorFeature, clip_geometry};
use crate::element::{Source, Transform};
use crate::error::{Error, Result};
use crate::window::{GridSpec, WindowReq};
use futures::future::BoxFuture;
use terrano_core::{BandedRaster, Raster};
use topoi_core::geojson::{FeatureCollection, FeatureGeometry};
use topoi_core::{Coord, GridWindow, LineString, MultiPolygon, Polygon, Ring, rasterize, simplify};

/// in-memory feature source. a level simplifies whole features first
/// (douglas-peucker at the level's resolution, deterministic, so every tile
/// sees the same simplified geometry and seams stay consistent), drops
/// features smaller than a pixel, then clips to the tile window. level 0 is
/// the raw geometry
pub struct VecSrc {
    features: Vec<VectorFeature>,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    crs: Crs,
}

impl VecSrc {
    /// errors when no feature has geometry, there is no grid to anchor.
    /// the base resolution is the median segment length of the collection,
    /// features get their stable ids here in collection order
    pub fn new(collection: FeatureCollection, crs: Crs) -> Result<Self> {
        let features: Vec<VectorFeature> = collection
            .features
            .into_iter()
            .filter_map(|f| f.geometry.map(|geometry| (geometry, f.properties)))
            .enumerate()
            .map(|(id, (geometry, properties))| VectorFeature {
                id: id as u64,
                geometry,
                properties,
            })
            .collect();
        let envelope = features
            .iter()
            .map(|f| geometry_envelope(&f.geometry))
            .reduce(|a, b| (a.0.min(b.0), a.1.min(b.1), a.2.max(b.2), a.3.max(b.3)))
            .ok_or(Error::Source("empty feature collection".into()))?;
        let mut lengths: Vec<f64> = features
            .iter()
            .flat_map(|f| segment_lengths(&f.geometry))
            .filter(|l| *l > 0.0)
            .collect();
        let base_resolution = if lengths.is_empty() {
            let dim = (envelope.2 - envelope.0).max(envelope.3 - envelope.1);
            if dim > 0.0 { dim / 1024.0 } else { 1.0 }
        } else {
            lengths.sort_by(|a, b| a.total_cmp(b));
            lengths[lengths.len() / 2]
        };
        Ok(VecSrc {
            features,
            origin_x: envelope.0,
            origin_y: envelope.3,
            base_resolution,
            crs,
        })
    }

    pub fn from_geojson(json: &str, crs: Crs) -> Result<Self> {
        let collection = topoi_core::geojson::read_geojson(json)
            .map_err(|e| Error::Source(format!("geojson read: {e}")))?;
        VecSrc::new(collection, crs)
    }
}

impl Source for VecSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Vector(VectorPattern {
            crs: SetField::one(self.crs),
            resolution: ResRange::at_least(self.base_resolution),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            base_resolution: self.base_resolution,
            chunk_px: 256,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        Box::pin(async move {
            let coarse = req.resolution > self.base_resolution;
            let mut fragments = Vec::new();
            for f in &self.features {
                let simplified;
                let geometry = if coarse {
                    match simplify_geometry(&f.geometry, req.resolution) {
                        Some(g) => {
                            simplified = g;
                            &simplified
                        }
                        None => continue,
                    }
                } else {
                    &f.geometry
                };
                for geometry in clip_geometry(geometry, &req.bbox) {
                    fragments.push(VectorFeature {
                        id: f.id,
                        geometry,
                        properties: f.properties.clone(),
                    });
                }
            }
            Ok(Chunk::Vector(VectorChunk::new(
                fragments,
                req.bbox,
                req.resolution,
                self.crs,
            )))
        })
    }
}

fn geometry_envelope(geometry: &FeatureGeometry) -> (f64, f64, f64, f64) {
    let mut env = (
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for c in all_coords(geometry) {
        env = (
            env.0.min(c.x),
            env.1.min(c.y),
            env.2.max(c.x),
            env.3.max(c.y),
        );
    }
    env
}

fn all_coords(geometry: &FeatureGeometry) -> Vec<Coord> {
    match geometry {
        FeatureGeometry::Point(p) => vec![p.0],
        FeatureGeometry::LineString(l) => l.coords().to_vec(),
        FeatureGeometry::Polygon(p) => polygon_coords(p),
        FeatureGeometry::MultiPolygon(mp) => {
            mp.polygons().iter().flat_map(polygon_coords).collect()
        }
    }
}

fn polygon_coords(p: &Polygon) -> Vec<Coord> {
    let mut coords = p.exterior().coords().to_vec();
    for hole in p.interiors() {
        coords.extend_from_slice(hole.coords());
    }
    coords
}

fn segment_lengths(geometry: &FeatureGeometry) -> Vec<f64> {
    let rings: Vec<&[Coord]> = match geometry {
        FeatureGeometry::Point(_) => Vec::new(),
        FeatureGeometry::LineString(l) => vec![l.coords()],
        FeatureGeometry::Polygon(p) => polygon_rings(p),
        FeatureGeometry::MultiPolygon(mp) => mp.polygons().iter().flat_map(polygon_rings).collect(),
    };
    rings
        .iter()
        .flat_map(|coords| coords.windows(2).map(|w| w[0].distance_to(&w[1])))
        .collect()
}

fn polygon_rings(p: &Polygon) -> Vec<&[Coord]> {
    let mut rings = vec![p.exterior().coords()];
    rings.extend(p.interiors().iter().map(|r| r.coords()));
    rings
}

/// simplify whole features for a ladder level, dropping features smaller
/// than a pixel. points are exempt from the drop
fn simplify_geometry(geometry: &FeatureGeometry, resolution: f64) -> Option<FeatureGeometry> {
    if !matches!(geometry, FeatureGeometry::Point(_)) {
        let env = geometry_envelope(geometry);
        if (env.2 - env.0).max(env.3 - env.1) < resolution {
            return None;
        }
    }
    match geometry {
        FeatureGeometry::Point(p) => Some(FeatureGeometry::Point(*p)),
        FeatureGeometry::LineString(l) => Some(FeatureGeometry::LineString(LineString::new(
            simplify(l.coords(), resolution),
        ))),
        FeatureGeometry::Polygon(p) => {
            simplify_polygon(p, resolution).map(FeatureGeometry::Polygon)
        }
        FeatureGeometry::MultiPolygon(mp) => {
            let polys: Vec<Polygon> = mp
                .polygons()
                .iter()
                .filter_map(|p| simplify_polygon(p, resolution))
                .collect();
            (!polys.is_empty()).then(|| FeatureGeometry::MultiPolygon(MultiPolygon::new(polys)))
        }
    }
}

fn simplify_polygon(p: &Polygon, resolution: f64) -> Option<Polygon> {
    // a closed ring degenerates below 4 coords (first == last), drop it
    let exterior = simplify(p.exterior().coords(), resolution);
    if exterior.len() < 4 {
        return None;
    }
    let holes = p
        .interiors()
        .iter()
        .filter_map(|h| {
            let c = simplify(h.coords(), resolution);
            (c.len() >= 4).then(|| Ring::new(c))
        })
        .collect();
    Some(Polygon::new(Ring::new(exterior), holes))
}

/// what `Rasterize` burns per feature: a constant, or a numeric property
/// looked up by name (features without it are skipped)
pub enum Burn {
    Constant(f64),
    Property(String),
}

/// vector in, single-band raster out. no halo: the coverage rules are
/// preserved under clipping at pixel-aligned tile edges, so the input
/// window equals the output window
pub struct Rasterize {
    pub burn: Burn,
}

impl Rasterize {
    fn value(&self, properties: &HashMap<String, serde_json::Value>) -> Option<f64> {
        match &self.burn {
            Burn::Constant(v) => Some(*v),
            Burn::Property(name) => properties.get(name).and_then(|v| v.as_f64()),
        }
    }
}

impl Transform for Rasterize {
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::one(CapsPattern::Vector(VectorPattern::default())),
            passthrough: FieldMask {
                crs: true,
                chunk_px: true,
                dtype: false,
                bands: false,
                resolution: false,
            },
            output: CapsPattern::Raster(RasterPattern {
                dtype: SetField::one(Dtype::F64),
                bands: SetField::one(1),
                ..RasterPattern::default()
            }),
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.vector()?;
        let res = out.resolution;
        let cols = (out.bbox.width() / res).round() as usize;
        let rows = (out.bbox.height() / res).round() as usize;
        let shapes: Vec<(FeatureGeometry, f64)> = input
            .features
            .iter()
            .filter_map(|f| self.value(&f.properties).map(|v| (f.geometry.clone(), v)))
            .collect();
        // topoi burns bottom-up rows, flip into the raster order
        let window = GridWindow {
            origin_x: out.bbox.min_x,
            origin_y: out.bbox.min_y,
            width: cols,
            height: rows,
            cell_size: res,
        };
        let grid = rasterize(&shapes, &window);
        let mut data = vec![f64::NAN; cols * rows];
        for row in 0..rows {
            let src = (rows - 1 - row) * cols;
            data[row * cols..(row + 1) * cols].copy_from_slice(&grid[src..src + cols]);
        }
        let band = Raster::from_vec(cols, rows, data, res, f64::NAN).expect("rasterize dims");
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(vec![band]).expect("single band"),
            bbox: out.bbox,
            resolution: res,
            crs: input.crs,
        }))
    }
}
