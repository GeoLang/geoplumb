//! self-describing chunks. a pull response carries its resolved grid rather
//! than trusting the request, since snapping may widen or align the window

use std::collections::HashMap;

use crate::caps::Crs;
use crate::error::{Error, Result};
use crate::window::Bbox;
use nubis_core::{Point3, PointCloud};
use terrano_core::{BandedRaster, Raster};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{
    LineString, MultiPolygon, Point, Polygon, Ring, clip_linestring_rect, clip_polygon_rect,
};

#[derive(Debug, Clone)]
pub struct RasterChunk {
    pub bands: BandedRaster,
    pub bbox: Bbox,
    pub resolution: f64,
    pub crs: Crs,
}

/// points inside one tile window. resolution is the ladder level the points
/// were thinned for, the point analogue of pixel size
#[derive(Debug, Clone)]
pub struct PointChunk {
    pub points: PointCloud,
    pub bbox: Bbox,
    pub resolution: f64,
    pub crs: Crs,
}

/// one fragment of a source feature: the piece inside a tile window. `id`
/// is the source-assigned feature identity, shared by every fragment of
/// one feature, so a later dissolve can reassemble seam-split features
#[derive(Debug, Clone)]
pub struct VectorFeature {
    pub id: u64,
    pub geometry: FeatureGeometry,
    pub properties: HashMap<String, serde_json::Value>,
}

/// features clipped to one tile window, simplified for the ladder level.
/// resolution is the simplification tolerance the fragments were cut for
#[derive(Debug, Clone)]
pub struct VectorChunk {
    pub features: Vec<VectorFeature>,
    pub bbox: Bbox,
    pub resolution: f64,
    pub crs: Crs,
    byte_size: usize,
}

/// the tensor variant is reserved here
#[derive(Debug, Clone)]
pub enum Chunk {
    Raster(RasterChunk),
    PointCloud(PointChunk),
    Vector(VectorChunk),
}

impl Chunk {
    pub fn raster(&self) -> Result<&RasterChunk> {
        match self {
            Chunk::Raster(r) => Ok(r),
            _ => Err(Error::Kind("raster")),
        }
    }

    pub fn into_raster(self) -> Result<RasterChunk> {
        match self {
            Chunk::Raster(r) => Ok(r),
            _ => Err(Error::Kind("raster")),
        }
    }

    pub fn points(&self) -> Result<&PointChunk> {
        match self {
            Chunk::PointCloud(p) => Ok(p),
            _ => Err(Error::Kind("point cloud")),
        }
    }

    pub fn into_points(self) -> Result<PointChunk> {
        match self {
            Chunk::PointCloud(p) => Ok(p),
            _ => Err(Error::Kind("point cloud")),
        }
    }

    pub fn vector(&self) -> Result<&VectorChunk> {
        match self {
            Chunk::Vector(v) => Ok(v),
            _ => Err(Error::Kind("vector")),
        }
    }

    pub fn into_vector(self) -> Result<VectorChunk> {
        match self {
            Chunk::Vector(v) => Ok(v),
            _ => Err(Error::Kind("vector")),
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Chunk::Raster(r) => r.byte_size(),
            Chunk::PointCloud(p) => p.byte_size(),
            Chunk::Vector(v) => v.byte_size(),
        }
    }
}

/// tile membership for points: x in [min, max), y in (min, max]. tiles step
/// down from the grid origin, so the top edge is the inclusive one. every
/// producer and consumer of point chunks must share this convention or
/// points on tile seams duplicate or vanish
pub fn tile_contains(bbox: &Bbox, x: f64, y: f64) -> bool {
    x >= bbox.min_x && x < bbox.max_x && y > bbox.min_y && y <= bbox.max_y
}

impl PointChunk {
    pub fn byte_size(&self) -> usize {
        self.points.len() * size_of::<Point3>()
    }

    /// keep the points inside `bbox` under the tile membership convention
    pub fn crop_to(&self, bbox: &Bbox) -> PointChunk {
        let kept: Vec<Point3> = self
            .points
            .points()
            .iter()
            .filter(|p| tile_contains(bbox, p.x, p.y))
            .copied()
            .collect();
        PointChunk {
            points: PointCloud::from_points(kept),
            bbox: *bbox,
            resolution: self.resolution,
            crs: self.crs,
        }
    }
}

impl VectorChunk {
    pub fn new(features: Vec<VectorFeature>, bbox: Bbox, resolution: f64, crs: Crs) -> VectorChunk {
        let byte_size = features
            .iter()
            .map(|f| {
                let coords = geometry_coord_count(&f.geometry);
                let props = serde_json::to_string(&f.properties).map_or(0, |s| s.len());
                16 + coords * 16 + props
            })
            .sum();
        VectorChunk {
            features,
            bbox,
            resolution,
            crs,
            byte_size,
        }
    }

    pub fn byte_size(&self) -> usize {
        self.byte_size
    }

    /// re-clip the fragments to a narrower window, dropping the ones that
    /// fall out. fragments keep their source order
    pub fn crop_to(&self, bbox: &Bbox) -> VectorChunk {
        let mut kept = Vec::new();
        for f in &self.features {
            for geometry in clip_geometry(&f.geometry, bbox) {
                kept.push(VectorFeature {
                    id: f.id,
                    geometry,
                    properties: f.properties.clone(),
                });
            }
        }
        VectorChunk::new(kept, *bbox, self.resolution, self.crs)
    }
}

fn geometry_coord_count(geometry: &FeatureGeometry) -> usize {
    match geometry {
        FeatureGeometry::Point(_) => 1,
        FeatureGeometry::LineString(l) => l.coords().len(),
        FeatureGeometry::Polygon(p) => polygon_coord_count(p),
        FeatureGeometry::MultiPolygon(mp) => mp.polygons().iter().map(polygon_coord_count).sum(),
    }
}

fn polygon_coord_count(p: &Polygon) -> usize {
    p.exterior().coords().len()
        + p.interiors()
            .iter()
            .map(|r| r.coords().len())
            .sum::<usize>()
}

/// clip one geometry to a tile window. points use the tile membership
/// convention, lines split into parts where they leave the window, polygon
/// rings clip independently (Sutherland-Hodgman, exact float math), which
/// keeps even-odd fill correct for pixel centers inside the window
pub fn clip_geometry(geometry: &FeatureGeometry, bbox: &Bbox) -> Vec<FeatureGeometry> {
    match geometry {
        FeatureGeometry::Point(p) => {
            if tile_contains(bbox, p.0.x, p.0.y) {
                vec![FeatureGeometry::Point(Point(p.0))]
            } else {
                Vec::new()
            }
        }
        FeatureGeometry::LineString(l) => {
            clip_linestring_rect(l.coords(), bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
                .into_iter()
                .map(|part| FeatureGeometry::LineString(LineString::new(part)))
                .collect()
        }
        FeatureGeometry::Polygon(p) => clip_polygon_to(p, bbox)
            .map(FeatureGeometry::Polygon)
            .into_iter()
            .collect(),
        FeatureGeometry::MultiPolygon(mp) => {
            let polys: Vec<Polygon> = mp
                .polygons()
                .iter()
                .filter_map(|p| clip_polygon_to(p, bbox))
                .collect();
            if polys.is_empty() {
                Vec::new()
            } else {
                vec![FeatureGeometry::MultiPolygon(MultiPolygon::new(polys))]
            }
        }
    }
}

fn clip_polygon_to(p: &Polygon, bbox: &Bbox) -> Option<Polygon> {
    let clip = |coords: &[topoi_core::Coord]| {
        clip_polygon_rect(coords, bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
    };
    let exterior = clip(p.exterior().coords());
    if exterior.len() < 3 {
        return None;
    }
    let holes = p
        .interiors()
        .iter()
        .filter_map(|h| {
            let c = clip(h.coords());
            (c.len() >= 3).then(|| Ring::new(c))
        })
        .collect();
    Some(Polygon::new(Ring::new(exterior), holes))
}

impl RasterChunk {
    pub fn width(&self) -> usize {
        self.bands.width()
    }

    pub fn height(&self) -> usize {
        self.bands.height()
    }

    pub fn byte_size(&self) -> usize {
        self.bands.band_count() * self.width() * self.height() * size_of::<f64>()
    }

    /// crop to a bbox that must lie on this chunk's pixel grid
    pub fn crop_to(&self, bbox: &Bbox) -> RasterChunk {
        let res = self.resolution;
        let col0 = ((bbox.min_x - self.bbox.min_x) / res).round().max(0.0) as usize;
        let row0 = ((self.bbox.max_y - bbox.max_y) / res).round().max(0.0) as usize;
        let cols = (bbox.width() / res).round() as usize;
        let rows = (bbox.height() / res).round() as usize;
        let cols = cols.min(self.width().saturating_sub(col0));
        let rows = rows.min(self.height().saturating_sub(row0));
        let bands = self
            .bands
            .bands()
            .iter()
            .map(|b| {
                let mut out = Vec::with_capacity(cols * rows);
                for r in 0..rows {
                    let src = (row0 + r) * self.width() + col0;
                    out.extend_from_slice(&b.data()[src..src + cols]);
                }
                Raster::from_vec(cols, rows, out, res, b.nodata).expect("crop dims consistent")
            })
            .collect();
        RasterChunk {
            bands: BandedRaster::new(bands).expect("uniform crop"),
            bbox: Bbox {
                min_x: self.bbox.min_x + col0 as f64 * res,
                max_y: self.bbox.max_y - row0 as f64 * res,
                max_x: self.bbox.min_x + (col0 + cols) as f64 * res,
                min_y: self.bbox.max_y - (row0 + rows) as f64 * res,
            },
            resolution: res,
            crs: self.crs,
        }
    }
}
