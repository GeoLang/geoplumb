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
    Coord, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, Ring,
    clip_linestring_rect, clip_polygon_rect,
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

/// a georeferenced tensor over one tile window. `data` is contiguous CHW:
/// channel-major, and inside a channel row-major with rows growing down
/// from the top of the bbox, the raster order. its length is always
/// channels * width * height, both derived from the bbox and resolution
#[derive(Debug, Clone)]
pub struct TensorChunk {
    pub data: Vec<f32>,
    pub channels: u16,
    pub bbox: Bbox,
    pub resolution: f64,
    pub crs: Crs,
}

#[derive(Debug, Clone)]
pub enum Chunk {
    Raster(RasterChunk),
    PointCloud(PointChunk),
    Vector(VectorChunk),
    Tensor(TensorChunk),
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

    pub fn tensor(&self) -> Result<&TensorChunk> {
        match self {
            Chunk::Tensor(t) => Ok(t),
            _ => Err(Error::Kind("tensor")),
        }
    }

    pub fn into_tensor(self) -> Result<TensorChunk> {
        match self {
            Chunk::Tensor(t) => Ok(t),
            _ => Err(Error::Kind("tensor")),
        }
    }

    pub fn byte_size(&self) -> usize {
        match self {
            Chunk::Raster(r) => r.byte_size(),
            Chunk::PointCloud(p) => p.byte_size(),
            Chunk::Vector(v) => v.byte_size(),
            Chunk::Tensor(t) => t.byte_size(),
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

    /// merge the fragments of each source feature back into one feature,
    /// the inverse of tile clipping. one feature per id, ordered by id,
    /// properties from the first fragment.
    ///
    /// this is a driver-side call, not a graph element: a transform computes
    /// one tile of its own grid at a time and only ever sees the fragments
    /// inside that tile, so no in-graph element can see two fragments of one
    /// feature. dissolve after an assembled pull instead
    pub fn dissolve(&self) -> VectorChunk {
        let mut groups: Vec<(u64, Vec<&VectorFeature>)> = Vec::new();
        for f in &self.features {
            match groups.iter_mut().find(|(id, _)| *id == f.id) {
                Some((_, group)) => group.push(f),
                None => groups.push((f.id, vec![f])),
            }
        }
        groups.sort_by_key(|(id, _)| *id);
        let features = groups
            .iter()
            .filter_map(|(id, group)| {
                let parts: Vec<&FeatureGeometry> = group.iter().map(|f| &f.geometry).collect();
                let geometry = merge_fragments(&parts)?;
                Some(VectorFeature {
                    id: *id,
                    geometry,
                    properties: group[0].properties.clone(),
                })
            })
            .collect();
        VectorChunk::new(features, self.bbox, self.resolution, self.crs)
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
        FeatureGeometry::MultiPoint(mp) => mp.points().len(),
        FeatureGeometry::LineString(l) => l.coords().len(),
        FeatureGeometry::MultiLineString(mls) => {
            mls.linestrings().iter().map(|l| l.coords().len()).sum()
        }
        FeatureGeometry::Polygon(p) => polygon_coord_count(p),
        FeatureGeometry::MultiPolygon(mp) => mp.polygons().iter().map(polygon_coord_count).sum(),
        FeatureGeometry::GeometryCollection(members) => {
            members.iter().map(geometry_coord_count).sum()
        }
    }
}

fn polygon_coord_count(p: &Polygon) -> usize {
    p.exterior().coords().len()
        + p.interiors()
            .iter()
            .map(|r| r.coords().len())
            .sum::<usize>()
}

/// clip one geometry to a tile window. points and lines use the tile
/// membership convention, lines splitting into parts where they leave the
/// window, polygon rings clip independently (Sutherland-Hodgman, exact
/// float math), which keeps even-odd fill correct for pixel centers inside
/// the window. multi geometries and collections stay one fragment, holding
/// whatever survived
pub fn clip_geometry(geometry: &FeatureGeometry, bbox: &Bbox) -> Vec<FeatureGeometry> {
    match geometry {
        FeatureGeometry::Point(p) => {
            if tile_contains(bbox, p.0.x, p.0.y) {
                vec![FeatureGeometry::Point(Point(p.0))]
            } else {
                Vec::new()
            }
        }
        FeatureGeometry::MultiPoint(mp) => {
            let kept: Vec<Point> = mp
                .points()
                .iter()
                .filter(|p| tile_contains(bbox, p.0.x, p.0.y))
                .copied()
                .collect();
            if kept.is_empty() {
                Vec::new()
            } else {
                vec![FeatureGeometry::MultiPoint(MultiPoint::new(kept))]
            }
        }
        FeatureGeometry::LineString(l) => clip_line_parts(l.coords(), bbox)
            .into_iter()
            .map(|part| FeatureGeometry::LineString(LineString::new(part)))
            .collect(),
        FeatureGeometry::MultiLineString(mls) => {
            let parts: Vec<LineString> = mls
                .linestrings()
                .iter()
                .flat_map(|l| clip_line_parts(l.coords(), bbox))
                .map(LineString::new)
                .collect();
            if parts.is_empty() {
                Vec::new()
            } else {
                vec![FeatureGeometry::MultiLineString(MultiLineString::new(
                    parts,
                ))]
            }
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
        FeatureGeometry::GeometryCollection(members) => {
            let kept: Vec<FeatureGeometry> = members
                .iter()
                .flat_map(|m| clip_geometry(m, bbox))
                .collect();
            if kept.is_empty() {
                Vec::new()
            } else {
                vec![FeatureGeometry::GeometryCollection(kept)]
            }
        }
    }
}

/// tile membership for lines, the half-open rule points already follow.
/// rect clipping is closed on all four sides, so a line lying exactly on a
/// seam survives in both neighbouring tiles: drop the parts that sit
/// entirely on an excluded edge, since the neighbour holds that edge as its
/// min_x or max_y and keeps them. parts of fewer than two coords carry no
/// length and go too
fn clip_line_parts(coords: &[Coord], bbox: &Bbox) -> Vec<Vec<Coord>> {
    clip_linestring_rect(coords, bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y)
        .into_iter()
        .filter(|part| {
            part.len() >= 2
                && !part.iter().all(|c| c.x == bbox.max_x)
                && !part.iter().all(|c| c.y == bbox.min_y)
        })
        .collect()
}

fn clip_polygon_to(p: &Polygon, bbox: &Bbox) -> Option<Polygon> {
    let clip = |coords: &[Coord]| {
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

#[derive(PartialEq, Clone, Copy)]
enum Class {
    Point,
    Line,
    Polygon,
    Collection,
}

fn class(geometry: &FeatureGeometry) -> Class {
    match geometry {
        FeatureGeometry::Point(_) | FeatureGeometry::MultiPoint(_) => Class::Point,
        FeatureGeometry::LineString(_) | FeatureGeometry::MultiLineString(_) => Class::Line,
        FeatureGeometry::Polygon(_) | FeatureGeometry::MultiPolygon(_) => Class::Polygon,
        FeatureGeometry::GeometryCollection(_) => Class::Collection,
    }
}

/// merge one feature's fragments. a lone fragment passes through untouched,
/// so an unsplit feature comes back exactly as it went in
fn merge_fragments(parts: &[&FeatureGeometry]) -> Option<FeatureGeometry> {
    match parts {
        [] => None,
        [single] => Some((*single).clone()),
        _ => merge_class(parts),
    }
}

fn merge_class(parts: &[&FeatureGeometry]) -> Option<FeatureGeometry> {
    match class(parts.first()?) {
        Class::Polygon => merge_polygonal(parts),
        Class::Line => merge_lineal(parts),
        Class::Point => merge_puntal(parts),
        Class::Collection => merge_collection(parts),
    }
}

fn merge_polygonal(parts: &[&FeatureGeometry]) -> Option<FeatureGeometry> {
    let mut polys: Vec<&Polygon> = Vec::new();
    for part in parts {
        match part {
            FeatureGeometry::Polygon(p) => polys.push(p),
            FeatureGeometry::MultiPolygon(mp) => polys.extend(mp.polygons()),
            _ => {}
        }
    }
    let (first, rest) = polys.split_first()?;
    let mut merged = MultiPolygon::new(vec![(*first).clone()]);
    for p in rest {
        merged = topoi_core::union(&merged, *p);
    }
    match merged.polygons() {
        [] => None,
        [single] => Some(FeatureGeometry::Polygon(single.clone())),
        _ => Some(FeatureGeometry::MultiPolygon(merged.clone())),
    }
}

fn merge_lineal(parts: &[&FeatureGeometry]) -> Option<FeatureGeometry> {
    let mut pieces: Vec<Vec<Coord>> = Vec::new();
    for part in parts {
        match part {
            FeatureGeometry::LineString(l) => pieces.push(l.coords().to_vec()),
            FeatureGeometry::MultiLineString(mls) => {
                pieces.extend(mls.linestrings().iter().map(|l| l.coords().to_vec()))
            }
            _ => {}
        }
    }
    let stitched = stitch(pieces);
    match stitched.len() {
        0 => None,
        1 => Some(FeatureGeometry::LineString(LineString::new(
            stitched.into_iter().next().expect("one piece"),
        ))),
        _ => Some(FeatureGeometry::MultiLineString(MultiLineString::new(
            stitched.into_iter().map(LineString::new).collect(),
        ))),
    }
}

/// join pieces whose endpoints meet. adjacent tiles solve the same
/// Liang-Barsky ratio for a seam crossing, so the two seam vertices start
/// out identical, but re-clipping rebuilds a kept endpoint as start plus
/// (end - start) and that can land an ulp away, hence the ulp-scale match
fn stitch(pieces: Vec<Vec<Coord>>) -> Vec<Vec<Coord>> {
    let meets = |a: Option<&Coord>, b: Option<&Coord>| match (a, b) {
        (Some(a), Some(b)) => same_coord(a.x, b.x) && same_coord(a.y, b.y),
        _ => false,
    };
    let mut pending = pieces;
    let mut out: Vec<Vec<Coord>> = Vec::new();
    while !pending.is_empty() {
        let mut piece = pending.remove(0);
        loop {
            if let Some(k) = pending.iter().position(|p| meets(p.first(), piece.last())) {
                let next = pending.remove(k);
                piece.extend_from_slice(&next[1..]);
                continue;
            }
            if let Some(k) = pending.iter().position(|p| meets(p.last(), piece.first())) {
                let mut prev = pending.remove(k);
                prev.extend_from_slice(&piece[1..]);
                piece = prev;
                continue;
            }
            break;
        }
        out.push(piece);
    }
    out
}

fn same_coord(a: f64, b: f64) -> bool {
    (a - b).abs() <= 8.0 * f64::EPSILON * a.abs().max(b.abs()).max(1.0)
}

/// tile membership gives each point to exactly one tile, so members concat
/// without dedupe
fn merge_puntal(parts: &[&FeatureGeometry]) -> Option<FeatureGeometry> {
    let mut points: Vec<Point> = Vec::new();
    for part in parts {
        match part {
            FeatureGeometry::Point(p) => points.push(*p),
            FeatureGeometry::MultiPoint(mp) => points.extend(mp.points()),
            _ => {}
        }
    }
    match points.len() {
        0 => None,
        1 => Some(FeatureGeometry::Point(points[0])),
        _ => Some(FeatureGeometry::MultiPoint(MultiPoint::new(points))),
    }
}

/// collection fragments merge classwise: every member of every fragment
/// joins the bucket for its geometry class, buckets keep first-seen order
fn merge_collection(parts: &[&FeatureGeometry]) -> Option<FeatureGeometry> {
    let mut buckets: Vec<(Class, Vec<&FeatureGeometry>)> = Vec::new();
    for part in parts {
        let members: &[FeatureGeometry] = match part {
            FeatureGeometry::GeometryCollection(m) => m,
            other => std::slice::from_ref(*other),
        };
        for m in members {
            let c = class(m);
            match buckets.iter_mut().find(|(bc, _)| *bc == c) {
                Some((_, bucket)) => bucket.push(m),
                None => buckets.push((c, vec![m])),
            }
        }
    }
    let merged: Vec<FeatureGeometry> = buckets
        .iter()
        .filter_map(|(_, bucket)| merge_class(bucket))
        .collect();
    (!merged.is_empty()).then_some(FeatureGeometry::GeometryCollection(merged))
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

impl TensorChunk {
    pub fn width(&self) -> usize {
        (self.bbox.width() / self.resolution).round() as usize
    }

    pub fn height(&self) -> usize {
        (self.bbox.height() / self.resolution).round() as usize
    }

    pub fn byte_size(&self) -> usize {
        self.data.len() * size_of::<f32>()
    }

    /// the row-major plane of one channel
    pub fn channel(&self, c: usize) -> &[f32] {
        let plane = self.width() * self.height();
        &self.data[c * plane..(c + 1) * plane]
    }

    /// crop to a bbox that must lie on this chunk's pixel grid
    pub fn crop_to(&self, bbox: &Bbox) -> TensorChunk {
        let res = self.resolution;
        let (width, height) = (self.width(), self.height());
        let col0 = ((bbox.min_x - self.bbox.min_x) / res).round().max(0.0) as usize;
        let row0 = ((self.bbox.max_y - bbox.max_y) / res).round().max(0.0) as usize;
        let cols = ((bbox.width() / res).round() as usize).min(width.saturating_sub(col0));
        let rows = ((bbox.height() / res).round() as usize).min(height.saturating_sub(row0));
        let mut data = Vec::with_capacity(self.channels as usize * cols * rows);
        for c in 0..self.channels as usize {
            let plane = c * width * height;
            for r in 0..rows {
                let src = plane + (row0 + r) * width + col0;
                data.extend_from_slice(&self.data[src..src + cols]);
            }
        }
        TensorChunk {
            data,
            channels: self.channels,
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
