//! window-local vector to vector elements: property filter, schema map and
//! boundary clip. each works fragment by fragment and never widens its
//! request, so chunked output equals whole-window output and none of them
//! needs a halo

use std::collections::HashMap;

use crate::caps::{CapsPattern, CapsSet, Constraint, VectorPattern};
use crate::chunk::{Chunk, VectorChunk, VectorFeature};
use crate::element::Transform;
use crate::error::Result;
use crate::window::WindowReq;
use serde_json::Value;
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{
    Coord, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, contains, intersection,
};

fn vector_identity() -> Constraint {
    Constraint::Identity(CapsSet::one(CapsPattern::Vector(VectorPattern::default())))
}

/// the plan is identity and the grid is unchanged, so the input chunk covers
/// exactly `out` and the fragments need no re-clipping
fn same_window(out: &WindowReq, input: &VectorChunk, features: Vec<VectorFeature>) -> Chunk {
    Chunk::Vector(VectorChunk::new(
        features,
        out.bbox,
        out.resolution,
        input.crs,
    ))
}

/// keep the fragments whose property equals a value. strings, integers and
/// bools compare exactly, floats within an epsilon, and a missing property
/// or a type mismatch drops the feature
pub struct VecFilter {
    pub field: String,
    pub equals: Value,
}

impl VecFilter {
    pub fn new(field: impl Into<String>, equals: Value) -> Self {
        VecFilter {
            field: field.into(),
            equals,
        }
    }

    fn keeps(&self, properties: &HashMap<String, Value>) -> bool {
        match (properties.get(&self.field), &self.equals) {
            (Some(Value::String(a)), Value::String(b)) => a == b,
            (Some(Value::Bool(a)), Value::Bool(b)) => a == b,
            (Some(Value::Number(a)), Value::Number(b)) => match (a.as_i64(), b.as_i64()) {
                (Some(x), Some(y)) => x == y,
                (None, None) => match (a.as_f64(), b.as_f64()) {
                    (Some(x), Some(y)) => (x - y).abs() < f64::EPSILON,
                    _ => false,
                },
                _ => false,
            },
            _ => false,
        }
    }
}

impl Transform for VecFilter {
    fn constraint(&self) -> Constraint {
        vector_identity()
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.vector()?;
        let features = input
            .features
            .iter()
            .filter(|f| self.keeps(&f.properties))
            .cloned()
            .collect();
        Ok(same_window(out, input, features))
    }
}

/// property rewrite: drop first, then rename what survives, then add the
/// defaults a fragment does not already carry. geometry is untouched
#[derive(Default)]
pub struct VecSchema {
    pub rename: HashMap<String, String>,
    pub drop: Vec<String>,
    pub add: HashMap<String, Value>,
}

impl VecSchema {
    fn map(&self, properties: &HashMap<String, Value>) -> HashMap<String, Value> {
        let mut mapped: HashMap<String, Value> = properties
            .iter()
            .filter(|(k, _)| !self.drop.contains(k))
            .map(|(k, v)| {
                let key = self.rename.get(k).unwrap_or(k);
                (key.clone(), v.clone())
            })
            .collect();
        for (k, v) in &self.add {
            mapped.entry(k.clone()).or_insert_with(|| v.clone());
        }
        mapped
    }
}

impl Transform for VecSchema {
    fn constraint(&self) -> Constraint {
        vector_identity()
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.vector()?;
        let features = input
            .features
            .iter()
            .map(|f| VectorFeature {
                id: f.id,
                geometry: f.geometry.clone(),
                properties: self.map(&f.properties),
            })
            .collect();
        Ok(same_window(out, input, features))
    }
}

/// intersect every fragment with a static boundary. the result is a subset
/// of the fragment, so tile splitting commutes with the clip and no halo is
/// needed. fragments reduced to nothing are dropped
pub struct VecClip {
    pub boundary: MultiPolygon,
}

impl Transform for VecClip {
    fn constraint(&self) -> Constraint {
        vector_identity()
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.vector()?;
        let features = input
            .features
            .iter()
            .filter_map(|f| {
                clip_to_boundary(&f.geometry, &self.boundary).map(|geometry| VectorFeature {
                    id: f.id,
                    geometry,
                    properties: f.properties.clone(),
                })
            })
            .collect();
        Ok(same_window(out, input, features))
    }
}

fn clip_to_boundary(
    geometry: &FeatureGeometry,
    boundary: &MultiPolygon,
) -> Option<FeatureGeometry> {
    match geometry {
        FeatureGeometry::Point(p) => inside(boundary, &p.0).then(|| geometry.clone()),
        FeatureGeometry::MultiPoint(mp) => {
            let kept: Vec<Point> = mp
                .points()
                .iter()
                .filter(|p| inside(boundary, &p.0))
                .copied()
                .collect();
            (!kept.is_empty()).then(|| FeatureGeometry::MultiPoint(MultiPoint::new(kept)))
        }
        FeatureGeometry::LineString(l) => lines(clip_line(l.coords(), boundary)),
        FeatureGeometry::MultiLineString(mls) => lines(
            mls.linestrings()
                .iter()
                .flat_map(|l| clip_line(l.coords(), boundary))
                .collect(),
        ),
        FeatureGeometry::Polygon(p) => polygons(intersection(p, boundary)),
        FeatureGeometry::MultiPolygon(mp) => polygons(intersection(mp, boundary)),
        FeatureGeometry::GeometryCollection(members) => {
            let kept: Vec<FeatureGeometry> = members
                .iter()
                .filter_map(|m| clip_to_boundary(m, boundary))
                .collect();
            (!kept.is_empty()).then_some(FeatureGeometry::GeometryCollection(kept))
        }
    }
}

fn polygons(clipped: MultiPolygon) -> Option<FeatureGeometry> {
    match clipped.polygons() {
        [] => None,
        [single] => Some(FeatureGeometry::Polygon(single.clone())),
        _ => Some(FeatureGeometry::MultiPolygon(clipped.clone())),
    }
}

fn lines(parts: Vec<Vec<Coord>>) -> Option<FeatureGeometry> {
    match parts.len() {
        0 => None,
        1 => Some(FeatureGeometry::LineString(LineString::new(
            parts.into_iter().next().expect("one part"),
        ))),
        _ => Some(FeatureGeometry::MultiLineString(MultiLineString::new(
            parts.into_iter().map(LineString::new).collect(),
        ))),
    }
}

fn inside(boundary: &MultiPolygon, c: &Coord) -> bool {
    boundary.polygons().iter().any(|p| contains(p, c))
}

/// cut a line at every boundary crossing and keep the pieces whose midpoint
/// is inside, holes included. geodukt's clip leaves lines and points
/// untouched, which emits geometry outside the boundary
fn clip_line(coords: &[Coord], boundary: &MultiPolygon) -> Vec<Vec<Coord>> {
    let mut parts: Vec<Vec<Coord>> = Vec::new();
    for seg in coords.windows(2) {
        let (a, b) = (seg[0], seg[1]);
        let mut ts = crossings(a, b, boundary);
        ts.push(0.0);
        ts.push(1.0);
        ts.sort_by(f64::total_cmp);
        for w in ts.windows(2) {
            let (t0, t1) = (w[0], w[1]);
            if t1 <= t0 {
                continue;
            }
            if !inside(boundary, &at(a, b, (t0 + t1) / 2.0)) {
                continue;
            }
            let (p0, p1) = (at(a, b, t0), at(a, b, t1));
            match parts.last_mut() {
                Some(part) if part.last() == Some(&p0) => part.push(p1),
                _ => parts.push(vec![p0, p1]),
            }
        }
    }
    parts
}

/// the ends stay bit-exact, so a segment the boundary does not touch keeps
/// its source vertices
fn at(a: Coord, b: Coord, t: f64) -> Coord {
    match t {
        0.0 => a,
        1.0 => b,
        t => Coord::new(a.x + t * (b.x - a.x), a.y + t * (b.y - a.y)),
    }
}

/// parameters along `a`-`b` where it meets a boundary ring edge
fn crossings(a: Coord, b: Coord, boundary: &MultiPolygon) -> Vec<f64> {
    let mut ts = Vec::new();
    for poly in boundary.polygons() {
        for ring in std::iter::once(poly.exterior()).chain(poly.interiors()) {
            let coords = ring.coords();
            for i in 0..coords.len() {
                let (c, d) = (coords[i], coords[(i + 1) % coords.len()]);
                if let Some(t) = segment_param(a, b, c, d) {
                    ts.push(t);
                }
            }
        }
    }
    ts
}

fn segment_param(a: Coord, b: Coord, c: Coord, d: Coord) -> Option<f64> {
    let (abx, aby) = (b.x - a.x, b.y - a.y);
    let (cdx, cdy) = (d.x - c.x, d.y - c.y);
    let denom = abx * cdy - aby * cdx;
    if denom.abs() < 1e-12 {
        return None;
    }
    let t = ((c.x - a.x) * cdy - (c.y - a.y) * cdx) / denom;
    let u = ((c.x - a.x) * aby - (c.y - a.y) * abx) / denom;
    ((0.0..=1.0).contains(&t) && (0.0..=1.0).contains(&u)).then_some(t)
}
