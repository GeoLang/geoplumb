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
use topoi_core::{MultiPolygon, clip_to_boundary};

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

/// intersect every fragment with a static boundary, the cut itself topoi's
/// `clip_to_boundary`. the result is a subset of the fragment, so tile
/// splitting commutes with the clip and no halo is needed. fragments
/// reduced to nothing are dropped
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
