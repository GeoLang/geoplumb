// adapted from glass2glass g2g-core/src/caps.rs (MPL-2.0), see DESIGN.md
// negotiation fixes dtype, bands, crs and chunk size per link, resolution
// stays a range on the fixed caps and is checked per pull

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Crs(pub u32);

impl Crs {
    pub const WGS84: Crs = Crs(4326);
    pub const WEB_MERCATOR: Crs = Crs(3857);

    pub fn authority(&self) -> String {
        format!("EPSG:{}", self.0)
    }
}

impl fmt::Display for Crs {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "EPSG:{}", self.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dtype {
    F64,
}

/// preference-ordered set pattern for one caps field
#[derive(Debug, Clone, PartialEq)]
pub enum SetField<T> {
    Any,
    OneOf(Vec<T>),
}

impl<T: PartialEq + Clone> SetField<T> {
    pub fn one(v: T) -> Self {
        SetField::OneOf(vec![v])
    }

    pub fn intersect(&self, other: &Self) -> Self {
        match (self, other) {
            (SetField::Any, o) => o.clone(),
            (s, SetField::Any) => s.clone(),
            (SetField::OneOf(a), SetField::OneOf(b)) => {
                SetField::OneOf(a.iter().filter(|v| b.contains(v)).cloned().collect())
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, SetField::OneOf(v) if v.is_empty())
    }

    fn fixate(&self) -> Option<T> {
        match self {
            SetField::Any => None,
            SetField::OneOf(v) => v.first().cloned(),
        }
    }
}

/// inclusive resolution range in ground units per pixel, `ANY` = unconstrained
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ResRange {
    pub min: f64,
    pub max: f64,
}

impl ResRange {
    pub const ANY: ResRange = ResRange {
        min: 0.0,
        max: f64::INFINITY,
    };

    pub fn at_least(min: f64) -> Self {
        ResRange {
            min,
            max: f64::INFINITY,
        }
    }

    pub fn intersect(&self, other: &Self) -> Self {
        ResRange {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.min > self.max
    }

    pub fn contains(&self, r: f64) -> bool {
        r >= self.min && r <= self.max
    }
}

/// negotiation-time pattern over raster link caps
#[derive(Debug, Clone, PartialEq)]
pub struct RasterPattern {
    pub dtype: SetField<Dtype>,
    pub bands: SetField<u16>,
    pub crs: SetField<Crs>,
    pub resolution: ResRange,
    pub chunk_px: SetField<u32>,
}

impl Default for RasterPattern {
    fn default() -> Self {
        RasterPattern {
            dtype: SetField::Any,
            bands: SetField::Any,
            crs: SetField::Any,
            resolution: ResRange::ANY,
            chunk_px: SetField::Any,
        }
    }
}

impl RasterPattern {
    pub fn intersect(&self, other: &Self) -> Self {
        RasterPattern {
            dtype: self.dtype.intersect(&other.dtype),
            bands: self.bands.intersect(&other.bands),
            crs: self.crs.intersect(&other.crs),
            resolution: self.resolution.intersect(&other.resolution),
            chunk_px: self.chunk_px.intersect(&other.chunk_px),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.dtype.is_empty()
            || self.bands.is_empty()
            || self.crs.is_empty()
            || self.resolution.is_empty()
            || self.chunk_px.is_empty()
    }
}

/// caps pattern, one alternative. vector and point cloud variants land here later
#[derive(Debug, Clone, PartialEq)]
pub enum CapsPattern {
    Raster(RasterPattern),
}

impl CapsPattern {
    pub fn raster(&self) -> &RasterPattern {
        match self {
            CapsPattern::Raster(r) => r,
        }
    }

    pub fn intersect(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (CapsPattern::Raster(a), CapsPattern::Raster(b)) => {
                let r = a.intersect(b);
                (!r.is_empty()).then_some(CapsPattern::Raster(r))
            }
        }
    }
}

/// preference-ordered alternatives, the negotiation vocabulary
#[derive(Debug, Clone, PartialEq)]
pub struct CapsSet {
    pub alternatives: Vec<CapsPattern>,
}

impl CapsSet {
    pub fn one(p: CapsPattern) -> Self {
        CapsSet {
            alternatives: vec![p],
        }
    }

    pub fn any_raster() -> Self {
        CapsSet::one(CapsPattern::Raster(RasterPattern::default()))
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let mut alternatives = Vec::new();
        for a in &self.alternatives {
            for b in &other.alternatives {
                if let Some(r) = a.intersect(b) {
                    if !alternatives.contains(&r) {
                        alternatives.push(r);
                    }
                }
            }
        }
        CapsSet { alternatives }
    }

    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }

    /// fixate the first alternative to concrete link caps, resolution stays
    /// ranged. dtype defaults to f64, bands and chunk size left `Any` by every
    /// constraint on the link fall back to 1 band and 256 px
    pub fn fixate(&self) -> Option<Caps> {
        let first = self.alternatives.first()?.raster();
        Some(Caps::Raster(RasterCaps {
            dtype: first.dtype.fixate().unwrap_or(Dtype::F64),
            bands: first.bands.fixate().unwrap_or(1),
            crs: first.crs.fixate()?,
            resolution: first.resolution,
            chunk_px: first.chunk_px.fixate().unwrap_or(256),
        }))
    }
}

/// fixed per-link caps handed to elements after the solve
#[derive(Debug, Clone, PartialEq)]
pub enum Caps {
    Raster(RasterCaps),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterCaps {
    pub dtype: Dtype,
    pub bands: u16,
    pub crs: Crs,
    pub resolution: ResRange,
    pub chunk_px: u32,
}

impl Caps {
    pub fn raster(&self) -> &RasterCaps {
        match self {
            Caps::Raster(r) => r,
        }
    }

    /// discriminates cache entries when a node's caps change across re-solves
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let r = self.raster();
        let mut h = std::hash::DefaultHasher::new();
        r.dtype.hash(&mut h);
        r.bands.hash(&mut h);
        r.crs.hash(&mut h);
        r.chunk_px.hash(&mut h);
        r.resolution.min.to_bits().hash(&mut h);
        r.resolution.max.to_bits().hash(&mut h);
        h.finish()
    }
}

/// fields a derived transform passes through unchanged, used for backward narrowing
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FieldMask {
    pub dtype: bool,
    pub bands: bool,
    pub crs: bool,
    pub resolution: bool,
    pub chunk_px: bool,
}

impl FieldMask {
    pub const ALL: FieldMask = FieldMask {
        dtype: true,
        bands: true,
        crs: true,
        resolution: true,
        chunk_px: true,
    };

    pub fn without_crs_resolution() -> Self {
        FieldMask {
            crs: false,
            resolution: false,
            ..FieldMask::ALL
        }
    }
}

/// copy `from`'s masked fields onto `onto` by intersection, keep `onto` elsewhere
pub fn mask_project(from: &RasterPattern, onto: &RasterPattern, mask: FieldMask) -> RasterPattern {
    RasterPattern {
        dtype: if mask.dtype {
            onto.dtype.intersect(&from.dtype)
        } else {
            onto.dtype.clone()
        },
        bands: if mask.bands {
            onto.bands.intersect(&from.bands)
        } else {
            onto.bands.clone()
        },
        crs: if mask.crs {
            onto.crs.intersect(&from.crs)
        } else {
            onto.crs.clone()
        },
        resolution: if mask.resolution {
            onto.resolution.intersect(&from.resolution)
        } else {
            onto.resolution
        },
        chunk_px: if mask.chunk_px {
            onto.chunk_px.intersect(&from.chunk_px)
        } else {
            onto.chunk_px.clone()
        },
    }
}

/// element-declared constraint over its input and output link caps
pub enum Constraint {
    /// source shape, no input
    Produces(CapsSet),
    /// pass-through transform, output equals input, optionally narrowed
    Identity(CapsSet),
    /// output derived from input: passthrough fields copy across, the
    /// override pattern pins the retargeted fields (a reproject pins crs)
    Derived {
        input: CapsSet,
        passthrough: FieldMask,
        output: RasterPattern,
    },
}

impl Constraint {
    /// the set this constraint accepts on its input link
    pub fn input_set(&self) -> CapsSet {
        match self {
            Constraint::Produces(_) => CapsSet::any_raster(),
            Constraint::Identity(set) => set.clone(),
            Constraint::Derived { input, .. } => input.clone(),
        }
    }

    /// the set this constraint can put on its output link given the input set
    pub fn output_set(&self, input_link: &CapsSet) -> CapsSet {
        match self {
            Constraint::Produces(set) => set.clone(),
            Constraint::Identity(set) => input_link.intersect(set),
            Constraint::Derived {
                input,
                passthrough,
                output,
            } => {
                let narrowed = input_link.intersect(input);
                let alternatives = narrowed
                    .alternatives
                    .iter()
                    .map(|p| CapsPattern::Raster(mask_project(p.raster(), output, *passthrough)))
                    .collect();
                CapsSet { alternatives }
            }
        }
    }

    /// narrow the input link from a pin on the output link, the backward sweep.
    /// only passthrough fields couple backward through a derived transform
    pub fn narrow_input(&self, input_link: &CapsSet, output_pin: &CapsSet) -> CapsSet {
        match self {
            Constraint::Produces(_) => input_link.clone(),
            Constraint::Identity(_) => input_link.intersect(output_pin),
            Constraint::Derived { passthrough, .. } => {
                let alternatives = input_link
                    .alternatives
                    .iter()
                    .filter_map(|inp| {
                        output_pin.alternatives.iter().find_map(|pin| {
                            let candidate = mask_project(pin.raster(), inp.raster(), *passthrough);
                            (!candidate.is_empty()).then_some(CapsPattern::Raster(candidate))
                        })
                    })
                    .collect();
                CapsSet { alternatives }
            }
        }
    }
}
