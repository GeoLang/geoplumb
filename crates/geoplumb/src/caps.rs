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

/// negotiation-time pattern over point cloud link caps. resolution is the
/// thinning ladder bound, chunk tiling works exactly as for rasters
#[derive(Debug, Clone, PartialEq)]
pub struct PointPattern {
    pub crs: SetField<Crs>,
    pub resolution: ResRange,
    pub chunk_px: SetField<u32>,
}

impl Default for PointPattern {
    fn default() -> Self {
        PointPattern {
            crs: SetField::Any,
            resolution: ResRange::ANY,
            chunk_px: SetField::Any,
        }
    }
}

impl PointPattern {
    pub fn intersect(&self, other: &Self) -> Self {
        PointPattern {
            crs: self.crs.intersect(&other.crs),
            resolution: self.resolution.intersect(&other.resolution),
            chunk_px: self.chunk_px.intersect(&other.chunk_px),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.crs.is_empty() || self.resolution.is_empty() || self.chunk_px.is_empty()
    }
}

/// caps pattern, one alternative per kind. vector and tensor variants land
/// here later. cross-kind intersection is empty: a link has one kind
#[derive(Debug, Clone, PartialEq)]
pub enum CapsPattern {
    Raster(RasterPattern),
    PointCloud(PointPattern),
}

/// the fields every caps kind shares, the cross-kind projection surface of
/// a `Derived` constraint whose output kind differs from its input
struct CommonFields {
    crs: SetField<Crs>,
    resolution: ResRange,
    chunk_px: SetField<u32>,
}

impl CapsPattern {
    pub fn intersect(&self, other: &Self) -> Option<Self> {
        match (self, other) {
            (CapsPattern::Raster(a), CapsPattern::Raster(b)) => {
                let r = a.intersect(b);
                (!r.is_empty()).then_some(CapsPattern::Raster(r))
            }
            (CapsPattern::PointCloud(a), CapsPattern::PointCloud(b)) => {
                let p = a.intersect(b);
                (!p.is_empty()).then_some(CapsPattern::PointCloud(p))
            }
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            CapsPattern::Raster(r) => r.is_empty(),
            CapsPattern::PointCloud(p) => p.is_empty(),
        }
    }

    fn common(&self) -> CommonFields {
        match self {
            CapsPattern::Raster(r) => CommonFields {
                crs: r.crs.clone(),
                resolution: r.resolution,
                chunk_px: r.chunk_px.clone(),
            },
            CapsPattern::PointCloud(p) => CommonFields {
                crs: p.crs.clone(),
                resolution: p.resolution,
                chunk_px: p.chunk_px.clone(),
            },
        }
    }

    /// intersect the masked common fields from `from` onto this pattern,
    /// keeping kind-specific fields untouched
    fn project_common(&self, from: &CommonFields, mask: FieldMask) -> CapsPattern {
        let mut out = self.clone();
        match &mut out {
            CapsPattern::Raster(r) => {
                if mask.crs {
                    r.crs = r.crs.intersect(&from.crs);
                }
                if mask.resolution {
                    r.resolution = r.resolution.intersect(&from.resolution);
                }
                if mask.chunk_px {
                    r.chunk_px = r.chunk_px.intersect(&from.chunk_px);
                }
            }
            CapsPattern::PointCloud(p) => {
                if mask.crs {
                    p.crs = p.crs.intersect(&from.crs);
                }
                if mask.resolution {
                    p.resolution = p.resolution.intersect(&from.resolution);
                }
                if mask.chunk_px {
                    p.chunk_px = p.chunk_px.intersect(&from.chunk_px);
                }
            }
        }
        out
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

    /// unconstrained across every kind, the solver's link seed
    pub fn any() -> Self {
        CapsSet {
            alternatives: vec![
                CapsPattern::Raster(RasterPattern::default()),
                CapsPattern::PointCloud(PointPattern::default()),
            ],
        }
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
        match self.alternatives.first()? {
            CapsPattern::Raster(first) => Some(Caps::Raster(RasterCaps {
                dtype: first.dtype.fixate().unwrap_or(Dtype::F64),
                bands: first.bands.fixate().unwrap_or(1),
                crs: first.crs.fixate()?,
                resolution: first.resolution,
                chunk_px: first.chunk_px.fixate().unwrap_or(256),
            })),
            CapsPattern::PointCloud(first) => Some(Caps::PointCloud(PointCaps {
                crs: first.crs.fixate()?,
                resolution: first.resolution,
                chunk_px: first.chunk_px.fixate().unwrap_or(256),
            })),
        }
    }
}

/// fixed per-link caps handed to elements after the solve
#[derive(Debug, Clone, PartialEq)]
pub enum Caps {
    Raster(RasterCaps),
    PointCloud(PointCaps),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterCaps {
    pub dtype: Dtype,
    pub bands: u16,
    pub crs: Crs,
    pub resolution: ResRange,
    pub chunk_px: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PointCaps {
    pub crs: Crs,
    pub resolution: ResRange,
    pub chunk_px: u32,
}

impl Caps {
    /// panics on a non-raster link, negotiation guarantees an element's kind
    pub fn raster(&self) -> &RasterCaps {
        match self {
            Caps::Raster(r) => r,
            other => panic!("expected raster caps, got {other:?}"),
        }
    }

    /// panics on a non-point link, negotiation guarantees an element's kind
    pub fn point(&self) -> &PointCaps {
        match self {
            Caps::PointCloud(p) => p,
            other => panic!("expected point cloud caps, got {other:?}"),
        }
    }

    pub fn crs(&self) -> Crs {
        match self {
            Caps::Raster(r) => r.crs,
            Caps::PointCloud(p) => p.crs,
        }
    }

    pub fn chunk_px(&self) -> u32 {
        match self {
            Caps::Raster(r) => r.chunk_px,
            Caps::PointCloud(p) => p.chunk_px,
        }
    }

    /// these caps as a single-alternative pattern for downstream derivation
    pub fn pattern(&self) -> CapsPattern {
        match self {
            Caps::Raster(r) => CapsPattern::Raster(RasterPattern {
                dtype: SetField::one(r.dtype),
                bands: SetField::one(r.bands),
                crs: SetField::one(r.crs),
                resolution: r.resolution,
                chunk_px: SetField::one(r.chunk_px),
            }),
            Caps::PointCloud(p) => CapsPattern::PointCloud(PointPattern {
                crs: SetField::one(p.crs),
                resolution: p.resolution,
                chunk_px: SetField::one(p.chunk_px),
            }),
        }
    }

    /// discriminates cache entries when a node's caps change across re-solves
    pub fn fingerprint(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut h = std::hash::DefaultHasher::new();
        match self {
            Caps::Raster(r) => {
                0u8.hash(&mut h);
                r.dtype.hash(&mut h);
                r.bands.hash(&mut h);
                r.crs.hash(&mut h);
                r.chunk_px.hash(&mut h);
                r.resolution.min.to_bits().hash(&mut h);
                r.resolution.max.to_bits().hash(&mut h);
            }
            Caps::PointCloud(p) => {
                1u8.hash(&mut h);
                p.crs.hash(&mut h);
                p.chunk_px.hash(&mut h);
                p.resolution.min.to_bits().hash(&mut h);
                p.resolution.max.to_bits().hash(&mut h);
            }
        }
        h.finish()
    }
}

/// fields a derived transform passes through unchanged, used for backward
/// narrowing. dtype and bands only exist on raster links, so those bits are
/// ignored when either side of a projection is another kind
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

/// copy `from`'s masked fields onto `onto` by intersection, keep `onto`
/// elsewhere. across kinds only the common fields (crs, resolution, chunk
/// size) project, which is how a cross-kind `Derived` couples its links
pub fn mask_project(from: &CapsPattern, onto: &CapsPattern, mask: FieldMask) -> CapsPattern {
    match (from, onto) {
        (CapsPattern::Raster(f), CapsPattern::Raster(o)) => CapsPattern::Raster(RasterPattern {
            dtype: if mask.dtype {
                o.dtype.intersect(&f.dtype)
            } else {
                o.dtype.clone()
            },
            bands: if mask.bands {
                o.bands.intersect(&f.bands)
            } else {
                o.bands.clone()
            },
            crs: if mask.crs {
                o.crs.intersect(&f.crs)
            } else {
                o.crs.clone()
            },
            resolution: if mask.resolution {
                o.resolution.intersect(&f.resolution)
            } else {
                o.resolution
            },
            chunk_px: if mask.chunk_px {
                o.chunk_px.intersect(&f.chunk_px)
            } else {
                o.chunk_px.clone()
            },
        }),
        (from, onto) => onto.project_common(&from.common(), mask),
    }
}

/// element-declared constraint over its input and output link caps
pub enum Constraint {
    /// source shape, no input
    Produces(CapsSet),
    /// pass-through transform, output equals input, optionally narrowed
    Identity(CapsSet),
    /// output derived from input: passthrough fields copy across, the
    /// override pattern pins the retargeted fields (a reproject pins crs).
    /// the output pattern's kind may differ from the input's, that is a
    /// cross-kind transform (a gridder takes points and makes a raster)
    Derived {
        input: CapsSet,
        passthrough: FieldMask,
        output: CapsPattern,
    },
}

impl Constraint {
    /// the set this constraint accepts on its input link
    pub fn input_set(&self) -> CapsSet {
        match self {
            Constraint::Produces(_) => CapsSet::any(),
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
                    .map(|p| mask_project(p, output, *passthrough))
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
                            let candidate = mask_project(pin, inp, *passthrough);
                            (!candidate.is_empty()).then_some(candidate)
                        })
                    })
                    .collect();
                CapsSet { alternatives }
            }
        }
    }
}
