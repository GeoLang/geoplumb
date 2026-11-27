// adapted from glass2glass g2g-core/src/caps.rs (MPL-2.0), see DESIGN.md
// negotiation fixes dtype, bands, crs and chunk size per link, resolution
// stays a range on the fixed caps and is checked per pull

use std::fmt;

use terrano_core::{SampleFormat, TypedBand};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum Dtype {
    U8 { nodata: Option<u8> },
    I8 { nodata: Option<i8> },
    U16 { nodata: Option<u16> },
    I16 { nodata: Option<i16> },
    U32 { nodata: Option<u32> },
    I32 { nodata: Option<i32> },
    F32,
    F64,
}

impl Dtype {
    pub fn of(band: &TypedBand) -> Dtype {
        match band {
            TypedBand::U8 { nodata, .. } => Dtype::U8 { nodata: *nodata },
            TypedBand::I8 { nodata, .. } => Dtype::I8 { nodata: *nodata },
            TypedBand::U16 { nodata, .. } => Dtype::U16 { nodata: *nodata },
            TypedBand::I16 { nodata, .. } => Dtype::I16 { nodata: *nodata },
            TypedBand::U32 { nodata, .. } => Dtype::U32 { nodata: *nodata },
            TypedBand::I32 { nodata, .. } => Dtype::I32 { nodata: *nodata },
            TypedBand::F32(_) => Dtype::F32,
            TypedBand::F64(_) => Dtype::F64,
        }
    }

    pub fn from_format(format: SampleFormat, nodata: Option<f64>) -> crate::error::Result<Dtype> {
        let sentinel = match (format.integer_range(), nodata) {
            (Some((low, high)), Some(value))
                if value.fract() == 0.0 && (low..=high).contains(&value) =>
            {
                Some(value)
            }
            (Some(_), Some(value)) => {
                return Err(terrano_core::Error::InvalidInput(format!(
                    "nodata {value} is not a {} value",
                    format.name()
                ))
                .into());
            }
            (Some(_), None) | (None, _) => None,
        };
        // float casts of a whole value inside the type's range are exact
        Ok(match format {
            SampleFormat::U8 => Dtype::U8 {
                nodata: sentinel.map(|value| value as u8),
            },
            SampleFormat::I8 => Dtype::I8 {
                nodata: sentinel.map(|value| value as i8),
            },
            SampleFormat::U16 => Dtype::U16 {
                nodata: sentinel.map(|value| value as u16),
            },
            SampleFormat::I16 => Dtype::I16 {
                nodata: sentinel.map(|value| value as i16),
            },
            SampleFormat::U32 => Dtype::U32 {
                nodata: sentinel.map(|value| value as u32),
            },
            SampleFormat::I32 => Dtype::I32 {
                nodata: sentinel.map(|value| value as i32),
            },
            SampleFormat::F32 => Dtype::F32,
            SampleFormat::F64 => Dtype::F64,
        })
    }

    pub fn format(&self) -> SampleFormat {
        match self {
            Dtype::U8 { .. } => SampleFormat::U8,
            Dtype::I8 { .. } => SampleFormat::I8,
            Dtype::U16 { .. } => SampleFormat::U16,
            Dtype::I16 { .. } => SampleFormat::I16,
            Dtype::U32 { .. } => SampleFormat::U32,
            Dtype::I32 { .. } => SampleFormat::I32,
            Dtype::F32 => SampleFormat::F32,
            Dtype::F64 => SampleFormat::F64,
        }
    }

    pub fn integer_nodata(&self) -> Option<f64> {
        match *self {
            Dtype::U8 { nodata } => nodata.map(f64::from),
            Dtype::I8 { nodata } => nodata.map(f64::from),
            Dtype::U16 { nodata } => nodata.map(f64::from),
            Dtype::I16 { nodata } => nodata.map(f64::from),
            Dtype::U32 { nodata } => nodata.map(f64::from),
            Dtype::I32 { nodata } => nodata.map(f64::from),
            Dtype::F32 | Dtype::F64 => None,
        }
    }

    pub fn marks_absent_cells(&self) -> bool {
        self.format().integer_range().is_none() || self.integer_nodata().is_some()
    }

    pub fn widens_losslessly_to(&self, wider: Dtype) -> bool {
        use SampleFormat::*;
        let formats = (self.format(), wider.format());
        let integer_pair = matches!(
            formats,
            (U8, U16 | I16 | U32 | I32) | (I8, I16 | I32) | (U16, U32 | I32) | (I16, I32)
        );
        let float_target = matches!(
            formats,
            (U8 | I8 | U16 | I16, F32) | (U8 | I8 | U16 | I16 | U32 | I32 | F32, F64)
        );
        float_target || (integer_pair && self.integer_nodata() == wider.integer_nodata())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum ValueKind {
    Continuous,
    Categorical,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Hash)]
pub struct PerBand<T>(Vec<T>);

pub type BandKinds = PerBand<ValueKind>;
pub type BandDtypes = PerBand<Dtype>;

impl<T> From<T> for PerBand<T> {
    fn from(every_band: T) -> PerBand<T> {
        PerBand(vec![every_band])
    }
}

impl<T: Copy + PartialEq> PerBand<T> {
    pub fn from_bands(values: Vec<T>) -> PerBand<T> {
        let first = *values.first().expect("a raster has at least one band");
        if values.iter().all(|value| *value == first) {
            return PerBand::from(first);
        }
        PerBand(values)
    }

    pub fn band(&self, band: usize) -> T {
        match self.0.as_slice() {
            [every_band] => *every_band,
            values => values[band],
        }
    }

    pub fn fits(&self, bands: usize) -> bool {
        self.0.len() == 1 || self.0.len() == bands
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum TensorDtype {
    F32,
}

// PartialOrd only because SetField needs it, the order means nothing
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum Domain {
    System,
    Wgpu,
}

/// preference-ordered set pattern for one caps field. `AtLeast` is an open
/// lower bound, for demands like "at least n bands" that have no ceiling to
/// enumerate. construct it only where the order means something
#[derive(Debug, Clone, PartialEq)]
pub enum SetField<T> {
    Any,
    OneOf(Vec<T>),
    AtLeast(T),
}

impl<T: PartialOrd + Clone> SetField<T> {
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
            (SetField::AtLeast(a), SetField::AtLeast(b)) => {
                SetField::AtLeast(if a >= b { a.clone() } else { b.clone() })
            }
            (SetField::AtLeast(n), SetField::OneOf(vs))
            | (SetField::OneOf(vs), SetField::AtLeast(n)) => {
                SetField::OneOf(vs.iter().filter(|v| *v >= n).cloned().collect())
            }
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, SetField::OneOf(v) if v.is_empty())
    }

    /// a bound nothing narrowed to concrete offers fixates to the bound itself
    fn fixate(&self) -> Option<T> {
        match self {
            SetField::Any => None,
            SetField::OneOf(v) => v.first().cloned(),
            SetField::AtLeast(n) => Some(n.clone()),
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

#[derive(Debug, Clone, PartialEq)]
pub struct RasterPattern {
    pub dtypes: SetField<BandDtypes>,
    pub bands: SetField<u16>,
    pub crs: SetField<Crs>,
    pub resolution: ResRange,
    pub chunk_px: SetField<u32>,
    pub value_kinds: SetField<BandKinds>,
}

impl Default for RasterPattern {
    fn default() -> Self {
        RasterPattern {
            dtypes: SetField::Any,
            bands: SetField::Any,
            crs: SetField::Any,
            resolution: ResRange::ANY,
            chunk_px: SetField::Any,
            value_kinds: SetField::Any,
        }
    }
}

impl RasterPattern {
    pub fn intersect(&self, other: &Self) -> Self {
        RasterPattern {
            dtypes: self.dtypes.intersect(&other.dtypes),
            bands: self.bands.intersect(&other.bands),
            crs: self.crs.intersect(&other.crs),
            resolution: self.resolution.intersect(&other.resolution),
            chunk_px: self.chunk_px.intersect(&other.chunk_px),
            value_kinds: self.value_kinds.intersect(&other.value_kinds),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.dtypes.is_empty()
            || self.bands.is_empty()
            || self.crs.is_empty()
            || self.resolution.is_empty()
            || self.chunk_px.is_empty()
            || self.value_kinds.is_empty()
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

/// negotiation-time pattern over vector link caps. resolution is the
/// simplification ladder bound, chunk tiling works exactly as for rasters
#[derive(Debug, Clone, PartialEq)]
pub struct VectorPattern {
    pub crs: SetField<Crs>,
    pub resolution: ResRange,
    pub chunk_px: SetField<u32>,
}

impl Default for VectorPattern {
    fn default() -> Self {
        VectorPattern {
            crs: SetField::Any,
            resolution: ResRange::ANY,
            chunk_px: SetField::Any,
        }
    }
}

impl VectorPattern {
    pub fn intersect(&self, other: &Self) -> Self {
        VectorPattern {
            crs: self.crs.intersect(&other.crs),
            resolution: self.resolution.intersect(&other.resolution),
            chunk_px: self.chunk_px.intersect(&other.chunk_px),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.crs.is_empty() || self.resolution.is_empty() || self.chunk_px.is_empty()
    }
}

/// negotiation-time pattern over tensor link caps. channels is the CHW
/// channel count, chunk tiling works exactly as for rasters and carries a
/// model's input size when one is on the link
#[derive(Debug, Clone, PartialEq)]
pub struct TensorPattern {
    pub dtype: SetField<TensorDtype>,
    pub channels: SetField<u16>,
    pub crs: SetField<Crs>,
    pub resolution: ResRange,
    pub chunk_px: SetField<u32>,
    pub domain: SetField<Domain>,
}

impl Default for TensorPattern {
    fn default() -> Self {
        TensorPattern {
            dtype: SetField::Any,
            channels: SetField::Any,
            crs: SetField::Any,
            resolution: ResRange::ANY,
            chunk_px: SetField::Any,
            domain: SetField::one(Domain::System),
        }
    }
}

impl TensorPattern {
    pub fn intersect(&self, other: &Self) -> Self {
        TensorPattern {
            dtype: self.dtype.intersect(&other.dtype),
            channels: self.channels.intersect(&other.channels),
            crs: self.crs.intersect(&other.crs),
            resolution: self.resolution.intersect(&other.resolution),
            chunk_px: self.chunk_px.intersect(&other.chunk_px),
            domain: self.domain.intersect(&other.domain),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.dtype.is_empty()
            || self.channels.is_empty()
            || self.crs.is_empty()
            || self.resolution.is_empty()
            || self.chunk_px.is_empty()
            || self.domain.is_empty()
    }
}

/// caps pattern, one alternative per kind. cross-kind intersection is
/// empty: a link has one kind
#[derive(Debug, Clone, PartialEq)]
pub enum CapsPattern {
    Raster(RasterPattern),
    PointCloud(PointPattern),
    Vector(VectorPattern),
    Tensor(TensorPattern),
}

/// the fields every caps kind shares, the cross-kind projection surface of
/// a `Derived` constraint whose output kind differs from its input. planes
/// is the plane count where the kind has one (raster bands, tensor
/// channels), `Any` elsewhere, so a band demand can narrow a channel count
/// and vice versa
struct CommonFields {
    crs: SetField<Crs>,
    resolution: ResRange,
    chunk_px: SetField<u32>,
    planes: SetField<u16>,
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
            (CapsPattern::Vector(a), CapsPattern::Vector(b)) => {
                let v = a.intersect(b);
                (!v.is_empty()).then_some(CapsPattern::Vector(v))
            }
            (CapsPattern::Tensor(a), CapsPattern::Tensor(b)) => {
                let t = a.intersect(b);
                (!t.is_empty()).then_some(CapsPattern::Tensor(t))
            }
            _ => None,
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            CapsPattern::Raster(r) => r.is_empty(),
            CapsPattern::PointCloud(p) => p.is_empty(),
            CapsPattern::Vector(v) => v.is_empty(),
            CapsPattern::Tensor(t) => t.is_empty(),
        }
    }

    fn empty_fields(&self) -> Vec<&'static str> {
        let fields = match self {
            CapsPattern::Raster(r) => vec![
                ("dtypes", r.dtypes.is_empty()),
                ("bands", r.bands.is_empty()),
                ("crs", r.crs.is_empty()),
                ("resolution", r.resolution.is_empty()),
                ("chunk_px", r.chunk_px.is_empty()),
                ("value_kinds", r.value_kinds.is_empty()),
            ],
            CapsPattern::PointCloud(p) => vec![
                ("crs", p.crs.is_empty()),
                ("resolution", p.resolution.is_empty()),
                ("chunk_px", p.chunk_px.is_empty()),
            ],
            CapsPattern::Vector(v) => vec![
                ("crs", v.crs.is_empty()),
                ("resolution", v.resolution.is_empty()),
                ("chunk_px", v.chunk_px.is_empty()),
            ],
            CapsPattern::Tensor(t) => vec![
                ("dtype", t.dtype.is_empty()),
                ("channels", t.channels.is_empty()),
                ("crs", t.crs.is_empty()),
                ("resolution", t.resolution.is_empty()),
                ("chunk_px", t.chunk_px.is_empty()),
                ("domain", t.domain.is_empty()),
            ],
        };
        fields
            .into_iter()
            .filter(|(_, empty)| *empty)
            .map(|(name, _)| name)
            .collect()
    }

    fn common(&self) -> CommonFields {
        match self {
            CapsPattern::Raster(r) => CommonFields {
                crs: r.crs.clone(),
                resolution: r.resolution,
                chunk_px: r.chunk_px.clone(),
                planes: r.bands.clone(),
            },
            CapsPattern::PointCloud(p) => CommonFields {
                crs: p.crs.clone(),
                resolution: p.resolution,
                chunk_px: p.chunk_px.clone(),
                planes: SetField::Any,
            },
            CapsPattern::Vector(v) => CommonFields {
                crs: v.crs.clone(),
                resolution: v.resolution,
                chunk_px: v.chunk_px.clone(),
                planes: SetField::Any,
            },
            CapsPattern::Tensor(t) => CommonFields {
                crs: t.crs.clone(),
                resolution: t.resolution,
                chunk_px: t.chunk_px.clone(),
                planes: t.channels.clone(),
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
                if mask.bands {
                    r.bands = r.bands.intersect(&from.planes);
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
            CapsPattern::Vector(v) => {
                if mask.crs {
                    v.crs = v.crs.intersect(&from.crs);
                }
                if mask.resolution {
                    v.resolution = v.resolution.intersect(&from.resolution);
                }
                if mask.chunk_px {
                    v.chunk_px = v.chunk_px.intersect(&from.chunk_px);
                }
            }
            CapsPattern::Tensor(t) => {
                if mask.crs {
                    t.crs = t.crs.intersect(&from.crs);
                }
                if mask.resolution {
                    t.resolution = t.resolution.intersect(&from.resolution);
                }
                if mask.chunk_px {
                    t.chunk_px = t.chunk_px.intersect(&from.chunk_px);
                }
                if mask.bands {
                    t.channels = t.channels.intersect(&from.planes);
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

    pub fn any_vector() -> Self {
        CapsSet::one(CapsPattern::Vector(VectorPattern::default()))
    }

    /// unconstrained across every kind, the solver's link seed
    pub fn any() -> Self {
        CapsSet {
            alternatives: vec![
                CapsPattern::Raster(RasterPattern::default()),
                CapsPattern::PointCloud(PointPattern::default()),
                CapsPattern::Vector(VectorPattern::default()),
                CapsPattern::Tensor(TensorPattern {
                    domain: SetField::Any,
                    ..TensorPattern::default()
                }),
            ],
        }
    }

    pub fn intersect(&self, other: &Self) -> Self {
        let mut alternatives = Vec::new();
        for a in &self.alternatives {
            for b in &other.alternatives {
                if let Some(r) = a.intersect(b)
                    && !alternatives.contains(&r)
                {
                    alternatives.push(r);
                }
            }
        }
        CapsSet { alternatives }
    }

    pub fn is_empty(&self) -> bool {
        self.alternatives.is_empty()
    }

    pub fn fixate(&self) -> Option<Caps> {
        match self.alternatives.first()? {
            CapsPattern::Raster(first) => Some(Caps::Raster(RasterCaps {
                dtypes: first
                    .dtypes
                    .fixate()
                    .unwrap_or_else(|| BandDtypes::from(Dtype::F64)),
                bands: first.bands.fixate().unwrap_or(1),
                crs: first.crs.fixate()?,
                resolution: first.resolution,
                chunk_px: first.chunk_px.fixate().unwrap_or(256),
                value_kinds: first
                    .value_kinds
                    .fixate()
                    .unwrap_or_else(|| BandKinds::from(ValueKind::Continuous)),
            })),
            CapsPattern::PointCloud(first) => Some(Caps::PointCloud(PointCaps {
                crs: first.crs.fixate()?,
                resolution: first.resolution,
                chunk_px: first.chunk_px.fixate().unwrap_or(256),
            })),
            CapsPattern::Vector(first) => Some(Caps::Vector(VectorCaps {
                crs: first.crs.fixate()?,
                resolution: first.resolution,
                chunk_px: first.chunk_px.fixate().unwrap_or(256),
            })),
            CapsPattern::Tensor(first) => Some(Caps::Tensor(TensorCaps {
                dtype: first.dtype.fixate().unwrap_or(TensorDtype::F32),
                channels: first.channels.fixate().unwrap_or(1),
                crs: first.crs.fixate()?,
                resolution: first.resolution,
                chunk_px: first.chunk_px.fixate().unwrap_or(256),
                domain: first.domain.fixate().unwrap_or(Domain::System),
            })),
        }
    }
}

/// fixed per-link caps handed to elements after the solve
#[derive(Debug, Clone, PartialEq)]
pub enum Caps {
    Raster(RasterCaps),
    PointCloud(PointCaps),
    Vector(VectorCaps),
    Tensor(TensorCaps),
}

#[derive(Debug, Clone, PartialEq)]
pub struct RasterCaps {
    pub dtypes: BandDtypes,
    pub bands: u16,
    pub crs: Crs,
    pub resolution: ResRange,
    pub chunk_px: u32,
    pub value_kinds: BandKinds,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PointCaps {
    pub crs: Crs,
    pub resolution: ResRange,
    pub chunk_px: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VectorCaps {
    pub crs: Crs,
    pub resolution: ResRange,
    pub chunk_px: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TensorCaps {
    pub dtype: TensorDtype,
    pub channels: u16,
    pub crs: Crs,
    pub resolution: ResRange,
    pub chunk_px: u32,
    pub domain: Domain,
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

    /// panics on a non-vector link, negotiation guarantees an element's kind
    pub fn vector(&self) -> &VectorCaps {
        match self {
            Caps::Vector(v) => v,
            other => panic!("expected vector caps, got {other:?}"),
        }
    }

    /// panics on a non-tensor link, negotiation guarantees an element's kind
    pub fn tensor(&self) -> &TensorCaps {
        match self {
            Caps::Tensor(t) => t,
            other => panic!("expected tensor caps, got {other:?}"),
        }
    }

    pub fn crs(&self) -> Crs {
        match self {
            Caps::Raster(r) => r.crs,
            Caps::PointCloud(p) => p.crs,
            Caps::Vector(v) => v.crs,
            Caps::Tensor(t) => t.crs,
        }
    }

    pub fn chunk_px(&self) -> u32 {
        match self {
            Caps::Raster(r) => r.chunk_px,
            Caps::PointCloud(p) => p.chunk_px,
            Caps::Vector(v) => v.chunk_px,
            Caps::Tensor(t) => t.chunk_px,
        }
    }

    /// these caps as a single-alternative pattern for downstream derivation
    pub fn pattern(&self) -> CapsPattern {
        match self {
            Caps::Raster(r) => CapsPattern::Raster(RasterPattern {
                dtypes: SetField::one(r.dtypes.clone()),
                bands: SetField::one(r.bands),
                crs: SetField::one(r.crs),
                resolution: r.resolution,
                chunk_px: SetField::one(r.chunk_px),
                value_kinds: SetField::one(r.value_kinds.clone()),
            }),
            Caps::PointCloud(p) => CapsPattern::PointCloud(PointPattern {
                crs: SetField::one(p.crs),
                resolution: p.resolution,
                chunk_px: SetField::one(p.chunk_px),
            }),
            Caps::Vector(v) => CapsPattern::Vector(VectorPattern {
                crs: SetField::one(v.crs),
                resolution: v.resolution,
                chunk_px: SetField::one(v.chunk_px),
            }),
            Caps::Tensor(t) => CapsPattern::Tensor(TensorPattern {
                dtype: SetField::one(t.dtype),
                channels: SetField::one(t.channels),
                crs: SetField::one(t.crs),
                resolution: t.resolution,
                chunk_px: SetField::one(t.chunk_px),
                domain: SetField::one(t.domain),
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
                r.dtypes.hash(&mut h);
                r.bands.hash(&mut h);
                r.crs.hash(&mut h);
                r.chunk_px.hash(&mut h);
                r.resolution.min.to_bits().hash(&mut h);
                r.resolution.max.to_bits().hash(&mut h);
                r.value_kinds.hash(&mut h);
            }
            Caps::PointCloud(p) => {
                1u8.hash(&mut h);
                p.crs.hash(&mut h);
                p.chunk_px.hash(&mut h);
                p.resolution.min.to_bits().hash(&mut h);
                p.resolution.max.to_bits().hash(&mut h);
            }
            Caps::Vector(v) => {
                2u8.hash(&mut h);
                v.crs.hash(&mut h);
                v.chunk_px.hash(&mut h);
                v.resolution.min.to_bits().hash(&mut h);
                v.resolution.max.to_bits().hash(&mut h);
            }
            Caps::Tensor(t) => {
                3u8.hash(&mut h);
                t.dtype.hash(&mut h);
                t.channels.hash(&mut h);
                t.crs.hash(&mut h);
                t.chunk_px.hash(&mut h);
                t.resolution.min.to_bits().hash(&mut h);
                t.resolution.max.to_bits().hash(&mut h);
                t.domain.hash(&mut h);
            }
        }
        h.finish()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FieldMask {
    pub dtypes: bool,
    pub bands: bool,
    pub crs: bool,
    pub resolution: bool,
    pub chunk_px: bool,
    pub value_kinds: bool,
}

impl FieldMask {
    pub const ALL: FieldMask = FieldMask {
        dtypes: true,
        bands: true,
        crs: true,
        resolution: true,
        chunk_px: true,
        value_kinds: true,
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
            dtypes: if mask.dtypes {
                o.dtypes.intersect(&f.dtypes)
            } else {
                o.dtypes.clone()
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
            value_kinds: if mask.value_kinds {
                o.value_kinds.intersect(&f.value_kinds)
            } else {
                o.value_kinds.clone()
            },
        }),
        (from, onto) => onto.project_common(&from.common(), mask),
    }
}

pub fn meet(onto: &CapsSet, from: &CapsSet, coupling: Option<FieldMask>) -> CapsSet {
    let Some(mask) = coupling else {
        return onto.intersect(from);
    };
    let mut alternatives = Vec::new();
    for o in &onto.alternatives {
        for f in &from.alternatives {
            let met = mask_project(f, o, mask);
            if !met.is_empty() && !alternatives.contains(&met) {
                alternatives.push(met);
            }
        }
    }
    CapsSet { alternatives }
}

pub fn conflicting_fields(
    onto: &CapsSet,
    from: &CapsSet,
    coupling: Option<FieldMask>,
) -> Vec<&'static str> {
    let mask = coupling.unwrap_or(FieldMask::ALL);
    onto.alternatives
        .iter()
        .flat_map(move |o| {
            from.alternatives.iter().map(move |f| {
                if coupling.is_none() && std::mem::discriminant(o) != std::mem::discriminant(f) {
                    return vec!["kind"];
                }
                mask_project(f, o, mask).empty_fields()
            })
        })
        .min_by_key(Vec::len)
        .unwrap_or_default()
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
    AnyOf(Vec<Constraint>),
}

fn union(sets: impl IntoIterator<Item = CapsSet>) -> CapsSet {
    let mut alternatives: Vec<CapsPattern> = Vec::new();
    for pattern in sets.into_iter().flat_map(|set| set.alternatives) {
        if !alternatives.contains(&pattern) {
            alternatives.push(pattern);
        }
    }
    CapsSet { alternatives }
}

impl Constraint {
    pub fn pinned_value_kind_and_dtype(
        input: CapsSet,
        value_kind: ValueKind,
        dtype: Dtype,
    ) -> Constraint {
        Constraint::Derived {
            input,
            passthrough: FieldMask {
                value_kinds: false,
                dtypes: false,
                ..FieldMask::ALL
            },
            output: CapsPattern::Raster(RasterPattern {
                value_kinds: SetField::one(BandKinds::from(value_kind)),
                dtypes: SetField::one(BandDtypes::from(dtype)),
                ..RasterPattern::default()
            }),
        }
    }

    pub fn passes_dtypes_through(&self) -> bool {
        match self {
            Constraint::Identity(_) => true,
            Constraint::Derived { passthrough, .. } => passthrough.dtypes,
            Constraint::AnyOf(members) => members.iter().all(Constraint::passes_dtypes_through),
            Constraint::Produces(_) => false,
        }
    }

    pub fn coupled_fields(&self) -> Option<FieldMask> {
        match self {
            Constraint::Derived { passthrough, .. } => Some(*passthrough),
            Constraint::Produces(_) | Constraint::Identity(_) => None,
            Constraint::AnyOf(members) => members.iter().map(Constraint::coupled_fields).try_fold(
                FieldMask::default(),
                |coupled, member| {
                    let member = member?;
                    Some(FieldMask {
                        dtypes: coupled.dtypes || member.dtypes,
                        bands: coupled.bands || member.bands,
                        crs: coupled.crs || member.crs,
                        resolution: coupled.resolution || member.resolution,
                        chunk_px: coupled.chunk_px || member.chunk_px,
                        value_kinds: coupled.value_kinds || member.value_kinds,
                    })
                },
            ),
        }
    }

    /// the set this constraint accepts on its input link
    pub fn input_set(&self) -> CapsSet {
        match self {
            Constraint::Produces(_) => CapsSet::any(),
            Constraint::Identity(set) => set.clone(),
            Constraint::Derived { input, .. } => input.clone(),
            Constraint::AnyOf(members) => union(members.iter().map(Constraint::input_set)),
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
            Constraint::AnyOf(members) => union(
                members
                    .iter()
                    .map(|member| member.output_set(&input_link.intersect(&member.input_set()))),
            ),
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
            Constraint::AnyOf(members) => union(members.iter().map(|member| {
                let accepted = member.input_set();
                let producible = member.output_set(&accepted);
                member.narrow_input(
                    &input_link.intersect(&accepted),
                    &output_pin.intersect(&producible),
                )
            })),
        }
    }
}
