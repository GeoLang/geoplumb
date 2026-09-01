//! the layer file: one entry per served layer, either a source plus an
//! ordered op pipeline or several inputs joined by a fanin. every layer
//! ends reprojected to web mercator and encoded as grayscale png, so the
//! file says what varies and nothing else

use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use geoplumb::elements::{Burn, Composite, FocalOp, StacSearch};
use serde::Deserialize;
use serde_json::Value;
use terrano_core::{BinaryOp, UnaryOp};
use topoi_core::MultiPolygon;
use topoi_core::geojson::FeatureGeometry;

/// the gray range png encoding stretches over when neither an op nor the
/// layer names one, hillshade's own range
const DEFAULT_GRAY: (f64, f64) = (0.0, 255.0);
/// slope is degrees from horizontal
const SLOPE_GRAY: (f64, f64) = (0.0, 90.0);
/// aspect is compass degrees
const ASPECT_GRAY: (f64, f64) = (0.0, 360.0);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    #[serde(default, rename = "layer")]
    pub layers: Vec<LayerConfig>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LayerConfig {
    /// url path segment the tiles are served under
    pub name: String,
    /// the one source a linear layer reads, against `input` plus `fanin`
    /// for a layer joining several
    pub source: Option<SourceConfig>,
    #[serde(default, rename = "input")]
    pub inputs: Vec<InputConfig>,
    pub fanin: Option<FaninConfig>,
    /// the chain over the source, or the chain after the fanin where the
    /// layer has one
    #[serde(default, rename = "op")]
    pub ops: Vec<OpConfig>,
    /// the value range the png encoding stretches over, overriding the one
    /// the ops imply, `gray = { min = 0.0, max = 3.0 }`
    pub gray: Option<GrayConfig>,
}

/// one input of a fan-in layer: a source and the chain over it, the pair
/// a linear layer names at the top level
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InputConfig {
    pub source: SourceConfig,
    #[serde(default, rename = "op")]
    pub ops: Vec<OpConfig>,
}

/// what a layer reads: one source, or several inputs and the element
/// joining them
pub enum LayerShape<'a> {
    Single(&'a SourceConfig),
    Fanin(&'a FaninConfig, &'a [InputConfig]),
}

/// how a fan-in layer joins its inputs. `mosaic` takes the first input
/// with a value at a pixel, `combine` runs a per-cell binary op across
/// exactly two
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum FaninConfig {
    Mosaic,
    Combine { op: BinaryOpConfig },
}

/// the per-cell operation a `combine` runs over its two inputs, none of
/// them taking a constant
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BinaryOpConfig {
    Add,
    Subtract,
    Multiply,
    Divide,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GrayConfig {
    pub min: f64,
    pub max: f64,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
pub enum SourceConfig {
    Stac {
        api: String,
        collection: String,
        assets: Vec<String>,
        /// anchor bbox, min lon, min lat, max lon, max lat. searched once
        /// at startup to fix the grid, crs and band count, pulls past it
        /// search lazily, so it wants to be a populated corner of the
        /// collection rather than the whole world
        bbox: [f64; 4],
        /// the interval every pull takes when it names none
        datetime: Option<String>,
        composite: Option<CompositeConfig>,
    },
    Cog {
        path: PathBuf,
    },
    /// a feature collection read whole at startup, lon/lat per rfc 7946.
    /// features are not pixels, so the chain over it needs a `rasterize`
    Geojson {
        path: PathBuf,
    },
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CompositeConfig {
    Latest,
    Mean,
    Median,
    Min,
    Max,
    /// takes the percent, e.g. `composite = { percentile = 90.0 }`
    Percentile(f64),
    StdDev,
    Count,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OpConfig {
    Hillshade {
        azimuth: f64,
        altitude: f64,
    },
    Bandmath {
        expr: String,
        /// the value range the gray encoding stretches over
        min: f64,
        max: f64,
    },
    Slope,
    Aspect,
    Focal {
        op: FocalOpConfig,
        /// the window is the square of side `2 * radius + 1` cells
        radius: u32,
    },
    Mask {
        /// index of the band the quality codes are read from
        band: usize,
        /// the codes a pixel is kept for, matched exactly
        valid_values: Vec<f64>,
    },
    Reclassify {
        classes: Vec<ClassRange>,
    },
    Unary {
        op: UnaryOpConfig,
    },
    Rasterize {
        burn: BurnConfig,
    },
    Convolve {
        /// the 3x3 taps, rows top to bottom
        kernel: [[f32; 3]; 3],
        /// one scale per band, and their count is the band count the
        /// convolution runs over
        #[serde(default = "identity_scales")]
        scales: Vec<f32>,
        #[serde(default = "identity_offsets")]
        offsets: Vec<f32>,
    },
    /// keep the features whose `field` equals a value, everything else
    /// dropped. `equals = "residential"` and `equals = 3` both work
    VecFilter {
        field: String,
        equals: Value,
    },
    /// rewrite feature properties, geometry untouched: drop first, then
    /// rename what survives, then add the defaults a feature lacks
    VecSchema {
        #[serde(default)]
        drop: Vec<String>,
        #[serde(default)]
        rename: HashMap<String, String>,
        #[serde(default)]
        add: HashMap<String, Value>,
    },
    /// intersect every feature with the polygons of a geojson file, and
    /// drop the ones the cut leaves empty
    VecClip {
        boundary: PathBuf,
    },
}

/// one class a `reclassify` maps into: `min` inclusive, `max` exclusive,
/// and the value every cell between them becomes. a cell in no class
/// becomes nodata, which renders transparent
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClassRange {
    pub min: f64,
    pub max: f64,
    pub value: f64,
}

/// what a `rasterize` burns into every cell a feature covers: a constant
/// for every feature, `burn = { constant = 1.0 }`, or a numeric property
/// read per feature, `burn = { property = "depth" }`, features without it
/// being skipped
#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BurnConfig {
    Constant(f64),
    Property(String),
}

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FocalOpConfig {
    Mean,
    Median,
    Min,
    Max,
}

/// the per-cell operation a `unary` op applies. `add` and `multiply` take
/// their constant, e.g. `op = { multiply = 2.0 }`
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UnaryOpConfig {
    Add(f64),
    Multiply(f64),
    Sqrt,
    Abs,
    Log,
}

/// one channel, values unchanged: what a convolve scales by when the
/// layer names nothing
fn identity_scales() -> Vec<f32> {
    vec![1.0]
}

fn identity_offsets() -> Vec<f32> {
    vec![0.0]
}

impl From<FocalOpConfig> for FocalOp {
    fn from(op: FocalOpConfig) -> FocalOp {
        match op {
            FocalOpConfig::Mean => FocalOp::Mean,
            FocalOpConfig::Median => FocalOp::Median,
            FocalOpConfig::Min => FocalOp::Min,
            FocalOpConfig::Max => FocalOp::Max,
        }
    }
}

impl From<UnaryOpConfig> for UnaryOp {
    fn from(op: UnaryOpConfig) -> UnaryOp {
        match op {
            UnaryOpConfig::Add(constant) => UnaryOp::Add(constant),
            UnaryOpConfig::Multiply(constant) => UnaryOp::Multiply(constant),
            UnaryOpConfig::Sqrt => UnaryOp::Sqrt,
            UnaryOpConfig::Abs => UnaryOp::Abs,
            UnaryOpConfig::Log => UnaryOp::Log,
        }
    }
}

impl From<BinaryOpConfig> for BinaryOp {
    fn from(op: BinaryOpConfig) -> BinaryOp {
        match op {
            BinaryOpConfig::Add => BinaryOp::Add,
            BinaryOpConfig::Subtract => BinaryOp::Subtract,
            BinaryOpConfig::Multiply => BinaryOp::Multiply,
            BinaryOpConfig::Divide => BinaryOp::Divide,
            BinaryOpConfig::Min => BinaryOp::Min,
            BinaryOpConfig::Max => BinaryOp::Max,
        }
    }
}

impl From<&BurnConfig> for Burn {
    fn from(burn: &BurnConfig) -> Burn {
        match burn {
            BurnConfig::Constant(value) => Burn::Constant(*value),
            BurnConfig::Property(name) => Burn::Property(name.clone()),
        }
    }
}

impl From<CompositeConfig> for Composite {
    fn from(c: CompositeConfig) -> Composite {
        match c {
            CompositeConfig::Latest => Composite::Latest,
            CompositeConfig::Mean => Composite::Mean,
            CompositeConfig::Median => Composite::Median,
            CompositeConfig::Min => Composite::Min,
            CompositeConfig::Max => Composite::Max,
            CompositeConfig::Percentile(percent) => Composite::Percentile(percent),
            CompositeConfig::StdDev => Composite::StdDev,
            CompositeConfig::Count => Composite::Count,
        }
    }
}

impl Config {
    /// parse a layer file, or say what is wrong with it. the checks here
    /// are the ones a startup open would only reach after network work
    pub fn parse(text: &str) -> Result<Config, String> {
        let config: Config = toml::from_str(text).map_err(|e| e.to_string())?;
        if config.layers.is_empty() {
            return Err("the layer file names no layers".into());
        }
        let mut seen = HashSet::new();
        for layer in &config.layers {
            if layer.name.is_empty() || layer.name.contains(['/', ' ']) {
                return Err(format!(
                    "layer name {:?} is not a url path segment",
                    layer.name
                ));
            }
            if !seen.insert(&layer.name) {
                return Err(format!("two layers are named {}", layer.name));
            }
            let named = |e: String| format!("layer {}: {e}", layer.name);
            match layer.shape().map_err(named)? {
                LayerShape::Single(source) => {
                    check_branch(source, &layer.ops).map_err(named)?;
                }
                LayerShape::Fanin(_, inputs) => {
                    for (i, input) in inputs.iter().enumerate() {
                        check_branch(&input.source, &input.ops)
                            .map_err(|e| named(format!("input {}: {e}", i + 1)))?;
                    }
                    check_ops(&layer.ops).map_err(named)?;
                }
            }
            layer.gray_range().map_err(named)?;
        }
        Ok(config)
    }
}

/// what one source and the chain over it have to satisfy before the layer
/// can serve tiles
fn check_branch(source: &SourceConfig, ops: &[OpConfig]) -> Result<(), String> {
    match source {
        SourceConfig::Stac { assets, .. } if assets.is_empty() => {
            return Err("stac source names no assets".into());
        }
        SourceConfig::Geojson { .. }
            if !ops
                .iter()
                .any(|op| matches!(op, OpConfig::Rasterize { .. })) =>
        {
            return Err("the geojson source has no rasterize op, so the chain \
                        hands features to a png encoder that wants pixels"
                .into());
        }
        _ => {}
    }
    check_ops(ops)
}

fn check_ops(ops: &[OpConfig]) -> Result<(), String> {
    for op in ops {
        match op {
            OpConfig::Convolve {
                scales, offsets, ..
            } if scales.len() != offsets.len() => {
                return Err(format!(
                    "convolve names {} scales against {} offsets",
                    scales.len(),
                    offsets.len()
                ));
            }
            OpConfig::VecClip { boundary } => {
                read_boundary(boundary)?;
            }
            _ => {}
        }
    }
    Ok(())
}

/// the polygons a `vec_clip` cuts against, every one the file holds joined
/// into the single boundary the element takes. a clip intersects areas, so
/// a file holding anything else has nothing to cut with
pub fn read_boundary(path: &Path) -> Result<MultiPolygon, String> {
    let named = |e: String| format!("clip boundary {}: {e}", path.display());
    let text = std::fs::read_to_string(path).map_err(|e| named(e.to_string()))?;
    let collection = topoi_core::geojson::read_geojson(&text).map_err(|e| named(e.to_string()))?;
    let mut polygons = Vec::new();
    for feature in &collection.features {
        match &feature.geometry {
            Some(FeatureGeometry::Polygon(polygon)) => polygons.push(polygon.clone()),
            Some(FeatureGeometry::MultiPolygon(multi)) => {
                polygons.extend(multi.polygons().iter().cloned())
            }
            other => {
                return Err(named(format!(
                    "a boundary takes polygons, this holds {}",
                    geometry_kind(other)
                )));
            }
        }
    }
    if polygons.is_empty() {
        return Err(named("the file holds no polygons".into()));
    }
    Ok(MultiPolygon::new(polygons))
}

fn geometry_kind(geometry: &Option<FeatureGeometry>) -> &'static str {
    match geometry {
        None => "a feature with no geometry",
        Some(FeatureGeometry::Point(_)) => "a point",
        Some(FeatureGeometry::LineString(_)) => "a linestring",
        Some(FeatureGeometry::MultiPoint(_)) => "a multipoint",
        Some(FeatureGeometry::MultiLineString(_)) => "a multilinestring",
        Some(FeatureGeometry::GeometryCollection(_)) => "a geometry collection",
        Some(FeatureGeometry::Polygon(_)) | Some(FeatureGeometry::MultiPolygon(_)) => "a polygon",
    }
}

impl LayerConfig {
    /// the layer's source side, or what the file got wrong naming it
    pub fn shape(&self) -> Result<LayerShape<'_>, String> {
        match (&self.source, &self.fanin) {
            (Some(_), Some(_)) => {
                Err("names a fanin beside a source, a fanin joins `input` entries".into())
            }
            (Some(_), None) if !self.inputs.is_empty() => {
                Err("names both a source and inputs, a layer takes one or the other".into())
            }
            (Some(source), None) => Ok(LayerShape::Single(source)),
            (None, Some(_)) if self.inputs.is_empty() => {
                Err("names a fanin with no inputs to join".into())
            }
            (None, Some(fanin)) => {
                fanin.check_input_count(self.inputs.len())?;
                Ok(LayerShape::Fanin(fanin, &self.inputs))
            }
            (None, None) => Err("names neither a source nor inputs".into()),
        }
    }

    /// the value range the png encoding stretches over: the layer's own
    /// `gray` where it names one, else the range of the last op that fixes
    /// one. a reclassify leaves class numbers and a rasterize leaves burned
    /// values, which no op range covers, so from there on only the layer
    /// can say what to stretch over. a fan-in layer starts with no range at
    /// all, the input chains not carrying one across the join
    pub fn gray_range(&self) -> Result<(f64, f64), String> {
        if let Some(gray) = self.gray {
            return Ok((gray.min, gray.max));
        }
        let mut range = self.source.as_ref().map(|_| DEFAULT_GRAY);
        for op in &self.ops {
            range = match op {
                OpConfig::Bandmath { min, max, .. } => Some((*min, *max)),
                OpConfig::Slope => Some(SLOPE_GRAY),
                OpConfig::Aspect => Some(ASPECT_GRAY),
                OpConfig::Reclassify { .. } | OpConfig::Rasterize { .. } => None,
                OpConfig::Hillshade { .. }
                | OpConfig::Focal { .. }
                | OpConfig::Mask { .. }
                | OpConfig::Unary { .. }
                | OpConfig::Convolve { .. }
                | OpConfig::VecFilter { .. }
                | OpConfig::VecSchema { .. }
                | OpConfig::VecClip { .. } => range,
            };
        }
        range.ok_or_else(|| match self.source {
            Some(_) => "the ops end in values no op range covers, so the layer needs its own \
                        gray = { min = .., max = .. }"
                .into(),
            None => "a fan-in layer takes no range from its inputs, so it needs its own \
                     gray = { min = .., max = .. } unless an op after the fanin fixes one"
                .to_string(),
        })
    }
}

impl FaninConfig {
    fn check_input_count(&self, count: usize) -> Result<(), String> {
        match self {
            FaninConfig::Mosaic if count < 2 => Err(format!(
                "mosaic joins two or more inputs, the layer names {count}"
            )),
            FaninConfig::Combine { .. } if count != 2 => Err(format!(
                "combine joins exactly two inputs, the layer names {count}"
            )),
            _ => Ok(()),
        }
    }
}

impl SourceConfig {
    /// the search a stac layer opens with, `None` for any other kind
    pub fn stac_search(&self) -> Option<StacSearch> {
        match self {
            SourceConfig::Cog { .. } | SourceConfig::Geojson { .. } => None,
            SourceConfig::Stac {
                api,
                collection,
                assets,
                bbox,
                datetime,
                composite,
            } => {
                let mut search = StacSearch::new(api, collection, &assets[0], *bbox);
                search.assets = assets.clone();
                search.datetime = datetime.clone();
                search.composite = composite.map(Composite::from).unwrap_or_default();
                Some(search)
            }
        }
    }
}
