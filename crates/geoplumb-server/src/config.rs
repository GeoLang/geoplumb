//! the layer file: one entry per served layer, a source plus an ordered
//! op pipeline. every layer ends reprojected to web mercator and encoded
//! as grayscale png, so the file says what varies and nothing else

use std::collections::HashSet;
use std::path::PathBuf;

use geoplumb::elements::{Composite, FocalOp, StacSearch};
use serde::Deserialize;
use terrano_core::UnaryOp;

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
    pub source: SourceConfig,
    #[serde(default, rename = "op")]
    pub ops: Vec<OpConfig>,
    /// the value range the png encoding stretches over, overriding the one
    /// the ops imply, `gray = { min = 0.0, max = 3.0 }`
    pub gray: Option<GrayConfig>,
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
#[serde(tag = "kind", rename_all = "lowercase", deny_unknown_fields)]
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
            if let SourceConfig::Stac { assets, .. } = &layer.source {
                if assets.is_empty() {
                    return Err(format!("layer {}: stac source names no assets", layer.name));
                }
            }
            layer
                .gray_range()
                .map_err(|e| format!("layer {}: {e}", layer.name))?;
            for op in &layer.ops {
                if let OpConfig::Convolve {
                    scales, offsets, ..
                } = op
                {
                    if scales.len() != offsets.len() {
                        return Err(format!(
                            "layer {}: convolve names {} scales against {} offsets",
                            layer.name,
                            scales.len(),
                            offsets.len()
                        ));
                    }
                }
            }
        }
        Ok(config)
    }
}

impl LayerConfig {
    /// the value range the png encoding stretches over: the layer's own
    /// `gray` where it names one, else the range of the last op that fixes
    /// one. a reclassify leaves class numbers, which no op range covers,
    /// so from there on only the layer can say what to stretch over
    pub fn gray_range(&self) -> Result<(f64, f64), String> {
        if let Some(gray) = self.gray {
            return Ok((gray.min, gray.max));
        }
        let mut range = Some(DEFAULT_GRAY);
        for op in &self.ops {
            range = match op {
                OpConfig::Bandmath { min, max, .. } => Some((*min, *max)),
                OpConfig::Slope => Some(SLOPE_GRAY),
                OpConfig::Aspect => Some(ASPECT_GRAY),
                OpConfig::Reclassify { .. } => None,
                OpConfig::Hillshade { .. }
                | OpConfig::Focal { .. }
                | OpConfig::Mask { .. }
                | OpConfig::Unary { .. }
                | OpConfig::Convolve { .. } => range,
            };
        }
        range.ok_or_else(|| {
            "the ops end in reclassify class values, so the layer needs its own \
             gray = { min = .., max = .. }"
                .into()
        })
    }
}

impl SourceConfig {
    /// the search a stac layer opens with, `None` for any other kind
    pub fn stac_search(&self) -> Option<StacSearch> {
        match self {
            SourceConfig::Cog { .. } => None,
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
