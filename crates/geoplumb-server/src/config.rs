//! the layer file: one entry per served layer, a source plus an ordered
//! op pipeline. every layer ends reprojected to web mercator and encoded
//! as grayscale png, so the file says what varies and nothing else

use std::collections::HashSet;
use std::path::PathBuf;

use geoplumb::elements::{Composite, StacSearch};
use serde::Deserialize;

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
        }
        Ok(config)
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
