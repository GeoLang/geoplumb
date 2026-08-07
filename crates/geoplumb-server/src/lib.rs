//! http tile server over geoplumb. one engine per configured layer, xyz
//! png tiles, and a per-request time interval wherever the layer's source
//! resolves one. layers come from the toml named by `GEOPLUMB_LAYERS`,
//! the service invents none of its own.
//!
//! the endpoints are public by decision: these are tiles rendered from
//! public collections, the same policy tiletopia's public tiles follow

pub mod config;

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use serde::{Deserialize, Serialize};

use geoplumb::elements::{Aspect, BandMath, CogSrc, Hillshade, Reproject, Slope, StacSrc};
use geoplumb::tile::{XyzTile, render_tile_at};
use geoplumb::{Crs, Engine, Graph, NodeId, Source, TimeInterval};

use config::{Config, LayerConfig, OpConfig, SourceConfig};

/// past this a tile index stops fitting the zoom arithmetic, and no
/// viewer asks for it
const MAX_ZOOM: u8 = 24;

/// the gray range png encoding stretches over when no bandmath op names
/// one, hillshade's own range
const DEFAULT_GRAY: (f64, f64) = (0.0, 255.0);
/// slope is degrees from horizontal
const SLOPE_GRAY: (f64, f64) = (0.0, 90.0);
/// aspect is compass degrees
const ASPECT_GRAY: (f64, f64) = (0.0, 360.0);

/// disk tier size per layer, derived from the memory budget rather than
/// given its own knob: the tier only exists when a disk dir is set
const DISK_BUDGET_FACTOR: usize = 8;

/// one served layer: the graph is solved once at startup and the engine
/// caches every chunk it computes for the process's life
pub struct Layer {
    engine: Engine,
    node: NodeId,
    gray: (f64, f64),
    info: LayerInfo,
}

/// what `/layers` publishes about a layer
#[derive(Debug, Clone, Serialize)]
pub struct LayerInfo {
    pub name: String,
    /// `stac` or `cog`
    pub source: &'static str,
    pub collection: Option<String>,
    /// the interval pulls take when they name none
    pub default_datetime: Option<String>,
    pub temporal_extent: Option<TemporalExtent>,
}

/// a stac collection's advertised temporal extent, either end open
#[derive(Debug, Clone, Serialize)]
pub struct TemporalExtent {
    pub start: Option<String>,
    pub end: Option<String>,
}

impl Layer {
    /// build every layer, or fail naming the one that broke. blocking: a
    /// stac layer searches its anchor bbox and a cog layer reads a file
    /// header, so callers under a runtime want `spawn_blocking`
    pub fn build_all(
        config: &Config,
        budget_bytes: usize,
        disk: Option<&Path>,
    ) -> Result<Vec<Layer>, String> {
        config
            .layers
            .iter()
            .map(|layer| {
                Layer::build(layer, budget_bytes, disk)
                    .map_err(|e| format!("layer {}: {e}", layer.name))
            })
            .collect()
    }

    fn build(cfg: &LayerConfig, budget_bytes: usize, disk: Option<&Path>) -> Result<Layer, String> {
        let (source, info) = open_source(cfg)?;
        let mut graph = Graph::new();
        let mut node = graph.add_source(source);
        let mut gray = DEFAULT_GRAY;
        for op in &cfg.ops {
            node = match op {
                OpConfig::Hillshade { azimuth, altitude } => {
                    graph.add_transform(node, Box::new(Hillshade::new(*azimuth, *altitude)))
                }
                OpConfig::Bandmath { expr, min, max } => {
                    gray = (*min, *max);
                    let math = BandMath::new(expr).map_err(|e| e.to_string())?;
                    graph.add_transform(node, Box::new(math))
                }
                OpConfig::Slope => {
                    gray = SLOPE_GRAY;
                    graph.add_transform(node, Box::new(Slope))
                }
                OpConfig::Aspect => {
                    gray = ASPECT_GRAY;
                    graph.add_transform(node, Box::new(Aspect))
                }
            };
        }
        let node = graph.add_transform(node, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
        let engine = match disk {
            None => Engine::new(graph, budget_bytes),
            Some(dir) => {
                Engine::with_disk_cache(graph, budget_bytes, dir, budget_bytes * DISK_BUDGET_FACTOR)
            }
        }
        .map_err(|e| e.to_string())?;
        Ok(Layer {
            engine,
            node,
            gray,
            info,
        })
    }

    pub fn info(&self) -> &LayerInfo {
        &self.info
    }
}

fn open_source(cfg: &LayerConfig) -> Result<(Box<dyn Source>, LayerInfo), String> {
    let name = cfg.name.clone();
    match &cfg.source {
        SourceConfig::Cog { path } => {
            let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let src = CogSrc::open(file).map_err(|e| e.to_string())?;
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "cog",
                    collection: None,
                    default_datetime: None,
                    temporal_extent: None,
                },
            ))
        }
        SourceConfig::Stac {
            api,
            collection,
            datetime,
            ..
        } => {
            let search = cfg
                .source
                .stac_search()
                .expect("a stac source has a search");
            let src = StacSrc::open(&search).map_err(|e| e.to_string())?;
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "stac",
                    collection: Some(collection.clone()),
                    default_datetime: datetime.clone(),
                    temporal_extent: temporal_extent(api, collection),
                },
            ))
        }
    }
}

/// the collection's advertised temporal extent, so a client knows what
/// `t` can ask for. one short probe at startup, and `None` whenever the
/// api does not answer it: a layer that serves tiles should not fail
/// over a metadata nicety, and no request ever probes again
fn temporal_extent(api: &str, collection: &str) -> Option<TemporalExtent> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .ok()?;
    let text = client
        .get(format!("{api}/collections/{collection}"))
        .send()
        .and_then(|r| r.error_for_status())
        .and_then(|r| r.text())
        .ok()?;
    let body: serde_json::Value = serde_json::from_str(&text).ok()?;
    let interval = body["extent"]["temporal"]["interval"][0].as_array()?;
    let at = |i: usize| interval.get(i).and_then(|v| v.as_str()).map(str::to_string);
    Some(TemporalExtent {
        start: at(0),
        end: at(1),
    })
}

/// the `t` query parameter: an rfc 3339 `start/end` interval, absent
/// meaning the layer source's own configured time
pub fn parse_time(t: Option<&str>) -> Result<Option<TimeInterval>, String> {
    let Some(text) = t else { return Ok(None) };
    // the engine phrases its parse errors as source failures, which is
    // the wrong thing to tell someone who mistyped a query parameter
    TimeInterval::parse(text).map(Some).map_err(|e| match e {
        geoplumb::Error::Source(detail) => format!("bad t parameter: {detail}"),
        other => format!("bad t parameter: {other}"),
    })
}

type Layers = Arc<Vec<Layer>>;

pub fn router(layers: Vec<Layer>) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/layers", get(list_layers))
        .route("/tiles/{layer}/{z}/{x}/{y}", get(tile))
        .with_state(Arc::new(layers))
}

async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({ "status": "ok" }))
}

async fn list_layers(State(layers): State<Layers>) -> axum::Json<Vec<LayerInfo>> {
    axum::Json(layers.iter().map(|l| l.info.clone()).collect())
}

#[derive(Deserialize)]
struct TileQuery {
    t: Option<String>,
}

async fn tile(
    State(layers): State<Layers>,
    UrlPath((name, z, x, y)): UrlPath<(String, u8, u32, String)>,
    Query(query): Query<TileQuery>,
) -> Response {
    let Some(layer) = layers.iter().find(|l| l.info.name == name) else {
        return (StatusCode::NOT_FOUND, format!("unknown layer {name}")).into_response();
    };
    let Ok(y) = y.trim_end_matches(".png").parse::<u32>() else {
        return bad_request(format!("tile y {y} is not a number"));
    };
    if z > MAX_ZOOM {
        return bad_request(format!("zoom {z} is past the {MAX_ZOOM} cap"));
    }
    let side = 1u32 << z;
    if x >= side || y >= side {
        return bad_request(format!("tile {x}/{y} is outside zoom {z}"));
    }
    let time = match parse_time(query.t.as_deref()) {
        Ok(time) => time,
        Err(e) => return bad_request(e),
    };
    let chunk = match render_tile_at(&layer.engine, layer.node, XyzTile { z, x, y }, time).await {
        Ok(chunk) => chunk,
        Err(e) => return (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    };
    match geoplumb::encode::png_gray(&chunk, layer.gray.0, layer.gray.1) {
        Ok(png) => ([(header::CONTENT_TYPE, "image/png")], png).into_response(),
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

fn bad_request(message: String) -> Response {
    (StatusCode::BAD_REQUEST, message).into_response()
}
