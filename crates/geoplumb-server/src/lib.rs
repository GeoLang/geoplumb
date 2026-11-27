//! http tile server over geoplumb. one engine per configured layer, xyz
//! tiles as gray png or, where the path ends in `.tif`, as geotiff, or as
//! mapbox vector tiles for a layer ending in features, and a
//! per-request time interval wherever the layer's source resolves one.
//! layers come from the toml named by `GEOPLUMB_LAYERS`, the service
//! invents none of its own.
//!
//! the endpoints are public by decision: these are tiles rendered from
//! public collections, the same policy tiletopia's public tiles follow

// proving Send for geoplumb's wgpu chunks exceeds the default limit
#![recursion_limit = "256"]

pub mod cli;
pub mod config;
mod export;
mod mvt;
mod zonal;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::{Deserialize, Serialize};
use tokio::sync::Semaphore;

use geoplumb::caps::{BandDtypes, ValueKind};
#[cfg(feature = "gdal")]
use geoplumb::elements::GdalSrc;
use geoplumb::elements::algebra::AlgebraOp;
use geoplumb::elements::stac::format_rfc3339;
use geoplumb::elements::{
    Aspect, BandMath, Cast, ClipRaster, CogSrc, Combine, Contours, CopcSrc, FlowAccumulation,
    FlowDirection, Focal, GpkgSrc, Hillshade, IdwGrid, LasSrc, MapAlgebra, Mosaic, NetcdfSrc,
    Polygonize, QualityMask, RasterSrc, Rasterize, Reproject, Slope, StacSrc, StreamOrder,
    TensorConv, ToRaster, ToTensor, VecClip, VecFilter, VecSchema, VecSrc, Viewshed, Watershed,
};
use geoplumb::tile::{XyzTile, render_tile_at};
use geoplumb::{Bbox, Caps, CapsSet, Crs, Engine, Graph, NodeId, Source, TimeInterval};

use config::{Config, FaninConfig, LayerConfig, LayerShape, OpConfig, SourceConfig};

pub use export::ExportQuery;

pub const DEFAULT_CACHE_BYTES: usize = 256 << 20;

/// past this a tile index stops fitting the zoom arithmetic, and no
/// viewer asks for it
const MAX_ZOOM: u8 = 24;

/// disk tier size per layer, derived from the memory budget rather than
/// given its own knob: the tier only exists when a disk dir is set
const DISK_BUDGET_FACTOR: usize = 8;

/// one served layer: the graph is solved once at startup and the engine
/// caches every chunk it computes for the process's life
pub struct Layer {
    engine: Engine,
    node: NodeId,
    output: LayerOutput,
    info: LayerInfo,
    config: LayerConfig,
    layer_file: Arc<[LayerConfig]>,
    chain_crs: Crs,
    budget_bytes: usize,
}

enum LayerOutput {
    Raster { gray: (f64, f64) },
    Vector,
}

/// what `/layers` publishes about a layer
#[derive(Debug, Clone, Serialize)]
pub struct LayerInfo {
    pub name: String,
    /// `stac`, `cog`, `gdal`, `geojson`, `gpkg`, `las`, `copc`, `netcdf`, `grib`, or
    /// `composite` for a fan-in layer
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
        config: Config,
        budget_bytes: usize,
        disk: Option<&Path>,
    ) -> Result<Vec<Layer>, String> {
        let layer_file: Arc<[LayerConfig]> = config.layers.clone().into();
        config
            .layers
            .into_iter()
            .map(|layer| {
                let name = layer.name.clone();
                Layer::build(layer, Arc::clone(&layer_file), budget_bytes, disk)
                    .map_err(|e| format!("layer {name}: {e}"))
            })
            .collect()
    }

    fn build(
        cfg: LayerConfig,
        layer_file: Arc<[LayerConfig]>,
        budget_bytes: usize,
        disk: Option<&Path>,
    ) -> Result<Layer, String> {
        let LayerGraph {
            mut graph,
            node: chain_tip,
            info,
            chain_crs,
        } = build_layer_graph(&cfg, &layer_file)?;
        let node = graph.add_transform(chain_tip, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
        let engine = match disk {
            None => Engine::new(graph, budget_bytes),
            Some(dir) => {
                Engine::with_disk_cache(graph, budget_bytes, dir, budget_bytes * DISK_BUDGET_FACTOR)
            }
        }
        .map_err(|e| e.to_string())?;
        let output = match engine.caps(node) {
            Caps::Vector(_) if cfg.gray.is_some() => {
                return Err("the chain ends in features, served as vector tiles, \
                            so the layer takes no gray"
                    .into());
            }
            Caps::Vector(_) => LayerOutput::Vector,
            Caps::Raster(_) => LayerOutput::Raster {
                gray: cfg.gray_range()?,
            },
            Caps::PointCloud(_) | Caps::Tensor(_) => {
                unreachable!("a reproject outputs raster or vector caps")
            }
        };
        Ok(Layer {
            engine,
            node,
            output,
            info,
            config: cfg,
            layer_file,
            chain_crs,
            budget_bytes,
        })
    }

    pub fn info(&self) -> &LayerInfo {
        &self.info
    }
}

pub struct LayerGraph {
    pub graph: Graph,
    pub node: NodeId,
    pub info: LayerInfo,
    pub chain_crs: Crs,
}

pub fn build_layer_graph(
    layer: &LayerConfig,
    layer_file: &[LayerConfig],
) -> Result<LayerGraph, String> {
    let mut graph = Graph::new();
    let LayerChain {
        node,
        chain_crs,
        info,
    } = add_layer_chain(&mut graph, layer, layer_file)?;
    Ok(LayerGraph {
        graph,
        node,
        info,
        chain_crs,
    })
}

struct LayerChain {
    node: NodeId,
    chain_crs: Crs,
    info: LayerInfo,
}

fn add_layer_chain(
    graph: &mut Graph,
    layer: &LayerConfig,
    layer_file: &[LayerConfig],
) -> Result<LayerChain, String> {
    let (node, crs, info) = match layer.shape()? {
        LayerShape::Single(source) => {
            let (source, info) = open_source(&layer.name, source)?;
            let crs = source_crs(source.as_ref())?;
            (graph.add_source(source), crs, info)
        }
        LayerShape::Fanin(fanin, inputs) => {
            let mut tips = Vec::with_capacity(inputs.len());
            let mut infos = Vec::with_capacity(inputs.len());
            let mut input_crs = Vec::with_capacity(inputs.len());
            for input in inputs {
                let (source, info) = open_source(&layer.name, &input.source)?;
                let crs = source_crs(source.as_ref())?;
                let head = graph.add_source(source);
                let (tip, tip_crs) = add_ops(graph, head, &input.ops, crs, layer_file)?;
                tips.push(tip);
                infos.push(info);
                input_crs.push(tip_crs);
            }
            let joined = match fanin {
                FaninConfig::Mosaic => graph.add_fanin(&tips, Box::new(Mosaic::default())),
                FaninConfig::Combine { op } => {
                    graph.add_fanin(&tips, Box::new(Combine::new((*op).into())))
                }
            };
            // the solver reprojects every later fan-in input onto the first one's crs
            (joined, input_crs[0], composite_info(&layer.name, infos))
        }
    };
    let (node, chain_crs) = add_ops(graph, node, &layer.ops, crs, layer_file)?;
    Ok(LayerChain {
        node,
        chain_crs,
        info,
    })
}

fn source_crs(source: &dyn Source) -> Result<Crs, String> {
    source
        .constraint()
        .output_set(&CapsSet::any())
        .fixate()
        .map(|caps| caps.crs())
        .ok_or_else(|| "the source names no crs".to_string())
}

fn region_bbox([min_x, min_y, max_x, max_y]: [f64; 4]) -> Bbox {
    Bbox::new(min_x, min_y, max_x, max_y)
}

/// one op chain over `node`, the translation a layer's own ops and a
/// fan-in input's both take
fn add_ops(
    graph: &mut Graph,
    node: NodeId,
    ops: &[OpConfig],
    crs: Crs,
    layer_file: &[LayerConfig],
) -> Result<(NodeId, Crs), String> {
    let mut node = node;
    let mut crs = crs;
    for op in ops {
        node = match op {
            OpConfig::Hillshade { azimuth, altitude } => {
                graph.add_transform(node, Box::new(Hillshade::new(*azimuth, *altitude)))
            }
            OpConfig::Bandmath { expr, .. } => {
                let math = BandMath::new(expr).map_err(|e| e.to_string())?;
                graph.add_transform(node, Box::new(math))
            }
            OpConfig::Slope => graph.add_transform(node, Box::new(Slope)),
            OpConfig::Aspect => graph.add_transform(node, Box::new(Aspect)),
            OpConfig::Focal { op, radius } => {
                graph.add_transform(node, Box::new(Focal::new((*op).into(), *radius)))
            }
            OpConfig::Mask { band, valid_values } => graph.add_transform(
                node,
                Box::new(QualityMask::new(*band, valid_values.clone())),
            ),
            OpConfig::Reclassify { classes } => {
                let ranges = classes.iter().map(|c| (c.min, c.max, c.value)).collect();
                let algebra = MapAlgebra::new(AlgebraOp::Reclassify(ranges));
                graph.add_transform(node, Box::new(algebra))
            }
            OpConfig::Unary { op } => {
                let algebra = MapAlgebra::new(AlgebraOp::Unary((*op).into()));
                graph.add_transform(node, Box::new(algebra))
            }
            OpConfig::Cast { dtype, nodata } => {
                let to = config::cast_dtype(dtype, *nodata)?;
                graph.add_transform(node, Box::new(Cast::new(BandDtypes::from(to))))
            }
            OpConfig::ClipRaster { cutline, nodata } => {
                let cutline_layer = config::cutline_layer(layer_file, cutline)?;
                let polygons = add_layer_chain(graph, cutline_layer, layer_file)
                    .map_err(|e| format!("cutline layer {cutline}: {e}"))?;
                graph.add_fanin(&[node, polygons.node], Box::new(ClipRaster::new(*nodata)))
            }
            OpConfig::Rasterize { burn } => {
                graph.add_transform(node, Box::new(Rasterize { burn: burn.into() }))
            }
            OpConfig::Convolve {
                kernel,
                scales,
                offsets,
            } => {
                let to_tensor = ToTensor {
                    scales: scales.clone(),
                    offsets: offsets.clone(),
                };
                let tensor = graph.add_transform(node, Box::new(to_tensor));
                let convolved =
                    graph.add_transform(tensor, Box::new(TensorConv { kernel: *kernel }));
                graph.add_transform(convolved, Box::new(ToRaster))
            }
            OpConfig::VecFilter { field, equals } => graph.add_transform(
                node,
                Box::new(VecFilter::new(field.clone(), equals.clone())),
            ),
            OpConfig::VecSchema { drop, rename, add } => graph.add_transform(
                node,
                Box::new(VecSchema {
                    drop: drop.clone(),
                    rename: rename.clone(),
                    add: add.clone(),
                }),
            ),
            OpConfig::VecClip { boundary } => {
                let boundary = config::read_boundary(boundary)?;
                graph.add_transform(node, Box::new(VecClip { boundary }))
            }
            OpConfig::Idw {
                power,
                radius_px,
                min_points,
            } => {
                let default = IdwGrid::default();
                let idw = IdwGrid {
                    power: power.unwrap_or(default.power),
                    radius_px: radius_px.unwrap_or(default.radius_px),
                    min_points: min_points.unwrap_or(default.min_points),
                };
                graph.add_transform(node, Box::new(idw))
            }
            OpConfig::Contours { interval, base } => graph.add_transform(
                node,
                Box::new(Contours {
                    interval: *interval,
                    base: *base,
                }),
            ),
            OpConfig::Polygonize => graph.add_transform(node, Box::new(Polygonize)),
            OpConfig::Viewshed {
                lon,
                lat,
                height,
                radius,
            } => {
                let (observer_x, observer_y) =
                    projicio_core::Transform::new(&Crs::WGS84.authority(), &crs.authority())
                        .and_then(|transform| transform.convert(*lon, *lat))
                        .map_err(|e| format!("viewshed observer {lon}, {lat} into {crs}: {e}"))?;
                let viewshed = Viewshed {
                    observer_x,
                    observer_y,
                    observer_height: *height,
                    radius: *radius,
                };
                graph.add_transform(node, Box::new(viewshed))
            }
            OpConfig::FlowDirection { crs: to, region } => {
                crs = to.0;
                let element = FlowDirection {
                    crs,
                    region: region_bbox(*region),
                };
                graph.add_transform(node, Box::new(element))
            }
            OpConfig::FlowAccumulation { crs: to, region } => {
                crs = to.0;
                let element = FlowAccumulation {
                    crs,
                    region: region_bbox(*region),
                };
                graph.add_transform(node, Box::new(element))
            }
            OpConfig::Watershed { crs: to, region } => {
                crs = to.0;
                let element = Watershed {
                    crs,
                    region: region_bbox(*region),
                };
                graph.add_transform(node, Box::new(element))
            }
            OpConfig::StreamOrder {
                crs: to,
                region,
                threshold,
            } => {
                crs = to.0;
                let element = StreamOrder {
                    crs,
                    region: region_bbox(*region),
                    threshold: *threshold,
                };
                graph.add_transform(node, Box::new(element))
            }
        };
    }
    Ok((node, crs))
}

/// what `/layers` publishes about a fan-in layer: no collection of its
/// own, and the time fields of the first input that has any
fn composite_info(name: &str, inputs: Vec<LayerInfo>) -> LayerInfo {
    let timed = inputs
        .into_iter()
        .find(|i| i.default_datetime.is_some() || i.temporal_extent.is_some());
    LayerInfo {
        name: name.to_string(),
        source: "composite",
        collection: None,
        default_datetime: timed.as_ref().and_then(|i| i.default_datetime.clone()),
        temporal_extent: timed.and_then(|i| i.temporal_extent),
    }
}

fn open_source(layer: &str, source: &SourceConfig) -> Result<(Box<dyn Source>, LayerInfo), String> {
    let name = layer.to_string();
    match source {
        SourceConfig::Cog {
            path,
            value_kind,
            nodata,
        } => {
            let named = |e: String| format!("{}: {e}", path.display());
            let file = std::fs::File::open(path).map_err(|e| named(e.to_string()))?;
            let src = CogSrc::open(file, ValueKind::from(*value_kind), *nodata)
                .map_err(|e| named(e.to_string()))?;
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
        #[cfg(feature = "gdal")]
        SourceConfig::Gdal {
            path,
            band,
            value_kind,
        } => {
            let src = match band {
                None => GdalSrc::open(path, ValueKind::from(*value_kind)),
                Some(band) => GdalSrc::open_band(path, *band, ValueKind::from(*value_kind)),
            }
            .map_err(|e| e.to_string())?;
            println!("layer {layer}: gdal driver {}", src.driver());
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "gdal",
                    collection: None,
                    default_datetime: None,
                    temporal_extent: None,
                },
            ))
        }
        #[cfg(not(feature = "gdal"))]
        SourceConfig::Gdal { .. } => Err(config::GDAL_NOT_BUILT.into()),
        SourceConfig::Geojson { path } => {
            let text =
                std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
            // rfc 7946 puts geojson coordinates in lon/lat
            let src = VecSrc::from_geojson(&text, Crs::WGS84).map_err(|e| e.to_string())?;
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "geojson",
                    collection: None,
                    default_datetime: None,
                    temporal_extent: None,
                },
            ))
        }
        SourceConfig::Gpkg { path, table, crs } => {
            let src =
                GpkgSrc::open(path, table.as_deref(), crs.map(Crs)).map_err(|e| e.to_string())?;
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "gpkg",
                    collection: None,
                    default_datetime: None,
                    temporal_extent: None,
                },
            ))
        }
        SourceConfig::Las { path, crs } => {
            let mut file =
                std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
            let src = LasSrc::from_las(&mut file, Crs(*crs)).map_err(|e| e.to_string())?;
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "las",
                    collection: None,
                    default_datetime: None,
                    temporal_extent: None,
                },
            ))
        }
        SourceConfig::Copc { path, url, crs } => {
            let src: Box<dyn Source> = match (path, url) {
                (Some(path), None) => {
                    Box::new(CopcSrc::open_path(path, Crs(*crs)).map_err(|e| e.to_string())?)
                }
                (None, Some(url)) => {
                    Box::new(CopcSrc::open_url(url, Crs(*crs)).map_err(|e| e.to_string())?)
                }
                _ => return Err(config::COPC_NEEDS_ONE_LOCATION.into()),
            };
            Ok((
                src,
                LayerInfo {
                    name,
                    source: "copc",
                    collection: None,
                    default_datetime: None,
                    temporal_extent: None,
                },
            ))
        }
        SourceConfig::Netcdf {
            path,
            variable,
            crs,
            datetime,
            select,
            value_kind,
        } => {
            let interval = datetime
                .as_deref()
                .map(TimeInterval::parse)
                .transpose()
                .map_err(|e| e.to_string())?;
            let src = NetcdfSrc::open(
                path,
                variable,
                select,
                crs.map(Crs),
                interval,
                (*value_kind).into(),
            )
            .map_err(|e| e.to_string())?;
            let steps = src.step_instants_ms();
            let temporal_extent = steps
                .iter()
                .min()
                .zip(steps.iter().max())
                .map(|(start, end)| TemporalExtent {
                    start: Some(format_rfc3339(*start)),
                    end: Some(format_rfc3339(*end)),
                });
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "netcdf",
                    collection: None,
                    default_datetime: datetime.clone(),
                    temporal_extent,
                },
            ))
        }
        SourceConfig::Grib {
            path,
            message,
            value_kind,
        } => {
            let src = RasterSrc::from_grib(path, *message, (*value_kind).into())
                .map_err(|e| e.to_string())?;
            Ok((
                Box::new(src),
                LayerInfo {
                    name,
                    source: "grib",
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
            let search = source.stac_search().expect("a stac source has a search");
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

struct ServerState {
    layers: Vec<Layer>,
    reductions: Arc<Semaphore>,
    exports: Arc<Semaphore>,
    export_dir: Option<PathBuf>,
}

pub fn router(layers: Vec<Layer>, export_dir: Option<PathBuf>) -> Router {
    let state = Arc::new(ServerState {
        layers,
        reductions: Arc::new(Semaphore::new(zonal::MAX_CONCURRENT_REDUCTIONS)),
        exports: Arc::new(Semaphore::new(export::MAX_CONCURRENT_EXPORTS)),
        export_dir,
    });
    Router::new()
        .route("/health", get(health))
        .route("/layers", get(list_layers))
        .route("/tiles/{layer}/{z}/{x}/{y}", get(tile))
        .route("/zonal/{layer}", post(zonal::statistics))
        .route("/zonal/{layer}/series", post(zonal::series))
        .route("/export/{layer}", get(export::cog))
        .with_state(state)
}

async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(serde_json::json!({ "status": "ok" }))
}

#[derive(Serialize)]
struct ListedLayer {
    #[serde(flatten)]
    info: LayerInfo,
    kind: &'static str,
}

async fn list_layers(State(state): State<Arc<ServerState>>) -> axum::Json<Vec<ListedLayer>> {
    let listed = state.layers.iter().map(|l| ListedLayer {
        info: l.info.clone(),
        kind: match l.output {
            LayerOutput::Raster { .. } => "raster",
            LayerOutput::Vector => "vector",
        },
    });
    axum::Json(listed.collect())
}

#[derive(Deserialize)]
struct TileQuery {
    t: Option<String>,
}

#[derive(Clone, Copy)]
enum TileFormat {
    Png,
    GeoTiff,
    Mvt,
}

async fn tile(
    State(state): State<Arc<ServerState>>,
    UrlPath((name, z, x, y)): UrlPath<(String, u8, u32, String)>,
    Query(query): Query<TileQuery>,
) -> Response {
    let Some(layer) = state.layers.iter().find(|l| l.info.name == name) else {
        return (StatusCode::NOT_FOUND, format!("unknown layer {name}")).into_response();
    };
    let (row, format) = match (y.strip_suffix(".tif"), y.strip_suffix(".mvt")) {
        (Some(row), _) => (row, TileFormat::GeoTiff),
        (None, Some(row)) => (row, TileFormat::Mvt),
        (None, None) => (y.trim_end_matches(".png"), TileFormat::Png),
    };
    let Ok(y) = row.parse::<u32>() else {
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
    let tile = XyzTile { z, x, y };
    match encode_tile(layer, tile, time, format).await {
        Ok((content_type, bytes)) => {
            ([(header::CONTENT_TYPE, content_type)], bytes).into_response()
        }
        Err(TileError::WrongFormat(message)) => bad_request(message),
        Err(TileError::Engine(e)) => engine_failure(&e),
    }
}

enum TileError {
    WrongFormat(String),
    Engine(geoplumb::Error),
}

async fn encode_tile(
    layer: &Layer,
    tile: XyzTile,
    time: Option<TimeInterval>,
    format: TileFormat,
) -> Result<(&'static str, Vec<u8>), TileError> {
    let name = &layer.info.name;
    let encoded = match (&layer.output, format) {
        (LayerOutput::Raster { gray }, TileFormat::Png) => {
            render_tile_at(&layer.engine, layer.node, tile, time)
                .await
                .and_then(|chunk| geoplumb::encode::png_gray(&chunk, gray.0, gray.1))
                .map(|bytes| ("image/png", bytes))
        }
        (LayerOutput::Raster { .. }, TileFormat::GeoTiff) => {
            render_tile_at(&layer.engine, layer.node, tile, time)
                .await
                .and_then(|chunk| {
                    geoplumb::encode::geotiff(&chunk, layer.engine.palette(layer.node))
                })
                .map(|bytes| ("image/tiff", bytes))
        }
        (LayerOutput::Vector, TileFormat::Mvt) => mvt::vector_tile(layer, tile, time)
            .await
            .map(|bytes| (jung_mvt::MVT_MEDIA_TYPE, bytes)),
        (LayerOutput::Raster { .. }, TileFormat::Mvt) => {
            return Err(TileError::WrongFormat(format!(
                "layer {name} is a raster layer, its tiles are .png or .tif"
            )));
        }
        (LayerOutput::Vector, TileFormat::Png | TileFormat::GeoTiff) => {
            return Err(TileError::WrongFormat(format!(
                "layer {name} is a vector layer, its tiles are .mvt"
            )));
        }
    };
    encoded.map_err(TileError::Engine)
}

fn bad_request(message: String) -> Response {
    (StatusCode::BAD_REQUEST, message).into_response()
}

fn engine_failure(error: &geoplumb::Error) -> Response {
    let status = match error {
        geoplumb::Error::RegionTooLarge { .. } => StatusCode::BAD_REQUEST,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    (status, error.to_string()).into_response()
}
