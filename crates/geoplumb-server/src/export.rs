use std::io::SeekFrom;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use geoplumb::elements::Reproject;
use geoplumb::export::{COG_TILE_PX, ExportGrid, write_cog};
use geoplumb::{Bbox, Caps, Crs, Engine, NodeId, TimeInterval};
use projicio_core::Transform;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::config::parse_epsg;
use crate::{
    Layer, LayerGraph, ServerState, bad_request, build_layer_graph, engine_failure, parse_time,
};

pub(crate) const MAX_CONCURRENT_EXPORTS: usize = 2;

// proxies in front of the service typically cut a request at five minutes
const EXPORT_TIMEOUT: Duration = Duration::from_secs(240);

// 16384 squared f64 values is 2 GiB before deflate, and classic tiff offsets stop at 4 GiB
const MAX_EXPORT_VALUES: f64 = 16384.0 * 16384.0;

const STREAM_CHUNK_BYTES: usize = 64 << 10;

#[derive(Deserialize)]
pub struct ExportQuery {
    pub bbox: String,
    pub crs: String,
    pub resolution: f64,
    pub t: Option<String>,
}

#[derive(Clone, Copy)]
pub(crate) struct Plan {
    pub(crate) crs: Crs,
    pub(crate) grid: ExportGrid,
    pub(crate) time: Option<TimeInterval>,
}

impl Plan {
    pub(crate) fn from_query(query: &ExportQuery, chain_crs: Crs) -> Result<Plan, String> {
        let time = parse_time(query.t.as_deref())?;
        let bbox = parse_bbox(&query.bbox)?;
        let resolution = query.resolution;
        if !resolution.is_finite() || resolution <= 0.0 {
            return Err(format!(
                "resolution {resolution} is not a positive number of crs units per pixel"
            ));
        }
        let crs = parse_crs(&query.crs, chain_crs)?;

        // gdalwarp's -te with -tr rounds the pixel count and keeps at least one
        let cols = (bbox.width() / resolution).round().max(1.0);
        let rows = (bbox.height() / resolution).round().max(1.0);
        // tiff stores the image width and height as 32 bit integers
        let tiff_side = f64::from(u32::MAX);
        if cols > tiff_side || rows > tiff_side {
            return Err(format!(
                "{cols} x {rows} pixels is past the {} pixel side a tiff holds, coarsen the \
                 resolution or shrink the bbox",
                u32::MAX
            ));
        }
        Ok(Plan {
            crs,
            grid: ExportGrid {
                origin_x: bbox.min_x,
                origin_y: bbox.max_y,
                pixel_size: resolution,
                cols: cols as usize,
                rows: rows as usize,
            },
            time,
        })
    }
}

pub(crate) fn engine_onto_grid(
    layer_graph: LayerGraph,
    plan: &Plan,
    budget_bytes: usize,
) -> Result<(Engine, NodeId), String> {
    let LayerGraph {
        mut graph,
        node,
        chain_crs,
        ..
    } = layer_graph;
    let node = if plan.crs == chain_crs {
        node
    } else {
        let onto_grid = Reproject::onto_grid(
            plan.crs,
            plan.grid.origin_x,
            plan.grid.origin_y,
            plan.grid.pixel_size,
        );
        graph.add_transform(node, Box::new(onto_grid))
    };
    let engine = Engine::new(graph, budget_bytes).map_err(|e| e.to_string())?;
    Ok((engine, node))
}

pub(crate) async fn cog(
    State(state): State<Arc<ServerState>>,
    UrlPath(name): UrlPath<String>,
    Query(query): Query<ExportQuery>,
) -> Response {
    let Some(index) = state
        .layers
        .iter()
        .position(|layer| layer.info.name == name)
    else {
        return (StatusCode::NOT_FOUND, format!("unknown layer {name}")).into_response();
    };
    let Some(export_dir) = state.export_dir.clone() else {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            "an export writes its file under GEOPLUMB_EXPORT_DIR, which is unset".to_string(),
        )
            .into_response();
    };
    let plan = match plan(&state.layers[index], &query) {
        Ok(plan) => plan,
        Err(reason) => return bad_request(reason),
    };
    let Ok(slot) = state.exports.clone().try_acquire_owned() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            format!("all {MAX_CONCURRENT_EXPORTS} export slots are busy, retry later"),
        )
            .into_response();
    };

    let exporting = async move {
        let _slot = slot;
        let internal = |reason: String| (StatusCode::INTERNAL_SERVER_ERROR, reason).into_response();
        let built = tokio::task::spawn_blocking(move || {
            let layer = &state.layers[index];
            let layer_graph = build_layer_graph(&layer.config)?;
            let (engine, node) = engine_onto_grid(layer_graph, &plan, layer.budget_bytes)?;
            let file = tempfile::tempfile_in(&export_dir).map_err(|e| {
                format!(
                    "no temp file for the export in {}: {e}",
                    export_dir.display()
                )
            })?;
            Ok::<_, String>((engine, node, file))
        })
        .await
        .map_err(|_| internal("the export engine build did not finish".to_string()))?;
        let (engine, node, file) = built.map_err(internal)?;
        let file = write_cog(Arc::new(engine), node, plan.grid, plan.time, file)
            .await
            .map_err(|e| engine_failure(&e))?;
        Ok::<_, Response>(file)
    };
    let file = match tokio::time::timeout(EXPORT_TIMEOUT, exporting).await {
        Err(_) => {
            return (
                StatusCode::GATEWAY_TIMEOUT,
                format!(
                    "the export outran the {} second limit and its file is dropped",
                    EXPORT_TIMEOUT.as_secs()
                ),
            )
                .into_response();
        }
        Ok(Err(failure)) => return failure,
        Ok(Ok(file)) => file,
    };

    let mut file = tokio::fs::File::from_std(file);
    if let Err(error) = file.seek(SeekFrom::Start(0)).await {
        return (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response();
    }
    let chunks = futures::stream::unfold(file, |mut file| async move {
        let mut buffer = vec![0u8; STREAM_CHUNK_BYTES];
        match file.read(&mut buffer).await {
            Ok(0) => None,
            Ok(read) => {
                buffer.truncate(read);
                Some((Ok::<_, std::io::Error>(Bytes::from(buffer)), file))
            }
            Err(error) => Some((Err(error), file)),
        }
    });
    (
        [
            (header::CONTENT_TYPE, "image/tiff".to_string()),
            (
                header::CONTENT_DISPOSITION,
                format!("attachment; filename=\"{name}.tif\""),
            ),
        ],
        Body::from_stream(chunks),
    )
        .into_response()
}

fn plan(layer: &Layer, query: &ExportQuery) -> Result<Plan, String> {
    let Caps::Raster(caps) = layer.engine.caps(layer.node) else {
        return Err(format!(
            "layer {} is not a raster layer, and an export writes rasters",
            layer.info.name
        ));
    };
    let plan = Plan::from_query(query, layer.chain_crs)?;
    let (cols, rows) = (plan.grid.cols as f64, plan.grid.rows as f64);
    let tile = COG_TILE_PX as f64;
    // two 32 bit sides and a band count overflow a usize product
    let values = (cols / tile).ceil() * tile * (rows / tile).ceil() * tile * f64::from(caps.bands);
    if values > MAX_EXPORT_VALUES {
        return Err(format!(
            "{cols} x {rows} pixels in {} bands, padded to whole {COG_TILE_PX} px tiles, is past \
             the {} value export cap, coarsen the resolution or shrink the bbox",
            caps.bands, MAX_EXPORT_VALUES as u64
        ));
    }
    Ok(plan)
}

pub(crate) fn parse_bbox(text: &str) -> Result<Bbox, String> {
    let not_four_numbers = || format!("bbox {text:?} is not four numbers minx,miny,maxx,maxy");
    let numbers = text
        .split(',')
        .map(|part| part.trim().parse::<f64>())
        .collect::<Result<Vec<f64>, _>>()
        .map_err(|_| not_four_numbers())?;
    let &[min_x, min_y, max_x, max_y] = numbers.as_slice() else {
        return Err(not_four_numbers());
    };
    if numbers.iter().any(|number| !number.is_finite()) {
        return Err(format!("bbox {text:?} holds a number that is not finite"));
    }
    if min_x >= max_x || min_y >= max_y {
        return Err(format!(
            "bbox {text:?} has no area, each min has to be below its max"
        ));
    }
    Ok(Bbox::new(min_x, min_y, max_x, max_y))
}

fn parse_crs(text: &str, layer_crs: Crs) -> Result<Crs, String> {
    let crs = parse_epsg(text)?;
    Transform::new(&layer_crs.authority(), &crs.authority())
        .map_err(|error| format!("crs {crs} is unknown to projicio: {error}"))?;
    Ok(crs)
}
