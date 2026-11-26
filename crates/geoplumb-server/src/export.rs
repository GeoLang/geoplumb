use std::io::SeekFrom;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Path as UrlPath, Query, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use geoplumb::elements::Reproject;
use geoplumb::export::{COG_TILE_PX, ExportGrid, write_cog};
use geoplumb::{Bbox, Caps, Crs, Engine, TimeInterval};
use projicio_core::Transform;
use serde::Deserialize;
use tokio::io::{AsyncReadExt, AsyncSeekExt};

use crate::{Layer, LayerGraph, ServerState, bad_request, build_layer_graph, parse_time};

pub(crate) const MAX_CONCURRENT_EXPORTS: usize = 2;

// proxies in front of the service typically cut a request at five minutes
const EXPORT_TIMEOUT: Duration = Duration::from_secs(240);

// 16384 squared f64 values is 2 GiB before deflate, and classic tiff offsets stop at 4 GiB
const MAX_EXPORT_VALUES: f64 = 16384.0 * 16384.0;

const STREAM_CHUNK_BYTES: usize = 64 << 10;

#[derive(Deserialize)]
pub(crate) struct ExportQuery {
    bbox: String,
    crs: String,
    resolution: f64,
    t: Option<String>,
}

#[derive(Clone, Copy)]
struct Plan {
    crs: Crs,
    grid: ExportGrid,
    time: Option<TimeInterval>,
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

    let exporting = tokio::spawn(async move {
        let _slot = slot;
        let built = tokio::task::spawn_blocking(move || {
            let layer = &state.layers[index];
            let LayerGraph {
                mut graph, node, ..
            } = build_layer_graph(&layer.config)?;
            let node = if plan.crs == layer.chain_crs {
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
            let engine = Engine::new(graph, layer.budget_bytes).map_err(|e| e.to_string())?;
            let file =
                tempfile::tempfile().map_err(|e| format!("no temp file for the export: {e}"))?;
            Ok::<_, String>((engine, node, file))
        })
        .await
        .map_err(|_| "the export engine build did not finish".to_string())?;
        let (engine, node, file) = built?;
        let file = write_cog(Arc::new(engine), node, plan.grid, plan.time, file)
            .await
            .map_err(|e| e.to_string())?;
        Ok::<_, String>(file)
    });
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
        Ok(Err(_)) => {
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                "the export did not finish".to_string(),
            )
                .into_response();
        }
        Ok(Ok(Err(reason))) => {
            return (StatusCode::INTERNAL_SERVER_ERROR, reason).into_response();
        }
        Ok(Ok(Ok(file))) => file,
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
    let time = parse_time(query.t.as_deref())?;
    let bbox = parse_bbox(&query.bbox)?;
    let resolution = query.resolution;
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(format!(
            "resolution {resolution} is not a positive number of crs units per pixel"
        ));
    }
    let crs = parse_crs(&query.crs, layer.chain_crs)?;

    // gdalwarp's -te with -tr rounds the pixel count and keeps at least one
    let cols = (bbox.width() / resolution).round().max(1.0);
    let rows = (bbox.height() / resolution).round().max(1.0);
    let tile = COG_TILE_PX as f64;
    // compared as floats so an absurd resolution fails here instead of overflowing the casts below
    let values = (cols / tile).ceil() * tile * (rows / tile).ceil() * tile * f64::from(caps.bands);
    if !values.is_finite() || values > MAX_EXPORT_VALUES {
        return Err(format!(
            "{cols} x {rows} pixels in {} bands, padded to whole {COG_TILE_PX} px tiles, is past \
             the {} value export cap, coarsen the resolution or shrink the bbox",
            caps.bands, MAX_EXPORT_VALUES as u64
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

fn parse_bbox(text: &str) -> Result<Bbox, String> {
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
    let crs = text
        .strip_prefix("EPSG:")
        .and_then(|code| code.parse::<u32>().ok())
        .map(Crs)
        .ok_or_else(|| format!("crs {text:?} is not EPSG:<code>"))?;
    Transform::new(&layer_crs.authority(), &crs.authority())
        .map_err(|error| format!("crs {crs} is unknown to projicio: {error}"))?;
    Ok(crs)
}
