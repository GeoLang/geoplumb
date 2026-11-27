use std::io::Write;
use std::path::Path;
use std::sync::Arc;

use geoplumb::export::write_cog;
use geoplumb::tile::XyzTile;
use geoplumb::{Caps, Crs, Engine, NodeId, WindowReq};
use tempfile::NamedTempFile;
use terrano_core::CogReader;
use topoi_core::geojson::{Feature, FeatureCollection, write_geojson};

use crate::config::{Config, LayerConfig};
use crate::export::{ExportQuery, Plan, engine_onto_grid, parse_bbox};
use crate::{
    DEFAULT_CACHE_BYTES, Layer, LayerOutput, MAX_ZOOM, TileError, TileFormat, build_layer_graph,
    encode_tile, parse_time,
};

#[derive(Clone, Copy)]
enum OutputFormat {
    Cog,
    Geojson,
}

fn output_format(output: &Path) -> Result<OutputFormat, String> {
    let extension = output
        .extension()
        .and_then(|extension| extension.to_str())
        .map(str::to_ascii_lowercase);
    match extension.as_deref() {
        Some("tif" | "tiff") => Ok(OutputFormat::Cog),
        Some("geojson") => Ok(OutputFormat::Geojson),
        _ => Err(format!(
            "{}: the output is a .tif for a raster layer or a .geojson for a vector layer",
            output.display()
        )),
    }
}

struct LayerJob {
    engine: Engine,
    node: NodeId,
    plan: Plan,
}

async fn features(job: &LayerJob) -> Result<FeatureCollection, String> {
    let bbox = job.plan.grid.bbox();
    let request = WindowReq {
        bbox,
        resolution: job.plan.grid.pixel_size,
        time: job.plan.time,
    };
    let pulled = job
        .engine
        .pull(job.node, request)
        .await
        .and_then(|chunk| chunk.into_vector())
        .map_err(|e| e.to_string())?;
    let dissolved = pulled.crop_to(&bbox).dissolve();
    Ok(FeatureCollection {
        features: dissolved
            .features
            .into_iter()
            .map(|feature| Feature {
                geometry: Some(feature.geometry),
                properties: feature.properties,
            })
            .collect(),
    })
}

fn layer_config(layers: &Path, layer: &str) -> Result<LayerConfig, String> {
    let in_layer_file = |message: String| format!("{}: {message}", layers.display());
    let text = std::fs::read_to_string(layers).map_err(|e| in_layer_file(e.to_string()))?;
    let config = Config::parse(&text).map_err(in_layer_file)?;
    let names: Vec<String> = config.layers.iter().map(|l| l.name.clone()).collect();
    config
        .layers
        .into_iter()
        .find(|l| l.name == layer)
        .ok_or_else(|| {
            in_layer_file(format!(
                "no layer is named {layer}, the file names {}",
                names.join(", ")
            ))
        })
}

pub async fn write_layer(
    layers: &Path,
    layer: &str,
    query: ExportQuery,
    output: &Path,
) -> Result<(), String> {
    let format = output_format(output)?;
    let layer_config = layer_config(layers, layer)?;

    // opening sources blocks: a stac layer searches its anchor bbox and a cog layer reads a file header
    let job = tokio::task::spawn_blocking(move || {
        let layer_graph = build_layer_graph(&layer_config)
            .map_err(|e| format!("layer {}: {e}", layer_config.name))?;
        let plan = Plan::from_query(&query, layer_graph.chain_crs)?;
        let (engine, node) = engine_onto_grid(layer_graph, &plan, DEFAULT_CACHE_BYTES)?;
        Ok::<_, String>(LayerJob { engine, node, plan })
    })
    .await
    .map_err(|e| e.to_string())??;

    let layer_format = match job.engine.caps(job.node) {
        Caps::Raster(_) => OutputFormat::Cog,
        Caps::Vector(_) => OutputFormat::Geojson,
        Caps::PointCloud(_) | Caps::Tensor(_) => {
            return Err(format!(
                "layer {layer} ends in neither a raster nor features"
            ));
        }
    };
    let directory = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let mut file = NamedTempFile::new_in(directory).map_err(|e| {
        format!(
            "no temp file for the output in {}: {e}",
            directory.display()
        )
    })?;
    match (layer_format, format) {
        (OutputFormat::Cog, OutputFormat::Cog) => {
            let LayerJob { engine, node, plan } = job;
            file = write_cog(Arc::new(engine), node, plan.grid, plan.time, file)
                .await
                .map_err(|e| e.to_string())?;
        }
        (OutputFormat::Geojson, OutputFormat::Geojson) => {
            if job.plan.crs != Crs::WGS84 {
                return Err(format!(
                    "geojson coordinates are lon/lat (rfc 7946), so a .geojson output takes \
                     --crs EPSG:4326, not {}",
                    job.plan.crs
                ));
            }
            let collection = features(&job).await?;
            file.write_all(write_geojson(&collection).as_bytes())
                .map_err(|e| format!("{}: {e}", file.path().display()))?;
        }
        (OutputFormat::Cog, OutputFormat::Geojson) => {
            return Err(format!(
                "layer {layer} is a raster layer, its output is a .tif"
            ));
        }
        (OutputFormat::Geojson, OutputFormat::Cog) => {
            return Err(format!(
                "layer {layer} is a vector layer, its output is a .geojson"
            ));
        }
    }
    file.persist(output)
        .map_err(|e| format!("{}: {}", output.display(), e.error))?;
    Ok(())
}

pub struct PyramidQuery {
    pub bbox: String,
    pub min_zoom: u8,
    pub max_zoom: u8,
    pub t: Option<String>,
}

pub async fn write_pyramid(
    layers: &Path,
    layer: &str,
    query: PyramidQuery,
    output: &Path,
) -> Result<usize, String> {
    let lonlat = parse_bbox(&query.bbox)?;
    if query.min_zoom > query.max_zoom || query.max_zoom > MAX_ZOOM {
        return Err(format!(
            "zooms {} to {} are not a range within 0 to {MAX_ZOOM}",
            query.min_zoom, query.max_zoom
        ));
    }
    let time = parse_time(query.t.as_deref())?;
    let layer_config = layer_config(layers, layer)?;
    // opening sources blocks: a stac layer searches its anchor bbox and a cog layer reads a file header
    let served = tokio::task::spawn_blocking(move || {
        let name = layer_config.name.clone();
        Layer::build(layer_config, DEFAULT_CACHE_BYTES, None)
            .map_err(|e| format!("layer {name}: {e}"))
    })
    .await
    .map_err(|e| e.to_string())??;
    let (format, extension) = match served.output {
        LayerOutput::Raster { .. } => (TileFormat::Png, "png"),
        LayerOutput::Vector => (TileFormat::Mvt, "mvt"),
    };
    let mut written = 0;
    for z in query.min_zoom..=query.max_zoom {
        for tile in XyzTile::covering(z, &lonlat) {
            let (_, bytes) =
                encode_tile(&served, tile, time, format)
                    .await
                    .map_err(|e| match e {
                        TileError::WrongFormat(message) => message,
                        TileError::Engine(e) => format!("tile {z}/{}/{}: {e}", tile.x, tile.y),
                    })?;
            let directory = output.join(z.to_string()).join(tile.x.to_string());
            std::fs::create_dir_all(&directory)
                .map_err(|e| format!("{}: {e}", directory.display()))?;
            let path = directory.join(format!("{}.{extension}", tile.y));
            std::fs::write(&path, bytes).map_err(|e| format!("{}: {e}", path.display()))?;
            written += 1;
        }
    }
    Ok(written)
}

pub fn cog_info(path: &Path) -> Result<String, String> {
    let file = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let reader = CogReader::open(file).map_err(|e| format!("{}: {e}", path.display()))?;
    let meta = reader.meta();
    let (base, overviews) = reader
        .levels()
        .split_first()
        .expect("CogReader::open fails on a tiff with no image");
    let format = base.sample_format();
    let crs = match meta.epsg {
        // terrano reads a geotiff without an epsg geokey as code 0
        0 => "none".to_string(),
        code => format!("EPSG:{code}"),
    };
    // tiff compression and predictor tag values
    let compression = match base.compression() {
        1 => "none".to_string(),
        5 => "lzw".to_string(),
        8 => "deflate".to_string(),
        other => format!("tiff code {other}"),
    };
    let predictor = match base.predictor() {
        1 => "none".to_string(),
        2 => "horizontal".to_string(),
        3 => "floating point".to_string(),
        other => format!("tiff code {other}"),
    };
    let nodata = match base.nodata() {
        Some(value) => value.to_string(),
        None => "none".to_string(),
    };
    let mut lines = vec![
        format!("size: {} x {}", base.width, base.height),
        format!("bands: {}", base.samples),
        format!("sample format: {}", format.name()),
        format!("bits: {}", format.bits()),
        format!("crs: {crs}"),
        format!(
            "geotransform: {}, {}, 0, {}, 0, {}",
            meta.origin_x, meta.pixel_width, meta.origin_y, -meta.pixel_height
        ),
        format!("compression: {compression}"),
        format!("predictor: {predictor}"),
        format!("nodata: {nodata}"),
    ];
    if overviews.is_empty() {
        lines.push("overviews: none".to_string());
    } else {
        lines.push("overviews:".to_string());
        lines.extend(
            overviews
                .iter()
                .map(|level| format!("- {} x {}", level.width, level.height)),
        );
    }
    Ok(lines.join("\n") + "\n")
}
