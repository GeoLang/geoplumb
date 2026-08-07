//! zonal statistics and per-step time series over a layer, the reduction
//! side of the same engines the tiles come from.
//!
//! a request names its own window, so every number in it is a cap to check
//! before an engine is touched: nothing here is clamped to fit, a request
//! past a cap is refused. the endpoints are as public as the tiles are.
//!
//! zone coordinates and the bbox are lon/lat degrees, what rfc 7946 says
//! geojson carries, and the server projects them onto the layer's own grid,
//! web mercator, the crs every layer graph ends in.

use std::collections::HashMap;

use axum::body::Bytes;
use axum::extract::{Path as UrlPath, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use projicio_core::Transform;
use serde::{Deserialize, Serialize};

use geoplumb::window::GridSpec;
use geoplumb::{
    Bbox, Crs, FeatureStatistics, PixelStatistics, TimeInterval, VectorFeature, WindowReq,
    window_statistics, window_time_series, zonal_statistics, zonal_time_series,
};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{Coord, Envelope, MultiPolygon, Polygon, Ring};

use crate::{Layer, Layers, bad_request, parse_time};

/// pixels one request may reduce, summed over its steps. 4096 squared f64
/// pixels is 128 MiB a band, the budget tiletopia's export path caps at
const MAX_REQUEST_PIXELS: f64 = 4096.0 * 4096.0;

/// zones one request may name
const MAX_FEATURES: usize = 256;

/// intervals one series may ask for
const MAX_STEPS: usize = 64;

/// ring positions across every zone. the polygon burn walks each edge once
/// per raster row, so the pixel budget on its own does not bound the work
const MAX_POSITIONS: usize = 20_000;

const LONGITUDE_LIMIT: f64 = 180.0;
const LATITUDE_LIMIT: f64 = 90.0;

/// web mercator diverges at the poles, so a coordinate reaching this
/// latitude is refused: clamping it would answer about a place nobody asked
/// about
const WEB_MERCATOR_LATITUDE_LIMIT: f64 = 85.06;

/// a `FeatureCollection` carrying the window to reduce over
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ZonalRequest {
    #[serde(rename = "type")]
    collection_type: Option<String>,
    #[serde(default)]
    features: Vec<serde_json::Value>,
    /// the interval the tile endpoint takes, absent leaving the layer's own
    t: Option<String>,
    /// min lon, min lat, max lon, max lat in degrees, the layer file's own
    /// bbox order. absent it is the zones' envelope
    bbox: Option<[f64; 4]>,
    /// web mercator ground metres, the crs the layer graph ends in
    resolution: f64,
}

/// the same body with a step per pull. it takes no `t`: every step names
/// its own interval, so one here would be silently overridden
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SeriesRequest {
    #[serde(rename = "type")]
    collection_type: Option<String>,
    #[serde(default)]
    features: Vec<serde_json::Value>,
    /// min lon, min lat, max lon, max lat in degrees
    bbox: Option<[f64; 4]>,
    /// web mercator ground metres
    resolution: f64,
    steps: Vec<String>,
}

/// one zone's reduction, `id` its index in the request's feature order and
/// null for a whole-window row. a float is null wherever it is not finite:
/// an empty zone has no mean, minimum or maximum, and json has no nan
#[derive(Serialize)]
struct Row {
    id: Option<usize>,
    count: usize,
    sum: Option<f64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
    mean: Option<f64>,
}

#[derive(Serialize)]
struct ZonalResponse {
    rows: Vec<Row>,
}

#[derive(Serialize)]
struct SeriesStep {
    t: String,
    rows: Vec<Row>,
}

#[derive(Serialize)]
struct SeriesResponse {
    steps: Vec<SeriesStep>,
}

impl Row {
    fn new(id: Option<usize>, statistics: PixelStatistics) -> Row {
        Row {
            id,
            count: statistics.count,
            sum: finite(statistics.sum),
            minimum: finite(statistics.minimum),
            maximum: finite(statistics.maximum),
            mean: finite(statistics.mean()),
        }
    }
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

/// one row per requested zone in request order. a zone that caught no valid
/// pixel is absent from the driver's rows and lands here as an empty
/// reduction, so a client can line the rows up with what it sent
fn zone_rows(found: &[FeatureStatistics], zones: usize) -> Vec<Row> {
    (0..zones)
        .map(|index| {
            let statistics = found
                .iter()
                .find(|row| row.feature_id == index as u64)
                .map_or_else(PixelStatistics::default, |row| row.statistics);
            Row::new(Some(index), statistics)
        })
        .collect()
}

pub async fn statistics(
    State(layers): State<Layers>,
    UrlPath(name): UrlPath<String>,
    body: Bytes,
) -> Response {
    let Some(index) = layers.iter().position(|layer| layer.info.name == name) else {
        return (StatusCode::NOT_FOUND, format!("unknown layer {name}")).into_response();
    };
    let request: ZonalRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return bad_request(format!("bad request body: {error}")),
    };
    if let Err(reason) = check_collection_type(request.collection_type.as_deref()) {
        return bad_request(reason);
    }
    let time = match parse_time(request.t.as_deref()) {
        Ok(time) => time,
        Err(reason) => return bad_request(reason),
    };
    let plan = match plan(
        &layers[index],
        &request.features,
        request.bbox,
        request.resolution,
        time,
        1,
    ) {
        Ok(plan) => plan,
        Err(reason) => return bad_request(reason),
    };

    let zones = plan.zones.len();
    let handle = tokio::runtime::Handle::current();
    let reduced = offload(move || {
        let layer = &layers[index];
        handle.block_on(async {
            match plan.zones.is_empty() {
                true => window_statistics(&layer.engine, layer.node, plan.window)
                    .await
                    .map(|statistics| vec![Row::new(None, statistics)]),
                false => zonal_statistics(&layer.engine, layer.node, &plan.zones, plan.window)
                    .await
                    .map(|found| zone_rows(&found, zones)),
            }
        })
    })
    .await;
    match reduced {
        Ok(rows) => axum::Json(ZonalResponse { rows }).into_response(),
        Err(reason) => (StatusCode::INTERNAL_SERVER_ERROR, reason).into_response(),
    }
}

pub async fn series(
    State(layers): State<Layers>,
    UrlPath(name): UrlPath<String>,
    body: Bytes,
) -> Response {
    let Some(index) = layers.iter().position(|layer| layer.info.name == name) else {
        return (StatusCode::NOT_FOUND, format!("unknown layer {name}")).into_response();
    };
    let request: SeriesRequest = match serde_json::from_slice(&body) {
        Ok(request) => request,
        Err(error) => return bad_request(format!("bad request body: {error}")),
    };
    if let Err(reason) = check_collection_type(request.collection_type.as_deref()) {
        return bad_request(reason);
    }
    let steps = match parse_steps(&request.steps) {
        Ok(steps) => steps,
        Err(reason) => return bad_request(reason),
    };
    let plan = match plan(
        &layers[index],
        &request.features,
        request.bbox,
        request.resolution,
        None,
        steps.len(),
    ) {
        Ok(plan) => plan,
        Err(reason) => return bad_request(reason),
    };

    let zones = plan.zones.len();
    let handle = tokio::runtime::Handle::current();
    let reduced = offload(move || {
        let layer = &layers[index];
        handle.block_on(async {
            match plan.zones.is_empty() {
                true => window_time_series(&layer.engine, layer.node, plan.window, &steps)
                    .await
                    .map(|series| {
                        series
                            .into_iter()
                            .map(|(_, statistics)| vec![Row::new(None, statistics)])
                            .collect::<Vec<_>>()
                    }),
                false => {
                    zonal_time_series(&layer.engine, layer.node, &plan.zones, plan.window, &steps)
                        .await
                        .map(|series| {
                            series
                                .into_iter()
                                .map(|(_, found)| zone_rows(&found, zones))
                                .collect::<Vec<_>>()
                        })
                }
            }
        })
    })
    .await;
    match reduced {
        Ok(per_step) => axum::Json(SeriesResponse {
            steps: per_step
                .into_iter()
                .zip(request.steps)
                .map(|(rows, t)| SeriesStep { t, rows })
                .collect(),
        })
        .into_response(),
        Err(reason) => (StatusCode::INTERNAL_SERVER_ERROR, reason).into_response(),
    }
}

/// run a driver off the async workers. the engine already sends each node's
/// compute to the blocking pool, but the per-tile burn and reduce the zonal
/// drivers add run inline, and a zonal window is many tiles of it: left on
/// a worker thread that starves the tile path. a driver that panics comes
/// back as a join error rather than taking the service down
async fn offload<T: Send + 'static>(
    work: impl FnOnce() -> geoplumb::Result<T> + Send + 'static,
) -> Result<T, String> {
    tokio::task::spawn_blocking(work)
        .await
        .map_err(|_| "the reduction did not finish".to_string())?
        .map_err(|error| error.to_string())
}

/// the zones and the window a request asks for, every cap already checked
struct Plan {
    /// empty for a whole-window reduction
    zones: Vec<VectorFeature>,
    window: WindowReq,
}

fn plan(
    layer: &Layer,
    features: &[serde_json::Value],
    bbox: Option<[f64; 4]>,
    resolution: f64,
    time: Option<TimeInterval>,
    steps: usize,
) -> Result<Plan, String> {
    if features.len() > MAX_FEATURES {
        return Err(format!(
            "{} features is past the {MAX_FEATURES} cap",
            features.len()
        ));
    }
    if !resolution.is_finite() || resolution <= 0.0 {
        return Err(format!("resolution {resolution} is not a positive number"));
    }
    let transform = Transform::new(&Crs::WGS84.authority(), &Crs::WEB_MERCATOR.authority())
        .map_err(|error| format!("no projection onto the layer grid: {error}"))?;
    let zones = parse_zones(features, &transform)?;
    let window = match bbox {
        Some(corners) => window_bbox(corners, &transform)?,
        None => zones_envelope(&zones)?,
    };
    check_pixel_budget(layer.engine.grid(layer.node), &window, resolution, steps)?;
    Ok(Plan {
        zones,
        window: WindowReq {
            bbox: window,
            resolution,
            time,
        },
    })
}

fn check_collection_type(collection_type: Option<&str>) -> Result<(), String> {
    match collection_type {
        None | Some("FeatureCollection") => Ok(()),
        Some(other) => Err(format!("{other} is not a FeatureCollection")),
    }
}

fn parse_steps(steps: &[String]) -> Result<Vec<TimeInterval>, String> {
    if steps.is_empty() {
        return Err("a series needs at least one step".to_string());
    }
    if steps.len() > MAX_STEPS {
        return Err(format!("{} steps is past the {MAX_STEPS} cap", steps.len()));
    }
    let mut intervals = Vec::with_capacity(steps.len());
    for (index, text) in steps.iter().enumerate() {
        match parse_time(Some(text)) {
            Ok(Some(interval)) => intervals.push(interval),
            Ok(None) => return Err(format!("step {index} names no interval")),
            Err(reason) => return Err(format!("step {index}: {reason}")),
        }
    }
    Ok(intervals)
}

/// the request's window, lon/lat like the layer file's bbox, onto the
/// layer's grid. web mercator x follows longitude and y follows latitude,
/// both rising, so the two corners carry the whole window
fn window_bbox(corners: [f64; 4], transform: &Transform) -> Result<Bbox, String> {
    let [min_lon, min_lat, max_lon, max_lat] = corners;
    check_degrees(min_lon, min_lat)?;
    check_degrees(max_lon, max_lat)?;
    if max_lon <= min_lon || max_lat <= min_lat {
        return Err("the bbox has no area".to_string());
    }
    let (min_x, min_y) = project(transform, min_lon, min_lat)?;
    let (max_x, max_y) = project(transform, max_lon, max_lat)?;
    Ok(Bbox::new(min_x, min_y, max_x, max_y))
}

/// geojson positions are lon/lat degrees, so anything outside that range is
/// a client naming other units rather than a place on earth
fn check_degrees(longitude: f64, latitude: f64) -> Result<(), String> {
    if !longitude.is_finite() || longitude.abs() > LONGITUDE_LIMIT {
        return Err(format!("longitude {longitude} is not a longitude"));
    }
    if !latitude.is_finite() || latitude.abs() > LATITUDE_LIMIT {
        return Err(format!("latitude {latitude} is not a latitude"));
    }
    if latitude.abs() >= WEB_MERCATOR_LATITUDE_LIMIT {
        return Err(format!(
            "latitude {latitude} is past the {WEB_MERCATOR_LATITUDE_LIMIT} web mercator limit"
        ));
    }
    Ok(())
}

fn project(transform: &Transform, longitude: f64, latitude: f64) -> Result<(f64, f64), String> {
    let onto_grid = |error| format!("lon {longitude} lat {latitude} does not project: {error}");
    let (x, y) = transform
        .convert(longitude, latitude)
        .map_err(|error| onto_grid(error.to_string()))?;
    if !x.is_finite() || !y.is_finite() {
        return Err(onto_grid("it leaves the grid".to_string()));
    }
    Ok((x, y))
}

/// the window a zone set covers, already on the layer's grid, the default
/// when a request names no bbox. a zero-area envelope is refused rather than
/// pulled: it covers no whole pixel, so the reduction would answer an empty
/// result to a real question
fn zones_envelope(zones: &[VectorFeature]) -> Result<Bbox, String> {
    if zones.is_empty() {
        return Err("a request with no features needs a bbox".to_string());
    }
    let mut envelope: Option<Envelope> = None;
    for zone in zones {
        for ring in rings_of(&zone.geometry) {
            let Some(next) = Envelope::from_coords(ring.coords()) else {
                continue;
            };
            envelope = Some(match envelope {
                None => next,
                Some(previous) => previous.union(&next),
            });
        }
    }
    let envelope = envelope.ok_or_else(|| "the features hold no coordinates".to_string())?;
    if envelope.width() <= 0.0 || envelope.height() <= 0.0 {
        return Err("the features envelope has no area".to_string());
    }
    Ok(Bbox::new(
        envelope.min_x,
        envelope.min_y,
        envelope.max_x,
        envelope.max_y,
    ))
}

fn rings_of(geometry: &FeatureGeometry) -> Vec<&Ring> {
    match geometry {
        FeatureGeometry::Polygon(polygon) => polygon_rings(polygon),
        FeatureGeometry::MultiPolygon(multi) => {
            multi.polygons().iter().flat_map(polygon_rings).collect()
        }
        _ => Vec::new(),
    }
}

fn polygon_rings(polygon: &Polygon) -> Vec<&Ring> {
    std::iter::once(polygon.exterior())
        .chain(polygon.interiors())
        .collect()
}

/// the pixels the pulls will really reduce. the engine snaps a request onto
/// its node's resolution ladder, and the level it lands on can be finer
/// than the one asked for, so the budget has to be read off that one
fn check_pixel_budget(
    grid: &GridSpec,
    bbox: &Bbox,
    resolution: f64,
    steps: usize,
) -> Result<(), String> {
    let snapped = grid.resolution_at(grid.snap_level(resolution));
    let columns = (bbox.width() / snapped).ceil();
    let rows = (bbox.height() / snapped).ceil();
    let pixels = columns * rows * steps as f64;
    if !pixels.is_finite() || pixels > MAX_REQUEST_PIXELS {
        return Err(format!(
            "the request covers {pixels:.0} pixels at {snapped} m, past the {MAX_REQUEST_PIXELS:.0} cap"
        ));
    }
    Ok(())
}

/// the request's features as zones on the layer's grid, ided by their place
/// in the request. properties are dropped: nothing downstream reads them and
/// the id is positional, so keeping arbitrary client json would only cost
/// memory
fn parse_zones(
    features: &[serde_json::Value],
    transform: &Transform,
) -> Result<Vec<VectorFeature>, String> {
    let mut zones = Vec::with_capacity(features.len());
    let mut reader = Zones {
        transform,
        remaining: MAX_POSITIONS,
    };
    for (index, feature) in features.iter().enumerate() {
        let named = feature.get("type").and_then(serde_json::Value::as_str);
        if named != Some("Feature") {
            return Err(format!("feature {index} is not a Feature"));
        }
        let geometry = feature
            .get("geometry")
            .filter(|geometry| !geometry.is_null())
            .ok_or_else(|| format!("feature {index} has no geometry"))?;
        let geometry = reader
            .geometry(geometry)
            .map_err(|reason| format!("feature {index}: {reason}"))?;
        zones.push(VectorFeature {
            id: index as u64,
            geometry,
            properties: HashMap::new(),
        });
    }
    Ok(zones)
}

/// reads geojson geometry onto the layer's grid: the projection it lands
/// through, and how many positions the whole request has left to spend
struct Zones<'a> {
    transform: &'a Transform,
    remaining: usize,
}

impl Zones<'_> {
    /// only areal geometry: a zone is an area, and the tiled merge the
    /// drivers do is exact for the polygon burn, which fills by scanline
    /// inside the tile it is given
    fn geometry(&mut self, geometry: &serde_json::Value) -> Result<FeatureGeometry, String> {
        let kind = geometry
            .get("type")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| "the geometry names no type".to_string())?;
        let coordinates = geometry
            .get("coordinates")
            .ok_or_else(|| format!("the {kind} has no coordinates"))?;
        match kind {
            "Polygon" => Ok(FeatureGeometry::Polygon(self.polygon(coordinates)?)),
            "MultiPolygon" => {
                let parts = coordinates
                    .as_array()
                    .ok_or_else(|| "the MultiPolygon coordinates are not an array".to_string())?;
                if parts.is_empty() {
                    return Err("the MultiPolygon holds no polygon".to_string());
                }
                let polygons = parts
                    .iter()
                    .map(|part| self.polygon(part))
                    .collect::<Result<Vec<Polygon>, String>>()?;
                Ok(FeatureGeometry::MultiPolygon(MultiPolygon::new(polygons)))
            }
            other => Err(format!("a zone cannot be a {other}")),
        }
    }

    fn polygon(&mut self, coordinates: &serde_json::Value) -> Result<Polygon, String> {
        let rings = coordinates
            .as_array()
            .ok_or_else(|| "the polygon coordinates are not an array".to_string())?;
        let (exterior, holes) = rings
            .split_first()
            .ok_or_else(|| "the polygon holds no ring".to_string())?;
        let exterior = self.ring(exterior)?;
        let holes = holes
            .iter()
            .map(|hole| self.ring(hole))
            .collect::<Result<Vec<Ring>, String>>()?;
        Ok(Polygon::new(exterior, holes))
    }

    /// a linear ring, at the four position minimum geojson sets. the budget
    /// is spent before the positions are read, so an oversized ring is
    /// refused without being built
    fn ring(&mut self, ring: &serde_json::Value) -> Result<Ring, String> {
        let positions = ring
            .as_array()
            .ok_or_else(|| "a ring is not an array".to_string())?;
        if positions.len() < 4 {
            return Err(format!(
                "a ring of {} positions cannot close",
                positions.len()
            ));
        }
        if positions.len() > self.remaining {
            return Err(format!(
                "the features hold more than the {MAX_POSITIONS} position cap"
            ));
        }
        self.remaining -= positions.len();
        let coords = positions
            .iter()
            .map(|position| self.position(position))
            .collect::<Result<Vec<Coord>, String>>()?;
        Ok(Ring::new(coords))
    }

    fn position(&self, position: &serde_json::Value) -> Result<Coord, String> {
        let numbers = position
            .as_array()
            .ok_or_else(|| "a position is not an array".to_string())?;
        let at = |axis: usize| {
            numbers
                .get(axis)
                .and_then(serde_json::Value::as_f64)
                .filter(|value| value.is_finite())
        };
        let (Some(longitude), Some(latitude)) = (at(0), at(1)) else {
            return Err("a position needs two finite numbers".to_string());
        };
        check_degrees(longitude, latitude)?;
        let (x, y) = project(self.transform, longitude, latitude)?;
        Ok(Coord::new(x, y))
    }
}
