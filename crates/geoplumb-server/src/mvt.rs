use std::collections::BTreeMap;

use geoplumb::tile::XyzTile;
use geoplumb::{TimeInterval, VectorFeature, WindowReq};
use jung_mvt::{
    DEFAULT_EXTENT, MvtFeature, MvtGeometry, MvtLayer, MvtPolygon, MvtValue, encode_tile,
};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{Coord, Polygon};

use crate::Layer;

// mvt renderers clip strokes at the tile edge unless features reach past it
const TILE_BUFFER: u32 = 64;

pub(crate) async fn vector_tile(
    layer: &Layer,
    tile: XyzTile,
    time: Option<TimeInterval>,
) -> geoplumb::Result<Vec<u8>> {
    let bbox = tile.bbox();
    let buffered = bbox.expand(bbox.width() * f64::from(TILE_BUFFER) / f64::from(DEFAULT_EXTENT));
    let pulled = layer
        .engine
        .pull(
            layer.node,
            WindowReq {
                bbox: buffered,
                resolution: tile.resolution(),
                time,
            },
        )
        .await?;
    let dissolved = pulled.into_vector()?.crop_to(&buffered).dissolve();
    let mvt_layer = MvtLayer {
        name: layer.info.name.clone(),
        extent: DEFAULT_EXTENT,
        features: dissolved.features.iter().flat_map(mvt_features).collect(),
    };
    Ok(encode_tile(
        &[mvt_layer],
        [bbox.min_x, bbox.min_y, bbox.max_x, bbox.max_y],
    ))
}

#[derive(Default)]
struct MvtParts {
    points: Vec<[f64; 2]>,
    lines: Vec<Vec<[f64; 2]>>,
    polygons: Vec<MvtPolygon>,
}

fn mvt_features(feature: &VectorFeature) -> Vec<MvtFeature> {
    let mut parts = MvtParts::default();
    collect_parts(&feature.geometry, &mut parts);
    let properties: BTreeMap<String, MvtValue> = feature
        .properties
        .iter()
        .filter_map(|(key, value)| Some((key.clone(), mvt_value(value)?)))
        .collect();
    // mvt has no geometry collection type
    [
        (!parts.points.is_empty()).then_some(MvtGeometry::Point(parts.points)),
        (!parts.lines.is_empty()).then_some(MvtGeometry::LineString(parts.lines)),
        (!parts.polygons.is_empty()).then_some(MvtGeometry::Polygon(parts.polygons)),
    ]
    .into_iter()
    .flatten()
    .map(|geometry| MvtFeature {
        id: Some(feature.id),
        geometry,
        properties: properties.clone(),
    })
    .collect()
}

fn collect_parts(geometry: &FeatureGeometry, parts: &mut MvtParts) {
    match geometry {
        FeatureGeometry::Point(point) => parts.points.push([point.0.x, point.0.y]),
        FeatureGeometry::MultiPoint(multi) => parts
            .points
            .extend(multi.points().iter().map(|point| [point.0.x, point.0.y])),
        FeatureGeometry::LineString(line) => parts.lines.push(path(line.coords())),
        FeatureGeometry::MultiLineString(multi) => parts
            .lines
            .extend(multi.linestrings().iter().map(|line| path(line.coords()))),
        FeatureGeometry::Polygon(polygon) => parts.polygons.push(mvt_polygon(polygon)),
        FeatureGeometry::MultiPolygon(multi) => parts
            .polygons
            .extend(multi.polygons().iter().map(mvt_polygon)),
        FeatureGeometry::GeometryCollection(members) => {
            for member in members {
                collect_parts(member, parts);
            }
        }
    }
}

fn path(coords: &[Coord]) -> Vec<[f64; 2]> {
    coords.iter().map(|coord| [coord.x, coord.y]).collect()
}

fn mvt_polygon(polygon: &Polygon) -> MvtPolygon {
    MvtPolygon {
        exterior: path(polygon.exterior().coords()),
        holes: polygon
            .interiors()
            .iter()
            .map(|ring| path(ring.coords()))
            .collect(),
    }
}

fn mvt_value(value: &serde_json::Value) -> Option<MvtValue> {
    match value {
        serde_json::Value::String(text) => Some(MvtValue::String(text.clone())),
        serde_json::Value::Bool(flag) => Some(MvtValue::Bool(*flag)),
        serde_json::Value::Number(number) => number
            .as_i64()
            .map(MvtValue::Int)
            .or_else(|| number.as_u64().map(MvtValue::UInt))
            .or_else(|| number.as_f64().map(MvtValue::Double)),
        // mvt has no null value
        serde_json::Value::Null => None,
        // mvt values are scalars, tippecanoe writes arrays and objects as json text
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => {
            Some(MvtValue::String(value.to_string()))
        }
    }
}
