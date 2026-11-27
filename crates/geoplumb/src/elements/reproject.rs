use std::borrow::Cow;

use crate::bands::RasterBands;
use crate::caps::{
    BandKinds, Caps, CapsPattern, CapsSet, Constraint, Crs, FieldMask, RasterPattern, SetField,
    VectorPattern,
};
use crate::chunk::{Chunk, RasterChunk, VectorChunk, VectorFeature, clip_geometry};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::resample::{OutputGrid, resampled_band};
use crate::window::{Bbox, GridSpec, WindowReq};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{
    Coord, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, Ring,
};

pub struct Reproject {
    to: Crs,
    output_anchor: Option<GridAnchor>,
    configured: Option<ConfiguredReprojection>,
}

struct ConfiguredReprojection {
    kind: ReprojectedKind,
    forward: projicio_core::Transform,
    inverse: projicio_core::Transform,
}

enum ReprojectedKind {
    Raster { value_kinds: BandKinds },
    Vector { latitude_band: Option<LatitudeBand> },
}

#[derive(Clone, Copy)]
struct LatitudeBand {
    south: f64,
    north: f64,
}

// epsg 3857's area of use ends where web mercator y reaches the half width of its square
const WEB_MERCATOR_MAX_LATITUDE: f64 = 85.051_128_779_806_59;

fn latitude_band(from: Crs, to: Crs) -> Option<LatitudeBand> {
    let geographic = projicio_core::epsg::proj4_definition(from.0)
        .is_some_and(|definition| definition.contains("+proj=longlat"));
    // projicio's web mercator refuses latitudes near the poles
    (geographic && to == Crs::WEB_MERCATOR).then_some(LatitudeBand {
        south: -WEB_MERCATOR_MAX_LATITUDE,
        north: WEB_MERCATOR_MAX_LATITUDE,
    })
}

fn reproject_constraint(crs: SetField<Crs>) -> Constraint {
    Constraint::AnyOf(vec![
        Constraint::Derived {
            input: CapsSet::any_raster(),
            passthrough: FieldMask::without_crs_resolution(),
            output: CapsPattern::Raster(RasterPattern {
                crs: crs.clone(),
                ..RasterPattern::default()
            }),
        },
        Constraint::Derived {
            input: CapsSet::any_vector(),
            passthrough: FieldMask::without_crs_resolution(),
            output: CapsPattern::Vector(VectorPattern {
                crs,
                ..VectorPattern::default()
            }),
        },
    ])
}

#[derive(Clone, Copy)]
struct GridAnchor {
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
}

impl Reproject {
    pub fn new(to: Crs) -> Self {
        Reproject {
            to,
            output_anchor: None,
            configured: None,
        }
    }

    pub fn onto_grid(to: Crs, origin_x: f64, origin_y: f64, base_resolution: f64) -> Self {
        Reproject {
            output_anchor: Some(GridAnchor {
                origin_x,
                origin_y,
                base_resolution,
            }),
            ..Reproject::new(to)
        }
    }

    /// auto-plug template: like `constraint` but with the target crs left
    /// free, so the solver retargets it to whatever the demanding side needs
    pub fn adapter() -> crate::element::Adapter {
        crate::element::Adapter {
            template: reproject_constraint(SetField::Any),
            build: Box::new(|target| {
                let crs = match target {
                    CapsPattern::Raster(raster) => &raster.crs,
                    CapsPattern::Vector(vector) => &vector.crs,
                    CapsPattern::PointCloud(_) | CapsPattern::Tensor(_) => return None,
                };
                let SetField::OneOf(crss) = crs else {
                    return None;
                };
                Some(Box::new(Reproject::new(crss[0])))
            }),
        }
    }

    fn configured(&self) -> &ConfiguredReprojection {
        self.configured.as_ref().expect("configured")
    }
}

/// envelope of a bbox through a point transform, corners plus edge
/// midpoints so curved edges do not clip
fn envelope(t: &projicio_core::Transform, b: &Bbox) -> Result<Bbox> {
    let xs = [b.min_x, (b.min_x + b.max_x) / 2.0, b.max_x];
    let ys = [b.min_y, (b.min_y + b.max_y) / 2.0, b.max_y];
    let mut pts = Vec::with_capacity(8);
    for x in xs {
        for y in ys {
            if x == xs[1] && y == ys[1] {
                continue;
            }
            pts.push((x, y));
        }
    }
    let out = t
        .convert_batch(&pts)
        .map_err(|e| Error::Projection(e.to_string()))?;
    let mut env = Bbox::new(
        f64::INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NEG_INFINITY,
    );
    for (x, y) in out {
        env.min_x = env.min_x.min(x);
        env.min_y = env.min_y.min(y);
        env.max_x = env.max_x.max(x);
        env.max_y = env.max_y.max(y);
    }
    Ok(env)
}

/// ground distance a `step` at (x, y) maps to under `t`, the local scale
/// factor that rescales a resolution across the projection
pub(crate) fn local_scale(t: &projicio_core::Transform, x: f64, y: f64, step: f64) -> f64 {
    match (t.convert(x, y), t.convert(x + step, y)) {
        (Ok((x0, y0)), Ok((x1, y1))) => (x1 - x0).hypot(y1 - y0).max(1e-12),
        _ => step,
    }
}

/// the source window a warp reads to fill `bbox` at `resolution`: the
/// inverse-projected envelope widened for the bilinear kernel, and the
/// source resolution the local scale factor asks for
pub(crate) fn inverse_window(
    inverse: &projicio_core::Transform,
    bbox: &Bbox,
    resolution: f64,
) -> (Bbox, f64) {
    let cx = (bbox.min_x + bbox.max_x) / 2.0;
    let cy = (bbox.min_y + bbox.max_y) / 2.0;
    let source_resolution = local_scale(inverse, cx, cy, resolution);
    let env = envelope(inverse, bbox).unwrap_or(*bbox);
    (env.expand(2.0 * source_resolution), source_resolution)
}

pub(crate) fn warp_to_grid(
    input: &RasterChunk,
    out: &WindowReq,
    inverse: &projicio_core::Transform,
    to: Crs,
    value_kinds: &BandKinds,
) -> Result<RasterChunk> {
    let res = out.resolution;
    let cols = (out.bbox.width() / res).round() as usize;
    let rows = (out.bbox.height() / res).round() as usize;
    let mut centers = Vec::with_capacity(cols * rows);
    for row in 0..rows {
        for col in 0..cols {
            centers.push((
                out.bbox.min_x + (col as f64 + 0.5) * res,
                out.bbox.max_y - (row as f64 + 0.5) * res,
            ));
        }
    }
    let src_pts = inverse
        .convert_batch(&centers)
        .map_err(|e| Error::Projection(e.to_string()))?;
    let center_x = (out.bbox.min_x + out.bbox.max_x) / 2.0;
    let center_y = (out.bbox.min_y + out.bbox.max_y) / 2.0;
    let grid = OutputGrid {
        cols,
        rows,
        cell_size: res,
        cell_size_in_input_crs: local_scale(inverse, center_x, center_y, res),
    };
    let bands = input
        .bands
        .bands()
        .iter()
        .enumerate()
        .map(|(bi, band)| {
            resampled_band(band, value_kinds.band(bi), input, &grid, |index| {
                src_pts[index]
            })
        })
        .collect();
    Ok(RasterChunk {
        bands: RasterBands::new(bands)?,
        bbox: out.bbox,
        resolution: res,
        crs: to,
    })
}

/// canonical grid anchors for the common target crs, else the projected
/// input origin
fn canonical_origin(crs: Crs, projected_input_origin: (f64, f64)) -> (f64, f64) {
    match crs {
        Crs::WEB_MERCATOR => (-20037508.342789244, 20037508.342789244),
        Crs::WGS84 => (-180.0, 90.0),
        _ => projected_input_origin,
    }
}

impl Transform for Reproject {
    fn identity(&self) -> String {
        match self.output_anchor {
            None => format!("Reproject to={}", self.to),
            Some(anchor) => format!(
                "Reproject to={} grid={},{},{}",
                self.to, anchor.origin_x, anchor.origin_y, anchor.base_resolution
            ),
        }
    }

    fn constraint(&self) -> Constraint {
        reproject_constraint(SetField::one(self.to))
    }

    fn configure(&mut self, input: &Caps, output: &Caps) -> Result<()> {
        let from = input.crs();
        let to = output.crs();
        let kind = match input {
            Caps::Raster(raster) => ReprojectedKind::Raster {
                value_kinds: raster.value_kinds.clone(),
            },
            Caps::Vector(_) => ReprojectedKind::Vector {
                latitude_band: latitude_band(from, to),
            },
            other => {
                unreachable!("negotiation hands a reproject raster or vector caps, got {other:?}")
            }
        };
        let mk = |a: Crs, b: Crs| {
            projicio_core::Transform::new(&a.authority(), &b.authority())
                .map_err(|e| Error::Projection(e.to_string()))
        };
        self.configured = Some(ConfiguredReprojection {
            kind,
            forward: mk(from, to)?,
            inverse: mk(to, from)?,
        });
        Ok(())
    }

    fn output_grid(&self, input: &GridSpec) -> GridSpec {
        if let Some(anchor) = self.output_anchor {
            return GridSpec {
                origin_x: anchor.origin_x,
                origin_y: anchor.origin_y,
                base_resolution: anchor.base_resolution,
                chunk_px: input.chunk_px,
            };
        }
        let configured = self.configured();
        // a vector base resolution is a segment length and can be degrees
        // wide, so a thousand-cell probe walks off the projection's domain
        let probe_cells = match configured.kind {
            ReprojectedKind::Raster { .. } => 1000.0,
            ReprojectedKind::Vector { .. } => 1.0,
        };
        projected_grid(&configured.forward, self.to, input, probe_cells)
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        let (bbox, resolution) =
            inverse_window(&self.configured().inverse, &out.bbox, out.resolution);
        out.with_window(bbox, resolution)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        envelope(&self.configured().forward, dirty)
            .map(|e| e.expand(2.0 * resolution))
            .unwrap_or(*dirty)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let configured = self.configured();
        match &configured.kind {
            ReprojectedKind::Raster { value_kinds } => warp_to_grid(
                input.raster()?,
                out,
                &configured.inverse,
                self.to,
                value_kinds,
            )
            .map(Chunk::Raster),
            ReprojectedKind::Vector { latitude_band } => project_features(
                input.vector()?,
                out,
                &configured.forward,
                *latitude_band,
                self.to,
            )
            .map(Chunk::Vector),
        }
    }
}

fn projected_grid(
    forward: &projicio_core::Transform,
    to: Crs,
    input: &GridSpec,
    probe_cells: f64,
) -> GridSpec {
    let cx = input.origin_x + probe_cells * input.base_resolution;
    let cy = input.origin_y - probe_cells * input.base_resolution;
    let origin = forward
        .convert(input.origin_x, input.origin_y)
        .unwrap_or((input.origin_x, input.origin_y));
    let (origin_x, origin_y) = canonical_origin(to, origin);
    GridSpec {
        origin_x,
        origin_y,
        base_resolution: local_scale(forward, cx, cy, input.base_resolution),
        chunk_px: input.chunk_px,
    }
}

fn project_features(
    input: &VectorChunk,
    out: &WindowReq,
    forward: &projicio_core::Transform,
    latitude_band: Option<LatitudeBand>,
    to: Crs,
) -> Result<VectorChunk> {
    let reaching_past_band =
        latitude_band.filter(|band| input.bbox.min_y < band.south || input.bbox.max_y > band.north);
    let inside: Vec<(&VectorFeature, Cow<'_, FeatureGeometry>)> = match reaching_past_band {
        None => input
            .features
            .iter()
            .map(|f| (f, Cow::Borrowed(&f.geometry)))
            .collect(),
        Some(band) => {
            let band_bbox = Bbox::new(
                input.bbox.min_x,
                input.bbox.min_y.max(band.south),
                input.bbox.max_x,
                input.bbox.max_y.min(band.north),
            );
            if band_bbox.min_y >= band_bbox.max_y {
                Vec::new()
            } else {
                input
                    .features
                    .iter()
                    .flat_map(|f| {
                        clip_geometry(&f.geometry, &band_bbox)
                            .into_iter()
                            .map(move |g| (f, Cow::Owned(g)))
                    })
                    .collect()
            }
        }
    };
    let mut coords = Vec::new();
    for (_, geometry) in &inside {
        collect_coords(geometry, &mut coords);
    }
    // vertices only, no densification, the same trade geodukt's reproject makes
    let projected = forward
        .convert_batch(&coords)
        .map_err(|e| Error::Projection(e.to_string()))?;
    let mut src = projected.into_iter().map(|(x, y)| Coord::new(x, y));
    let mut features = Vec::new();
    for (f, geometry) in &inside {
        let geometry = rebuild(geometry, &mut src);
        for geometry in clip_geometry(&geometry, &out.bbox) {
            features.push(VectorFeature {
                id: f.id,
                geometry,
                properties: f.properties.clone(),
            });
        }
    }
    Ok(VectorChunk::new(features, out.bbox, out.resolution, to))
}

fn collect_coords(geometry: &FeatureGeometry, out: &mut Vec<(f64, f64)>) {
    match geometry {
        FeatureGeometry::Point(p) => out.push((p.0.x, p.0.y)),
        FeatureGeometry::MultiPoint(mp) => out.extend(mp.points().iter().map(|p| (p.0.x, p.0.y))),
        FeatureGeometry::LineString(l) => out.extend(l.coords().iter().map(|c| (c.x, c.y))),
        FeatureGeometry::MultiLineString(mls) => {
            for l in mls.linestrings() {
                out.extend(l.coords().iter().map(|c| (c.x, c.y)));
            }
        }
        FeatureGeometry::Polygon(p) => collect_polygon(p, out),
        FeatureGeometry::MultiPolygon(mp) => {
            for p in mp.polygons() {
                collect_polygon(p, out);
            }
        }
        FeatureGeometry::GeometryCollection(members) => {
            for m in members {
                collect_coords(m, out);
            }
        }
    }
}

fn collect_polygon(p: &Polygon, out: &mut Vec<(f64, f64)>) {
    out.extend(p.exterior().coords().iter().map(|c| (c.x, c.y)));
    for hole in p.interiors() {
        out.extend(hole.coords().iter().map(|c| (c.x, c.y)));
    }
}

/// rebuild a geometry from projected coordinates, walking exactly the order
/// `collect_coords` wrote them in
fn rebuild(geometry: &FeatureGeometry, src: &mut impl Iterator<Item = Coord>) -> FeatureGeometry {
    let mut take = |n: usize| -> Vec<Coord> { src.take(n).collect() };
    match geometry {
        FeatureGeometry::Point(_) => FeatureGeometry::Point(Point(take(1)[0])),
        FeatureGeometry::MultiPoint(mp) => FeatureGeometry::MultiPoint(MultiPoint::new(
            take(mp.points().len()).into_iter().map(Point).collect(),
        )),
        FeatureGeometry::LineString(l) => {
            FeatureGeometry::LineString(LineString::new(take(l.coords().len())))
        }
        FeatureGeometry::MultiLineString(mls) => {
            FeatureGeometry::MultiLineString(MultiLineString::new(
                mls.linestrings()
                    .iter()
                    .map(|l| LineString::new(take(l.coords().len())))
                    .collect(),
            ))
        }
        FeatureGeometry::Polygon(p) => FeatureGeometry::Polygon(rebuild_polygon(p, src)),
        FeatureGeometry::MultiPolygon(mp) => FeatureGeometry::MultiPolygon(MultiPolygon::new(
            mp.polygons()
                .iter()
                .map(|p| rebuild_polygon(p, src))
                .collect(),
        )),
        FeatureGeometry::GeometryCollection(members) => {
            FeatureGeometry::GeometryCollection(members.iter().map(|m| rebuild(m, src)).collect())
        }
    }
}

fn rebuild_polygon(p: &Polygon, src: &mut impl Iterator<Item = Coord>) -> Polygon {
    let exterior = Ring::new(src.take(p.exterior().coords().len()).collect());
    let holes = p
        .interiors()
        .iter()
        .map(|h| Ring::new(src.take(h.coords().len()).collect()))
        .collect();
    Polygon::new(exterior, holes)
}
