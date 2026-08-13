//! crs change via projicio. the request plan inverse-projects the output
//! window into the source crs, compute inverse-maps each output pixel
//! center and samples the input bilinearly. `VecReproject` is the vector
//! twin: it forward-projects the fragments themselves

use crate::caps::{
    Caps, CapsPattern, CapsSet, Constraint, Crs, FieldMask, RasterPattern, SetField, VectorPattern,
};
use crate::chunk::{Chunk, RasterChunk, VectorChunk, VectorFeature, clip_geometry};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, GridSpec, WindowReq};
use terrano_core::{BandedRaster, Raster};
use topoi_core::geojson::FeatureGeometry;
use topoi_core::{
    Coord, LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon, Ring,
};

pub struct Reproject {
    to: Crs,
    forward: Option<projicio_core::Transform>,
    inverse: Option<projicio_core::Transform>,
}

impl Reproject {
    pub fn new(to: Crs) -> Self {
        Reproject {
            to,
            forward: None,
            inverse: None,
        }
    }

    /// auto-plug template: like `constraint` but with the target crs left
    /// free, so the solver retargets it to whatever the demanding side needs
    pub fn adapter() -> crate::element::Adapter {
        crate::element::Adapter {
            template: Constraint::Derived {
                input: CapsSet::any_raster(),
                passthrough: FieldMask::without_crs_resolution(),
                output: CapsPattern::Raster(RasterPattern::default()),
            },
            build: |target| {
                let CapsPattern::Raster(target) = target else {
                    return None;
                };
                let SetField::OneOf(crss) = &target.crs else {
                    return None;
                };
                Some(Box::new(Reproject::new(crss[0])))
            },
        }
    }

    fn inv(&self) -> &projicio_core::Transform {
        self.inverse.as_ref().expect("configured")
    }

    fn fwd(&self) -> &projicio_core::Transform {
        self.forward.as_ref().expect("configured")
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

/// resample `input` onto the output window by inverse-projecting every
/// output pixel center and sampling there. the one warp both this element
/// and a stac item on another crs go through, so a pixel looks the same
/// whichever asked for it
pub(crate) fn warp_to_grid(
    input: &RasterChunk,
    out: &WindowReq,
    inverse: &projicio_core::Transform,
    to: Crs,
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
    let bands: Vec<Raster> = input
        .bands
        .bands()
        .iter()
        .map(|band| {
            let nodata = band.nodata;
            let data = src_pts
                .iter()
                .map(|&(sx, sy)| {
                    crate::resample::sample_bilinear(band, input, sx, sy).unwrap_or(nodata)
                })
                .collect();
            Raster::from_vec(cols, rows, data, res, nodata).expect("reproject dims")
        })
        .collect();
    Ok(RasterChunk {
        bands: BandedRaster::new(bands).expect("uniform bands"),
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
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::any_raster(),
            passthrough: FieldMask::without_crs_resolution(),
            output: CapsPattern::Raster(RasterPattern {
                crs: SetField::one(self.to),
                ..RasterPattern::default()
            }),
        }
    }

    fn configure(&mut self, input: &Caps, output: &Caps) -> Result<()> {
        let from = input.raster().crs;
        let to = output.raster().crs;
        let mk = |a: Crs, b: Crs| {
            projicio_core::Transform::new(&a.authority(), &b.authority())
                .map_err(|e| Error::Projection(e.to_string()))
        };
        self.forward = Some(mk(from, to)?);
        self.inverse = Some(mk(to, from)?);
        Ok(())
    }

    fn output_grid(&self, input: &GridSpec) -> GridSpec {
        let fwd = self.fwd();
        let cx = input.origin_x + 1000.0 * input.base_resolution;
        let cy = input.origin_y - 1000.0 * input.base_resolution;
        let scale = local_scale(fwd, cx, cy, input.base_resolution);
        let origin = fwd
            .convert(input.origin_x, input.origin_y)
            .unwrap_or((input.origin_x, input.origin_y));
        let (origin_x, origin_y) = canonical_origin(self.to, origin);
        GridSpec {
            origin_x,
            origin_y,
            base_resolution: scale,
            chunk_px: input.chunk_px,
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        let (bbox, resolution) = inverse_window(self.inv(), &out.bbox, out.resolution);
        out.with_window(bbox, resolution)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        envelope(self.fwd(), dirty)
            .map(|e| e.expand(2.0 * resolution))
            .unwrap_or(*dirty)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        warp_to_grid(input.raster()?, out, self.inv(), self.to).map(Chunk::Raster)
    }
}

/// vector twin of `Reproject`: every fragment vertex is forward-projected
/// and the result re-clipped to the output tile. vertices only, no
/// densification, so a long segment keeps its chord across the projection,
/// the same trade geodukt's reproject makes
pub struct VecReproject {
    to: Crs,
    forward: Option<projicio_core::Transform>,
    inverse: Option<projicio_core::Transform>,
}

impl VecReproject {
    pub fn new(to: Crs) -> Self {
        VecReproject {
            to,
            forward: None,
            inverse: None,
        }
    }

    /// auto-plug template, the vector-side counterpart of
    /// `Reproject::adapter`. it declines raster and other kinds, so the
    /// solver picks whichever of the two fits the failing link
    pub fn adapter() -> crate::element::Adapter {
        crate::element::Adapter {
            template: Constraint::Derived {
                input: CapsSet::any_vector(),
                passthrough: FieldMask::without_crs_resolution(),
                output: CapsPattern::Vector(VectorPattern::default()),
            },
            build: |target| {
                let CapsPattern::Vector(target) = target else {
                    return None;
                };
                let SetField::OneOf(crss) = &target.crs else {
                    return None;
                };
                Some(Box::new(VecReproject::new(crss[0])))
            },
        }
    }

    fn inv(&self) -> &projicio_core::Transform {
        self.inverse.as_ref().expect("configured")
    }

    fn fwd(&self) -> &projicio_core::Transform {
        self.forward.as_ref().expect("configured")
    }
}

impl Transform for VecReproject {
    fn constraint(&self) -> Constraint {
        Constraint::Derived {
            input: CapsSet::any_vector(),
            passthrough: FieldMask::without_crs_resolution(),
            output: CapsPattern::Vector(VectorPattern {
                crs: SetField::one(self.to),
                ..VectorPattern::default()
            }),
        }
    }

    fn configure(&mut self, input: &Caps, output: &Caps) -> Result<()> {
        let from = input.vector().crs;
        let to = output.vector().crs;
        let mk = |a: Crs, b: Crs| {
            projicio_core::Transform::new(&a.authority(), &b.authority())
                .map_err(|e| Error::Projection(e.to_string()))
        };
        self.forward = Some(mk(from, to)?);
        self.inverse = Some(mk(to, from)?);
        Ok(())
    }

    fn output_grid(&self, input: &GridSpec) -> GridSpec {
        let fwd = self.fwd();
        // one cell in from the corner, not the thousand the raster twin
        // uses: a vector base resolution is a segment length and can be
        // degrees wide, so a long probe walks off the projection's domain
        let cx = input.origin_x + input.base_resolution;
        let cy = input.origin_y - input.base_resolution;
        let origin = fwd
            .convert(input.origin_x, input.origin_y)
            .unwrap_or((input.origin_x, input.origin_y));
        let (origin_x, origin_y) = canonical_origin(self.to, origin);
        GridSpec {
            origin_x,
            origin_y,
            base_resolution: local_scale(fwd, cx, cy, input.base_resolution),
            chunk_px: input.chunk_px,
        }
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        let (bbox, resolution) = inverse_window(self.inv(), &out.bbox, out.resolution);
        out.with_window(bbox, resolution)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        envelope(self.fwd(), dirty)
            .map(|e| e.expand(2.0 * resolution))
            .unwrap_or(*dirty)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.vector()?;
        let mut coords = Vec::new();
        for f in &input.features {
            collect_coords(&f.geometry, &mut coords);
        }
        let projected = self
            .fwd()
            .convert_batch(&coords)
            .map_err(|e| Error::Projection(e.to_string()))?;
        let mut src = projected.into_iter().map(|(x, y)| Coord::new(x, y));
        let mut features = Vec::new();
        for f in &input.features {
            let geometry = rebuild(&f.geometry, &mut src);
            for geometry in clip_geometry(&geometry, &out.bbox) {
                features.push(VectorFeature {
                    id: f.id,
                    geometry,
                    properties: f.properties.clone(),
                });
            }
        }
        Ok(Chunk::Vector(VectorChunk::new(
            features,
            out.bbox,
            out.resolution,
            self.to,
        )))
    }
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
