//! crs change via projicio. the request plan inverse-projects the output
//! window into the source crs, compute inverse-maps each output pixel
//! center and samples the input bilinearly

use crate::caps::{
    Caps, CapsPattern, CapsSet, Constraint, Crs, FieldMask, RasterPattern, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, GridSpec, WindowReq};
use terrano_core::{BandedRaster, Raster};

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
        let scale = match (
            fwd.convert(cx, cy),
            fwd.convert(cx + input.base_resolution, cy),
        ) {
            (Ok((x0, y0)), Ok((x1, y1))) => ((x1 - x0).hypot(y1 - y0)).max(1e-12),
            _ => input.base_resolution,
        };
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
        let env = Self::envelope(self.inv(), &out.bbox).unwrap_or(out.bbox);
        // local scale at the window center decides the upstream resolution
        let cx = (out.bbox.min_x + out.bbox.max_x) / 2.0;
        let cy = (out.bbox.min_y + out.bbox.max_y) / 2.0;
        let in_res = match (
            self.inv().convert(cx, cy),
            self.inv().convert(cx + out.resolution, cy),
        ) {
            (Ok((x0, y0)), Ok((x1, y1))) => (x1 - x0).hypot(y1 - y0).max(1e-12),
            _ => out.resolution,
        };
        WindowReq {
            bbox: env.expand(2.0 * in_res),
            resolution: in_res,
        }
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        Self::envelope(self.fwd(), dirty)
            .map(|e| e.expand(2.0 * resolution))
            .unwrap_or(*dirty)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
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
        let src_pts = self
            .inv()
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
        Ok(Chunk::Raster(RasterChunk {
            bands: BandedRaster::new(bands).expect("uniform bands"),
            bbox: out.bbox,
            resolution: res,
            crs: self.to,
        }))
    }
}
