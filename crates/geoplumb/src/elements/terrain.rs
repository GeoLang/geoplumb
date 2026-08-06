//! terrain derivatives over terrano kernels. both use a 3x3 neighborhood,
//! so the upstream plan widens the window by one cell of halo and compute
//! crops it back off, which is what keeps chunk borders seam-free

use crate::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Transform;
use crate::error::{Error, Result};
use crate::window::{Bbox, WindowReq};
use terrano_core::{BandedRaster, Raster};

const HALO_CELLS: f64 = 2.0;

fn single_band() -> CapsSet {
    CapsSet::one(CapsPattern::Raster(RasterPattern {
        bands: SetField::one(1),
        ..RasterPattern::default()
    }))
}

fn plan_with_halo(out: &WindowReq) -> WindowReq {
    out.with_window(out.bbox.expand(HALO_CELLS * out.resolution), out.resolution)
}

fn crop(result: Raster, input: &RasterChunk, out: &WindowReq) -> Result<Chunk> {
    let full = RasterChunk {
        bands: BandedRaster::new(vec![result]).expect("one band"),
        bbox: input.bbox,
        resolution: input.resolution,
        crs: input.crs,
    };
    Ok(Chunk::Raster(full.crop_to(&out.bbox)))
}

fn dem_band(input: &RasterChunk) -> Result<&Raster> {
    input
        .bands
        .band(0)
        .ok_or(Error::Source("empty input chunk".into()))
}

pub struct Hillshade {
    pub azimuth: f64,
    pub altitude: f64,
}

impl Hillshade {
    pub fn new(azimuth: f64, altitude: f64) -> Self {
        Hillshade { azimuth, altitude }
    }
}

impl Transform for Hillshade {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(single_band())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        plan_with_halo(out)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        dirty.expand(HALO_CELLS * resolution)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let shaded = terrano_core::hillshade(dem_band(input)?, self.azimuth, self.altitude);
        crop(shaded, input, out)
    }
}

pub struct Slope;

impl Transform for Slope {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(single_band())
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        plan_with_halo(out)
    }

    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        dirty.expand(HALO_CELLS * resolution)
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> Result<Chunk> {
        let input = input.raster()?;
        let sloped = terrano_core::slope(dem_band(input)?);
        crop(sloped, input, out)
    }
}
