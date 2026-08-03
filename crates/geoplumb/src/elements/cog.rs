//! windowed cog source. each pull fetches only the tiles it touches
//! through terrano's CogReader, served from the file overview nearest the
//! requested ladder level. when the file pyramid is shallower than the
//! request the remainder is block-averaged, matching RasterSrc semantics

use std::sync::{Arc, Mutex};

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::error::Result;
use crate::window::{GridSpec, WindowReq};
use futures::future::BoxFuture;
use terrano_core::{BandedRaster, CogReader, RangeRead, Raster};

pub struct CogSrc<R: RangeRead + Send + 'static> {
    // read_window needs &mut, so concurrent chunk reads serialize here
    reader: Arc<Mutex<CogReader<R>>>,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    crs: Crs,
}

impl<R: RangeRead + Send + 'static> CogSrc<R> {
    /// reads the file layout with blocking range requests, call it off
    /// the async runtime (`spawn_blocking`) when one is running
    pub fn open(source: R) -> Result<Self> {
        let reader = CogReader::open(source)?;
        let meta = reader.meta().clone();
        Ok(CogSrc {
            origin_x: meta.origin_x,
            origin_y: meta.origin_y,
            base_resolution: meta.pixel_width,
            crs: Crs(u32::from(meta.epsg)),
            reader: Arc::new(Mutex::new(reader)),
        })
    }
}

impl<R: RangeRead + Send + 'static> Source for CogSrc<R> {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtype: SetField::one(Dtype::F64),
            bands: SetField::one(1),
            crs: SetField::one(self.crs),
            resolution: ResRange::at_least(self.base_resolution),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.origin_x,
            origin_y: self.origin_y,
            base_resolution: self.base_resolution,
            chunk_px: 256,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let reader = self.reader.clone();
        let req = *req;
        let (origin_x, origin_y, crs) = (self.origin_x, self.origin_y, self.crs);
        Box::pin(async move {
            crate::engine::offload(move || {
                read_chunk(&mut reader.lock().unwrap(), &req, origin_x, origin_y, crs)
                    .map(Chunk::Raster)
            })
            .await
        })
    }
}

pub(crate) fn read_chunk<R: RangeRead>(
    reader: &mut CogReader<R>,
    req: &WindowReq,
    origin_x: f64,
    origin_y: f64,
    crs: Crs,
) -> Result<RasterChunk> {
    let level = reader.select_level(req.resolution);
    let lres = reader.levels()[level].pixel_width;
    let factor = (req.resolution / lres).round().max(1.0) as usize;
    let cols = (req.bbox.width() / req.resolution).round() as usize;
    let rows = (req.bbox.height() / req.resolution).round() as usize;
    let (fcols, frows) = (cols * factor, rows * factor);
    let col0 = ((req.bbox.min_x - origin_x) / lres).round() as i64;
    let row0 = ((origin_y - req.bbox.max_y) / lres).round() as i64;

    // read_window pads right/bottom itself but cannot start left/above the
    // image, so clamp the start and copy at an offset
    let mut fine = vec![f64::NAN; fcols * frows];
    let (skip_c, skip_r) = ((-col0).max(0) as usize, (-row0).max(0) as usize);
    if skip_c < fcols && skip_r < frows {
        let window = reader.read_window(
            level,
            (col0 + skip_c as i64) as usize,
            (row0 + skip_r as i64) as usize,
            fcols - skip_c,
            frows - skip_r,
        )?;
        let w = window.width();
        for r in 0..window.height() {
            let dst = (skip_r + r) * fcols + skip_c;
            fine[dst..dst + w].copy_from_slice(&window.data()[r * w..(r + 1) * w]);
        }
    }

    let data = if factor == 1 {
        fine
    } else {
        decimate(&fine, cols, rows, factor)
    };
    let band = Raster::from_vec(cols, rows, data, req.resolution, f64::NAN).expect("window dims");
    Ok(RasterChunk {
        bands: BandedRaster::new(vec![band]).expect("single band"),
        bbox: req.bbox,
        resolution: req.resolution,
        crs,
    })
}

fn decimate(fine: &[f64], cols: usize, rows: usize, factor: usize) -> Vec<f64> {
    let fcols = cols * factor;
    let mut out = vec![f64::NAN; cols * rows];
    for row in 0..rows {
        for col in 0..cols {
            let mut sum = 0.0;
            let mut n = 0usize;
            for rr in 0..factor {
                for cc in 0..factor {
                    let v = fine[(row * factor + rr) * fcols + col * factor + cc];
                    if v.is_finite() {
                        sum += v;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                out[row * cols + col] = sum / n as f64;
            }
        }
    }
    out
}

/// range-request transport for remote cogs. blocking, so it must run off
/// the async runtime, which `CogSrc::read` already does
pub struct HttpRange {
    client: reqwest::blocking::Client,
    url: String,
}

impl HttpRange {
    pub fn new(url: impl Into<String>) -> Self {
        HttpRange {
            client: reqwest::blocking::Client::new(),
            url: url.into(),
        }
    }
}

impl RangeRead for HttpRange {
    fn read_range(
        &mut self,
        offset: u64,
        len: u64,
    ) -> core::result::Result<Vec<u8>, terrano_core::Error> {
        let fail = |detail: String| terrano_core::Error::Format(detail);
        let end = offset + len - 1;
        let resp = self
            .client
            .get(&self.url)
            .header(reqwest::header::RANGE, format!("bytes={offset}-{end}"))
            .send()
            .and_then(|r| r.error_for_status())
            .map_err(|e| fail(format!("range request failed: {e}")))?;
        if resp.status() != reqwest::StatusCode::PARTIAL_CONTENT {
            return Err(fail(format!(
                "server ignored the range header (status {})",
                resp.status()
            )));
        }
        let bytes = resp
            .bytes()
            .map_err(|e| fail(format!("range body read failed: {e}")))?;
        if bytes.len() as u64 != len {
            return Err(fail(format!(
                "range returned {} bytes, wanted {len}",
                bytes.len()
            )));
        }
        Ok(bytes.to_vec())
    }
}
