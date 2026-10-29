use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use gdal::Dataset;
use gdal::raster::{RasterBand, ResampleAlg};
use terrano_core::{BandedRaster, Raster};

use crate::caps::{
    CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange, SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::elements::cog::decimate;
use crate::error::{Error, Result};
use crate::window::{GridSpec, WindowReq};

const POOLED_DATASETS: usize = 4;

const SQUARE_PIXEL_TOLERANCE: f64 = 1e-9;

/// a raster any gdal driver opens, read window by window from the file
/// overview nearest the requested ladder level
pub struct GdalSrc {
    file: Arc<GdalFile>,
}

struct GdalFile {
    path: PathBuf,
    driver: String,
    // 1-based gdal band indices, in the order the chunk carries them
    bands: Vec<usize>,
    nodata: Vec<Option<f64>>,
    // pixel width of the base level, then of each overview in gdal order
    level_resolutions: Vec<f64>,
    origin_x: f64,
    origin_y: f64,
    crs: Crs,
    // a dataset is Send but not Sync, so each read takes one out
    pool: Mutex<Vec<Dataset>>,
}

impl GdalSrc {
    /// every band of the file, blocking, call it off the async runtime
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        Self::open_bands(path.as_ref(), None)
    }

    /// one 1-based band of the file, blocking like `open`
    pub fn open_band(path: impl AsRef<Path>, band: u16) -> Result<Self> {
        Self::open_bands(path.as_ref(), Some(band))
    }

    fn open_bands(path: &Path, band: Option<u16>) -> Result<Self> {
        let named = |detail: String| Error::Source(format!("{}: {detail}", path.display()));
        let dataset = Dataset::open(path).map_err(|e| named(e.to_string()))?;
        let count = dataset.raster_count();
        if count == 0 {
            return Err(named("the dataset has no raster bands".into()));
        }
        let band_count = u16::try_from(count)
            .map_err(|_| named(format!("{count} bands is more than a chunk carries")))?;
        let bands: Vec<usize> = match band {
            None => (1..=count).collect(),
            Some(b) if b == 0 || b > band_count => {
                return Err(named(format!(
                    "band {b} is out of range, the dataset has {count} bands"
                )));
            }
            Some(b) => vec![usize::from(b)],
        };

        let transform = dataset
            .geo_transform()
            .map_err(|e| named(format!("no geotransform: {e}")))?;
        let [
            origin_x,
            pixel_width,
            row_rotation,
            origin_y,
            column_rotation,
            pixel_height,
        ] = transform;
        if row_rotation != 0.0 || column_rotation != 0.0 {
            return Err(named(format!(
                "the geotransform is rotated or sheared ({row_rotation}, {column_rotation}), \
                 the ladder has no level for it"
            )));
        }
        let square = pixel_width > 0.0
            && (pixel_height + pixel_width).abs() <= SQUARE_PIXEL_TOLERANCE * pixel_width;
        if !square {
            return Err(named(format!(
                "pixel size {pixel_width} by {pixel_height} is not square and north up, \
                 the ladder has one resolution"
            )));
        }

        let spatial_ref = dataset
            .spatial_ref()
            .map_err(|e| named(format!("no spatial reference: {e}")))?;
        let authority = spatial_ref.auth_name();
        if authority.as_deref() != Some("EPSG") {
            return Err(named(format!(
                "the crs authority is {authority:?}, only EPSG codes are served"
            )));
        }
        let code = spatial_ref
            .auth_code()
            .map_err(|e| named(format!("the crs has no epsg code: {e}")))?;
        let code = u32::try_from(code).map_err(|_| named(format!("epsg code {code}")))?;

        let mut nodata = Vec::with_capacity(bands.len());
        let mut overview_widths: Option<Vec<usize>> = None;
        let base_width = dataset.raster_size().0;
        for &b in &bands {
            let raster_band = dataset.rasterband(b).map_err(|e| named(e.to_string()))?;
            nodata.push(raster_band.no_data_value());
            let widths = overview_widths_of(&raster_band).map_err(|e| named(e.to_string()))?;
            match &overview_widths {
                None => overview_widths = Some(widths),
                Some(first) if *first != widths => {
                    return Err(named(format!(
                        "band {b} has overview widths {widths:?}, band {} has {first:?}",
                        bands[0]
                    )));
                }
                Some(_) => {}
            }
        }
        let level_resolutions = std::iter::once(pixel_width)
            .chain(
                overview_widths
                    .expect("at least one band")
                    .into_iter()
                    .map(|w| pixel_width * base_width as f64 / w as f64),
            )
            .collect();

        Ok(GdalSrc {
            file: Arc::new(GdalFile {
                path: path.to_path_buf(),
                driver: dataset.driver().short_name(),
                bands,
                nodata,
                level_resolutions,
                origin_x,
                origin_y,
                crs: Crs(code),
                pool: Mutex::new(vec![dataset]),
            }),
        })
    }

    /// the short name of the gdal driver that opened the file, `GTiff` say
    pub fn driver(&self) -> &str {
        &self.file.driver
    }
}

fn overview_widths_of(band: &RasterBand) -> gdal::errors::Result<Vec<usize>> {
    (0..band.overview_count()?)
        .map(|i| Ok(band.overview(i as usize)?.x_size()))
        .collect()
}

impl GdalFile {
    fn served_band_count(&self) -> u16 {
        u16::try_from(self.bands.len()).expect("checked against u16 at open")
    }

    // same rule as terrano's CogReader::select_level
    fn select_level(&self, target_pixel_width: f64) -> usize {
        let mut best = 0;
        for (i, &resolution) in self.level_resolutions.iter().enumerate() {
            if resolution <= target_pixel_width * (1.0 + 1e-9)
                && resolution > self.level_resolutions[best]
            {
                best = i;
            }
        }
        best
    }

    fn take_dataset(&self) -> Result<Dataset> {
        let pooled = self.pool.lock().unwrap().pop();
        match pooled {
            Some(dataset) => Ok(dataset),
            None => Dataset::open(&self.path)
                .map_err(|e| Error::Source(format!("{}: {e}", self.path.display()))),
        }
    }

    fn return_dataset(&self, dataset: Dataset) {
        let mut pool = self.pool.lock().unwrap();
        if pool.len() < POOLED_DATASETS {
            pool.push(dataset);
        }
    }

    fn read_chunk(&self, req: &WindowReq) -> Result<RasterChunk> {
        let level = self.select_level(req.resolution);
        let level_resolution = self.level_resolutions[level];
        let factor = (req.resolution / level_resolution).round().max(1.0) as usize;
        let cols = (req.bbox.width() / req.resolution).round() as usize;
        let rows = (req.bbox.height() / req.resolution).round() as usize;
        let (fine_cols, fine_rows) = (cols * factor, rows * factor);
        let col0 = ((req.bbox.min_x - self.origin_x) / level_resolution).round() as i64;
        let row0 = ((self.origin_y - req.bbox.max_y) / level_resolution).round() as i64;

        let dataset = self.take_dataset()?;
        let planes = self.read_planes(&dataset, level, (col0, row0), (fine_cols, fine_rows));
        self.return_dataset(dataset);

        let bands: Vec<Raster> = planes?
            .into_iter()
            .map(|plane| {
                let data = if factor == 1 {
                    plane
                } else {
                    decimate(&plane, cols, rows, factor)
                };
                Raster::from_vec(cols, rows, data, req.resolution, f64::NAN).expect("window dims")
            })
            .collect();
        Ok(RasterChunk {
            bands: BandedRaster::new(bands).map_err(Error::Terrano)?,
            bbox: req.bbox,
            resolution: req.resolution,
            crs: self.crs,
        })
    }

    fn read_planes(
        &self,
        dataset: &Dataset,
        level: usize,
        (col0, row0): (i64, i64),
        (fine_cols, fine_rows): (usize, usize),
    ) -> Result<Vec<Vec<f64>>> {
        let named =
            |e: gdal::errors::GdalError| Error::Source(format!("{}: {e}", self.path.display()));
        let mut planes = Vec::with_capacity(self.bands.len());
        for (&b, &nodata) in self.bands.iter().zip(&self.nodata) {
            let base = dataset.rasterband(b).map_err(named)?;
            let band = match level {
                0 => base,
                overview => base.overview(overview - 1).map_err(named)?,
            };
            let mut plane = vec![f64::NAN; fine_cols * fine_rows];
            // gdal refuses a window past the band edge, so clamp both ends
            // and copy at an offset
            let (x0, x1) = (
                col0.max(0),
                (col0 + fine_cols as i64).min(band.x_size() as i64),
            );
            let (y0, y1) = (
                row0.max(0),
                (row0 + fine_rows as i64).min(band.y_size() as i64),
            );
            if x1 > x0 && y1 > y0 {
                let size = ((x1 - x0) as usize, (y1 - y0) as usize);
                let window = band
                    .read_as::<f64>(
                        (x0 as isize, y0 as isize),
                        size,
                        size,
                        Some(ResampleAlg::NearestNeighbour),
                    )
                    .map_err(named)?;
                let (skip_c, skip_r) = ((x0 - col0) as usize, (y0 - row0) as usize);
                for (r, row) in window.data().chunks_exact(size.0).enumerate() {
                    let dst = (skip_r + r) * fine_cols + skip_c;
                    for (out, &v) in plane[dst..dst + size.0].iter_mut().zip(row) {
                        *out = if Some(v) == nodata { f64::NAN } else { v };
                    }
                }
            }
            planes.push(plane);
        }
        Ok(planes)
    }
}

impl Source for GdalSrc {
    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtype: SetField::one(Dtype::F64),
            bands: SetField::one(self.file.served_band_count()),
            crs: SetField::one(self.file.crs),
            resolution: ResRange::at_least(self.file.level_resolutions[0]),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.file.origin_x,
            origin_y: self.file.origin_y,
            base_resolution: self.file.level_resolutions[0],
            chunk_px: 256,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let file = self.file.clone();
        let req = *req;
        Box::pin(async move {
            crate::engine::offload(move || file.read_chunk(&req).map(Chunk::Raster)).await
        })
    }
}
