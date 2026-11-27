use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use futures::future::BoxFuture;
use gdal::Dataset;
use gdal::raster::{GdalDataType, GdalType, RasterBand, ResampleAlg};
use terrano_core::{SampleFormat, TypedBand};

use crate::bands::{RasterBands, Sample};
use crate::caps::{
    BandDtypes, BandKinds, CapsPattern, CapsSet, Constraint, Crs, Dtype, RasterPattern, ResRange,
    SetField,
};
use crate::chunk::{Chunk, RasterChunk};
use crate::element::Source;
use crate::elements::cog::decimate_band;
use crate::elements::file_stamp;
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
    dtypes: Vec<Dtype>,
    readers: Vec<BandReader>,
    // pixel width of the base level, then of each overview in gdal order
    level_resolutions: Vec<f64>,
    origin_x: f64,
    origin_y: f64,
    crs: Crs,
    value_kinds: BandKinds,
    // a dataset is Send but not Sync, so each read takes one out
    pool: Mutex<Vec<Dataset>>,
}

impl GdalSrc {
    /// every band of the file, blocking, call it off the async runtime
    pub fn open(path: impl AsRef<Path>, value_kinds: impl Into<BandKinds>) -> Result<Self> {
        Self::open_bands(path.as_ref(), None, value_kinds.into())
    }

    /// one 1-based band of the file, blocking like `open`
    pub fn open_band(
        path: impl AsRef<Path>,
        band: u16,
        value_kinds: impl Into<BandKinds>,
    ) -> Result<Self> {
        Self::open_bands(path.as_ref(), Some(band), value_kinds.into())
    }

    fn open_bands(path: &Path, band: Option<u16>, value_kinds: BandKinds) -> Result<Self> {
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
        if !value_kinds.fits(bands.len()) {
            return Err(named(format!(
                "{value_kinds:?} does not fit the {} bands served",
                bands.len()
            )));
        }

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
        let mut dtypes = Vec::with_capacity(bands.len());
        let mut readers = Vec::with_capacity(bands.len());
        let mut overview_widths: Option<Vec<usize>> = None;
        let base_width = dataset.raster_size().0;
        for &b in &bands {
            let raster_band = dataset.rasterband(b).map_err(|e| named(e.to_string()))?;
            let (format, reader) = band_reader(raster_band.band_type());
            let dtype = Dtype::from_format(format, raster_band.no_data_value())
                .map_err(|e| named(format!("band {b}: {e}")))?;
            nodata.push(raster_band.no_data_value());
            dtypes.push(dtype);
            readers.push(reader);
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
                dtypes,
                readers,
                level_resolutions,
                origin_x,
                origin_y,
                crs: Crs(code),
                value_kinds,
                pool: Mutex::new(vec![dataset]),
            }),
        })
    }

    /// the short name of the gdal driver that opened the file, `GTiff` say
    pub fn driver(&self) -> &str {
        &self.file.driver
    }
}

struct BandWindow {
    col0: i64,
    row0: i64,
    cols: usize,
    rows: usize,
    cell_size: f64,
    nodata: Option<f64>,
}

type BandReader = fn(&RasterBand, &BandWindow) -> gdal::errors::Result<TypedBand>;

fn band_reader(band_type: GdalDataType) -> (SampleFormat, BandReader) {
    match band_type {
        GdalDataType::UInt8 => (SampleFormat::U8, read_native::<u8> as BandReader),
        GdalDataType::UInt16 => (SampleFormat::U16, read_native::<u16> as BandReader),
        GdalDataType::Int16 => (SampleFormat::I16, read_native::<i16> as BandReader),
        GdalDataType::UInt32 => (SampleFormat::U32, read_native::<u32> as BandReader),
        GdalDataType::Int32 => (SampleFormat::I32, read_native::<i32> as BandReader),
        GdalDataType::Float32 => (SampleFormat::F32, read_native::<f32> as BandReader),
        // the gdal crate builds Int8 only against gdal 3.7 and later, 64-bit integers have no terrano format
        _ => (SampleFormat::F64, read_native::<f64> as BandReader),
    }
}

fn read_native<T: Sample + GdalType>(
    band: &RasterBand,
    window: &BandWindow,
) -> gdal::errors::Result<TypedBand> {
    let nodata = window.nodata.map(|value| T::narrow(value, None));
    let sentinel = if T::FLOAT { None } else { nodata };
    let fill = T::fill(sentinel);
    let mut plane = vec![fill; window.cols * window.rows];
    // gdal refuses a window past the band edge, so clamp both ends and copy at an offset
    let (x0, x1) = (
        window.col0.max(0),
        (window.col0 + window.cols as i64).min(band.x_size() as i64),
    );
    let (y0, y1) = (
        window.row0.max(0),
        (window.row0 + window.rows as i64).min(band.y_size() as i64),
    );
    if x1 > x0 && y1 > y0 {
        let size = ((x1 - x0) as usize, (y1 - y0) as usize);
        let read = band.read_as::<T>(
            (x0 as isize, y0 as isize),
            size,
            size,
            Some(ResampleAlg::NearestNeighbour),
        )?;
        let (skip_c, skip_r) = ((x0 - window.col0) as usize, (y0 - window.row0) as usize);
        for (r, row) in read.data().chunks_exact(size.0).enumerate() {
            let dst = (skip_r + r) * window.cols + skip_c;
            for (out, &value) in plane[dst..dst + size.0].iter_mut().zip(row) {
                *out = if T::FLOAT && Some(value) == nodata {
                    fill
                } else {
                    value
                };
            }
        }
    }
    Ok(T::build(
        window.cols,
        window.rows,
        window.cell_size,
        plane,
        sentinel,
    ))
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
        let col0 = ((req.bbox.min_x - self.origin_x) / level_resolution).round() as i64;
        let row0 = ((self.origin_y - req.bbox.max_y) / level_resolution).round() as i64;
        let window = |nodata| BandWindow {
            col0,
            row0,
            cols: cols * factor,
            rows: rows * factor,
            cell_size: if factor == 1 {
                req.resolution
            } else {
                level_resolution
            },
            nodata,
        };

        let dataset = self.take_dataset()?;
        let fine = self.read_bands(&dataset, level, window);
        self.return_dataset(dataset);

        let bands = fine?
            .into_iter()
            .enumerate()
            .map(|(bi, band)| {
                if factor == 1 {
                    return band;
                }
                decimate_band(&band, factor, self.value_kinds.band(bi), req.resolution)
            })
            .collect();
        Ok(RasterChunk {
            bands: RasterBands::new(bands)?,
            bbox: req.bbox,
            resolution: req.resolution,
            crs: self.crs,
        })
    }

    fn read_bands(
        &self,
        dataset: &Dataset,
        level: usize,
        window: impl Fn(Option<f64>) -> BandWindow,
    ) -> Result<Vec<TypedBand>> {
        let named =
            |e: gdal::errors::GdalError| Error::Source(format!("{}: {e}", self.path.display()));
        let mut bands = Vec::with_capacity(self.bands.len());
        for ((&b, &nodata), read) in self.bands.iter().zip(&self.nodata).zip(&self.readers) {
            let base = dataset.rasterband(b).map_err(named)?;
            let band = match level {
                0 => base,
                overview => base.overview(overview - 1).map_err(named)?,
            };
            bands.push(read(&band, &window(nodata)).map_err(named)?);
        }
        Ok(bands)
    }
}

impl Source for GdalSrc {
    fn identity(&self) -> String {
        let file = &self.file;
        format!(
            "GdalSrc path={:?} driver={:?} bands={:?} nodata={:?} level_resolutions={:?} origin_x={:?} origin_y={:?} crs={} value_kinds={:?} {}",
            file.path,
            file.driver,
            file.bands,
            file.nodata,
            file.level_resolutions,
            file.origin_x,
            file.origin_y,
            file.crs,
            file.value_kinds,
            file_stamp(&file.path)
        )
    }

    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::Raster(RasterPattern {
            dtypes: SetField::one(BandDtypes::from_bands(self.file.dtypes.clone())),
            bands: SetField::one(self.file.served_band_count()),
            crs: SetField::one(self.file.crs),
            resolution: ResRange::at_least(self.file.level_resolutions[0]),
            chunk_px: SetField::Any,
            value_kinds: SetField::one(self.file.value_kinds.clone()),
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
