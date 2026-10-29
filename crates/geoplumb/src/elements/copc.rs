use std::fs::File;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use copc_core::{Bounds, CopcInfo};
use copc_reader::{
    BoundsSelection, CopcRangeReader, LasHeader, LodSelection, PointQuery, RangeRead,
};
use futures::future::BoxFuture;
use nubis_core::{Classification, Point3, PointCloud};

use crate::caps::{CapsPattern, CapsSet, Constraint, Crs, PointPattern, ResRange, SetField};
use crate::chunk::{Chunk, PointChunk, tile_contains};
use crate::element::Source;
use crate::elements::cog::HttpRange;
use crate::elements::file_stamp;
use crate::error::{Error, Result};
use crate::window::{GridSpec, WindowReq};

const POOLED_READERS: usize = 4;

type SourceOpener<S> = Box<dyn Fn() -> Result<S> + Send + Sync>;

// a tile pull fetches only the octree nodes its window and resolution need
pub struct CopcSrc<S: RangeRead + Send + 'static> {
    dataset: Arc<CopcDataset<S>>,
}

struct CopcDataset<S: RangeRead> {
    // the path or url every error names
    name: String,
    // set when the file is local, so its length and mtime join the identity
    path: Option<PathBuf>,
    header: LasHeader,
    info: CopcInfo,
    open_source: SourceOpener<S>,
    // a reader takes `&mut self` to read, so each read takes one out
    readers: Mutex<Vec<CopcRangeReader<S>>>,
    origin_x: f64,
    origin_y: f64,
    base_resolution: f64,
    min_z: f64,
    max_z: f64,
    crs: Crs,
}

impl<S: RangeRead + Clone + Send + Sync + 'static> CopcSrc<S> {
    // blocking, call it off the async runtime when one is running
    pub fn open(source: S, crs: Crs) -> Result<Self> {
        Self::open_with(
            "copc".into(),
            None,
            Box::new(move || Ok(source.clone())),
            crs,
        )
    }
}

impl CopcSrc<File> {
    pub fn open_path(path: impl AsRef<Path>, crs: Crs) -> Result<Self> {
        let path: PathBuf = path.as_ref().to_path_buf();
        let name = path.display().to_string();
        let opener_name = name.clone();
        let opener_path = path.clone();
        Self::open_with(
            name,
            Some(path),
            Box::new(move || {
                File::open(&opener_path).map_err(|e| Error::Source(format!("{opener_name}: {e}")))
            }),
            crs,
        )
    }
}

impl CopcSrc<CopcHttpRange> {
    // blocking like `open`, and the fetches run on the cog transport's runtime
    pub fn open_url(url: impl Into<String>, crs: Crs) -> Result<Self> {
        let url = url.into();
        let source = CopcHttpRange::new(&url)?;
        Self::open_with(url, None, Box::new(move || Ok(source.clone())), crs)
    }
}

impl<S: RangeRead + Send + 'static> CopcSrc<S> {
    // the crs comes from the caller since nothing in the org parses the file's wkt vlr
    fn open_with(
        name: String,
        path: Option<PathBuf>,
        open_source: SourceOpener<S>,
        crs: Crs,
    ) -> Result<Self> {
        let named = |e: copc_core::Error| Error::Source(format!("{name}: {e}"));
        let mut reader = CopcRangeReader::open(open_source()?).map_err(named)?;
        let deepest = reader
            .hierarchy_for(PointQuery::all())
            .map_err(named)?
            .iter()
            .filter(|entry| entry.has_point_data())
            .map(|entry| entry.key.level)
            .max()
            .ok_or_else(|| Error::Source(format!("{name}: the copc file holds no points")))?;
        let header = *reader.header();
        let info = *reader.copc_info();
        Ok(CopcSrc {
            dataset: Arc::new(CopcDataset {
                name,
                path,
                header,
                info,
                open_source,
                readers: Mutex::new(vec![reader]),
                origin_x: header.min_x,
                origin_y: header.max_y,
                base_resolution: info.spacing / 2f64.powi(deepest),
                min_z: header.min_z,
                max_z: header.max_z,
                crs,
            }),
        })
    }
}

impl<S: RangeRead + Send + 'static> CopcDataset<S> {
    fn named(&self, e: copc_core::Error) -> Error {
        Error::Source(format!("{}: {e}", self.name))
    }

    fn take_reader(&self) -> Result<CopcRangeReader<S>> {
        let pooled = self.readers.lock().unwrap().pop();
        match pooled {
            Some(reader) => Ok(reader),
            None => CopcRangeReader::open((self.open_source)()?).map_err(|e| self.named(e)),
        }
    }

    fn return_reader(&self, reader: CopcRangeReader<S>) {
        let mut readers = self.readers.lock().unwrap();
        if readers.len() < POOLED_READERS {
            readers.push(reader);
        }
    }

    fn read_chunk(&self, req: &WindowReq) -> Result<PointChunk> {
        let query = PointQuery::new(
            // levels 0 through ceil(log2(spacing / resolution)), as copc levels add density
            LodSelection::Resolution(req.resolution),
            BoundsSelection::Within(Bounds::new(
                (req.bbox.min_x, req.bbox.min_y, self.min_z),
                (req.bbox.max_x, req.bbox.max_y, self.max_z),
            )),
        );
        let mut reader = self.take_reader()?;
        // a reader marks a hierarchy page loaded before fetching it, so a failed read drops it
        let read = reader.read_points(query).map_err(|e| self.named(e))?;
        self.return_reader(reader);
        let points = read
            .into_iter()
            .filter(|point| tile_contains(&req.bbox, point.x, point.y))
            .map(|point| Point3 {
                x: point.x,
                y: point.y,
                z: point.z,
                intensity: point.intensity,
                classification: Classification::from_u8(u8::from(point.classification)),
            })
            .collect();
        Ok(PointChunk {
            points: PointCloud::from_points(points),
            bbox: req.bbox,
            resolution: req.resolution,
            crs: self.crs,
        })
    }
}

impl<S: RangeRead + Send + 'static> Source for CopcSrc<S> {
    fn identity(&self) -> String {
        let dataset = &self.dataset;
        let file = match &dataset.path {
            Some(path) => file_stamp(path),
            None => "file=none".into(),
        };
        format!(
            "CopcSrc name={:?} header={:?} info={:?} base_resolution={:?} crs={} {file}",
            dataset.name, dataset.header, dataset.info, dataset.base_resolution, dataset.crs
        )
    }

    fn constraint(&self) -> Constraint {
        Constraint::Produces(CapsSet::one(CapsPattern::PointCloud(PointPattern {
            crs: SetField::one(self.dataset.crs),
            resolution: ResRange::at_least(self.dataset.base_resolution),
            chunk_px: SetField::Any,
        })))
    }

    fn grid(&self) -> GridSpec {
        GridSpec {
            origin_x: self.dataset.origin_x,
            origin_y: self.dataset.origin_y,
            base_resolution: self.dataset.base_resolution,
            chunk_px: 256,
        }
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<Chunk>> {
        let dataset = self.dataset.clone();
        let req = *req;
        Box::pin(async move {
            crate::engine::offload(move || dataset.read_chunk(&req).map(Chunk::PointCloud)).await
        })
    }
}

// copc-reader's byte source over the cog transport, with its retries and range cache
#[derive(Clone)]
pub struct CopcHttpRange {
    range: Arc<HttpRange>,
    len: u64,
}

impl CopcHttpRange {
    // one HEAD request for the length, blocking
    pub fn new(url: &str) -> Result<Self> {
        let range = HttpRange::new(url);
        let len = range.content_length()?;
        Ok(CopcHttpRange {
            range: Arc::new(range),
            len,
        })
    }
}

impl RangeRead for CopcHttpRange {
    fn len(&mut self) -> copc_core::Result<u64> {
        Ok(self.len)
    }

    fn read_range(&mut self, offset: u64, buf: &mut [u8]) -> copc_core::Result<()> {
        // the transport asks for an inclusive end byte, which an empty range has none of
        if buf.is_empty() {
            return Ok(());
        }
        let bytes = terrano_core::RangeRead::read_range(&*self.range, offset, buf.len() as u64)
            .map_err(|e| copc_core::Error::io("fetch copc range", std::io::Error::other(e)))?;
        buf.copy_from_slice(&bytes);
        Ok(())
    }
}
