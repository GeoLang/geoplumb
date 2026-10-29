use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode, header};
use axum::routing::get;
use copc_reader::{CopcRangeReader, PointQuery, RangeRead};
use geoplumb::element::Source;
use geoplumb::elements::{CopcSrc, LasSrc};
use geoplumb::window::GridSpec;
use geoplumb::{Bbox, Crs, WindowReq};
use nubis_core::{Point3, PointCloud, write_las};

const COLUMNS: usize = 640;
const ROWS: usize = 640;
// a node's chunk outgrows the 64 KiB gap under which copc-reader merges two
// chunk fetches into one, and the lattice still splits over three levels
const MAX_POINTS_PER_NODE: u32 = 20_000;
const CRS: Crs = Crs(3857);

fn surface(x: f64, y: f64) -> f64 {
    40.0 + 0.08 * x - 0.05 * y + 3.0 * (0.3 * x).sin()
}

// one point per cell center of a unit lattice
fn lattice() -> PointCloud {
    let mut points = Vec::with_capacity(COLUMNS * ROWS);
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let (x, y) = (column as f64 + 0.5, row as f64 + 0.5);
            points.push(Point3::new(x, y, surface(x, y)));
        }
    }
    PointCloud::from_points(points)
}

struct Fixture {
    cloud: PointCloud,
    copc: Arc<Vec<u8>>,
}

// built once, the conversion is the slow part of every test here
static FIXTURE: LazyLock<Fixture> = LazyLock::new(|| {
    let dir = tempfile::tempdir().unwrap();
    let cloud = lattice();
    let las_path = dir.path().join("lattice.las");
    write_las(&cloud, &mut std::fs::File::create(&las_path).unwrap()).unwrap();
    let copc_path = dir.path().join("lattice.copc.laz");
    copc_writer::convert_las_to_copc_streaming(
        &las_path,
        &copc_path,
        &copc_writer::CopcWriterParams::new(MAX_POINTS_PER_NODE),
        dir.path(),
        &copc_core::NeverCancel,
    )
    .unwrap();
    Fixture {
        cloud,
        copc: Arc::new(std::fs::read(&copc_path).unwrap()),
    }
});

fn write_copc(dir: &Path) -> PathBuf {
    let path = dir.join("lattice.copc.laz");
    std::fs::write(&path, FIXTURE.copc.as_slice()).unwrap();
    path
}

fn copc_reader() -> CopcRangeReader<RecordingRange> {
    CopcRangeReader::open(RecordingRange::new()).unwrap()
}

fn deepest_level() -> i32 {
    copc_reader()
        .hierarchy_for(PointQuery::all())
        .unwrap()
        .iter()
        .filter(|entry| entry.has_point_data())
        .map(|entry| entry.key.level)
        .max()
        .unwrap()
}

fn tile(grid: &GridSpec, ladder_level: i32, column: i64, row: i64) -> WindowReq {
    let resolution = grid.base_resolution * 2f64.powi(ladder_level);
    let side = resolution * f64::from(grid.chunk_px);
    let max_y = grid.origin_y - row as f64 * side;
    let min_x = grid.origin_x + column as f64 * side;
    WindowReq {
        bbox: Bbox {
            min_x,
            max_x: min_x + side,
            min_y: max_y - side,
            max_y,
        },
        resolution,
        time: None,
    }
}

async fn pull(src: &impl Source, req: &WindowReq) -> Vec<Point3> {
    let mut points = src
        .read(req)
        .await
        .unwrap()
        .into_points()
        .unwrap()
        .points
        .points()
        .to_vec();
    points.sort_by(|a, b| {
        a.x.total_cmp(&b.x)
            .then(a.y.total_cmp(&b.y))
            .then(a.z.total_cmp(&b.z))
    });
    points
}

fn point_keys(points: &[Point3]) -> HashSet<(u64, u64, u64)> {
    points
        .iter()
        .map(|p| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits()))
        .collect()
}

#[test]
fn fixture_spans_at_least_three_octree_levels() {
    assert!(deepest_level() >= 2);
}

#[tokio::test]
async fn base_level_tile_matches_the_resident_las_source() {
    let dir = tempfile::tempdir().unwrap();
    let copc = CopcSrc::open_path(write_copc(dir.path()), CRS).unwrap();
    let las = LasSrc::new(FIXTURE.cloud.clone(), CRS).unwrap();
    let req = tile(&copc.grid(), 0, 1, 1);
    // a point within one quantisation step of an edge could land on either side
    let scale = copc_reader().header().x_scale_factor;
    for point in FIXTURE.cloud.points() {
        let b = &req.bbox;
        for distance in [
            point.x - b.min_x,
            point.x - b.max_x,
            point.y - b.min_y,
            point.y - b.max_y,
        ] {
            assert!(distance.abs() > scale, "lattice point sits on a tile edge");
        }
    }

    let from_copc = pull(&copc, &req).await;
    let from_las = pull(&las, &req).await;
    assert!(!from_copc.is_empty());
    assert_eq!(from_copc.len(), from_las.len());
    for (a, b) in from_copc.iter().zip(&from_las) {
        for (axis, c, l) in [("x", a.x, b.x), ("y", a.y, b.y), ("z", a.z, b.z)] {
            assert!((c - l).abs() <= scale, "{axis}: copc {c} vs las {l}");
        }
    }
}

#[tokio::test]
async fn coarse_tile_reads_a_subset_of_the_base_level() {
    let dir = tempfile::tempdir().unwrap();
    let copc = CopcSrc::open_path(write_copc(dir.path()), CRS).unwrap();
    let coarse = tile(&copc.grid(), deepest_level() - 1, 0, 0);
    let fine = WindowReq {
        resolution: copc.grid().base_resolution,
        ..coarse
    };
    let coarse_points = pull(&copc, &coarse).await;
    let fine_points = pull(&copc, &fine).await;
    assert!(!coarse_points.is_empty());
    assert!(
        coarse_points.len() < fine_points.len(),
        "coarse {} of fine {}",
        coarse_points.len(),
        fine_points.len()
    );
    let fine_keys = point_keys(&fine_points);
    for point in &coarse_points {
        assert!(
            fine_keys.contains(&(point.x.to_bits(), point.y.to_bits(), point.z.to_bits())),
            "coarse point {point:?} is not in the base level"
        );
    }
}

// the fixture's bytes, counting every byte a reader asks for
#[derive(Clone)]
struct RecordingRange {
    bytes: Arc<Vec<u8>>,
    fetched: Arc<AtomicU64>,
}

impl RecordingRange {
    fn new() -> Self {
        RecordingRange {
            bytes: FIXTURE.copc.clone(),
            fetched: Arc::new(AtomicU64::new(0)),
        }
    }
}

impl RangeRead for RecordingRange {
    fn len(&mut self) -> copc_core::Result<u64> {
        Ok(self.bytes.len() as u64)
    }

    fn read_range(&mut self, offset: u64, buf: &mut [u8]) -> copc_core::Result<()> {
        self.fetched.fetch_add(buf.len() as u64, Ordering::SeqCst);
        let start = offset as usize;
        let end = start + buf.len();
        let range = self.bytes.get(start..end).ok_or_else(|| {
            copc_core::Error::InvalidInput(format!("range {start}..{end} past the file"))
        })?;
        buf.copy_from_slice(range);
        Ok(())
    }
}

async fn bytes_fetched(req: impl Fn(&GridSpec) -> WindowReq) -> u64 {
    let range = RecordingRange::new();
    let fetched = range.fetched.clone();
    let src = CopcSrc::open(range, CRS).unwrap();
    let before = fetched.load(Ordering::SeqCst);
    pull(&src, &req(&src.grid())).await;
    fetched.load(Ordering::SeqCst) - before
}

#[tokio::test]
async fn a_tile_pull_fetches_only_the_nodes_it_touches() {
    let file_len = FIXTURE.copc.len() as u64;
    let base_tile = bytes_fetched(|grid| tile(grid, 0, 1, 1)).await;
    assert!(
        base_tile * 2 < file_len,
        "one base tile fetched {base_tile} of {file_len} bytes"
    );

    let coarse_level = deepest_level() - 1;
    let coarse = bytes_fetched(|grid| tile(grid, coarse_level, 0, 0)).await;
    let fine_same_window = bytes_fetched(|grid| WindowReq {
        resolution: grid.base_resolution,
        ..tile(grid, coarse_level, 0, 0)
    })
    .await;
    assert!(
        coarse < fine_same_window,
        "coarse pull fetched {coarse}, the same window at the base level {fine_same_window}"
    );
}

async fn range_handler(
    State(bytes): State<Arc<Vec<u8>>>,
    headers: HeaderMap,
) -> (StatusCode, Vec<u8>) {
    let Some(range) = headers.get(header::RANGE) else {
        return (StatusCode::OK, bytes.as_slice().to_vec());
    };
    let spec = range.to_str().unwrap().trim_start_matches("bytes=");
    let (start, end) = spec.split_once('-').unwrap();
    let (start, end): (usize, usize) = (start.parse().unwrap(), end.parse().unwrap());
    (
        StatusCode::PARTIAL_CONTENT,
        bytes[start..=end.min(bytes.len() - 1)].to_vec(),
    )
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn remote_tile_matches_the_local_file() {
    let app = axum::Router::new()
        .route("/lattice.copc.laz", get(range_handler))
        .with_state(FIXTURE.copc.clone());
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

    let url = format!("http://{address}/lattice.copc.laz");
    let remote = tokio::task::spawn_blocking(move || CopcSrc::open_url(url, CRS))
        .await
        .unwrap()
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let local = CopcSrc::open_path(write_copc(dir.path()), CRS).unwrap();
    let req = tile(&local.grid(), 0, 1, 1);
    let from_remote = pull(&remote, &req).await;
    assert!(!from_remote.is_empty());
    assert_eq!(from_remote, pull(&local, &req).await);
}
