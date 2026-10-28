//! point cloud chunks end to end: negotiation across the kind boundary,
//! point pulls with thinning, idw gridding seam-equality, spill reload

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use futures::future::BoxFuture;
use geoplumb::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use geoplumb::chunk::tile_contains;
use geoplumb::element::{Source, Transform};
use geoplumb::elements::{Hillshade, IdwGrid, LasSrc};
use geoplumb::window::ChunkKey;
use geoplumb::{Bbox, Caps, Chunk, Crs, Engine, Error, Graph, WindowReq};
use nubis_core::{Point3, PointCloud};

const W: usize = 64;
const H: usize = 64;

fn plane(x: f64, y: f64) -> f64 {
    40.0 + 0.08 * x - 0.05 * y
}

/// one point per cell center of a unit lattice, z on a gentle plane
fn lattice_cloud(z: impl Fn(f64, f64) -> f64) -> PointCloud {
    lattice(W, H, z)
}

fn lattice(columns: usize, rows: usize, z: impl Fn(f64, f64) -> f64) -> PointCloud {
    let mut pts = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for col in 0..columns {
            let (x, y) = (col as f64 + 0.5, row as f64 + 0.5);
            pts.push(Point3::new(x, y, z(x, y)));
        }
    }
    PointCloud::from_points(pts)
}

fn idw() -> IdwGrid {
    IdwGrid::default()
}

#[test]
fn point_chain_negotiates_across_the_kind_boundary() {
    let mut g = Graph::new();
    let las = g.add_source(Box::new(
        LasSrc::new(lattice_cloud(plane), Crs::WGS84).unwrap(),
    ));
    let grid = g.add_transform(las, Box::new(idw()));
    let hs = g.add_transform(grid, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();

    let point_caps = engine.caps(las).point();
    assert_eq!(point_caps.crs, Crs::WGS84);
    let raster_caps = engine.caps(grid).raster();
    assert_eq!(
        raster_caps.crs,
        Crs::WGS84,
        "crs passes through the gridder"
    );
    assert_eq!(raster_caps.bands, 1);
    assert_eq!(engine.caps(hs).raster().crs, Crs::WGS84);
}

#[test]
fn point_source_cannot_feed_a_raster_consumer_directly() {
    let mut g = Graph::new();
    let las = g.add_source(Box::new(
        LasSrc::new(lattice_cloud(plane), Crs::WGS84).unwrap(),
    ));
    g.add_transform(las, Box::new(Hillshade::new(315.0, 45.0)));
    match Engine::new(g, 64 << 20) {
        Err(Error::EmptyLink { .. }) => {}
        Err(other) => panic!("expected EmptyLink, got {other:?}"),
        Ok(_) => panic!("kind mismatch must fail negotiation"),
    }
}

#[tokio::test]
async fn point_pull_filters_the_window_and_thins_coarse_levels() {
    let mut g = Graph::new();
    let las = g.add_source(Box::new(
        LasSrc::new(lattice_cloud(plane), Crs::WGS84).unwrap(),
    ));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let base = engine.grid(las).base_resolution;

    let bbox = Bbox::new(0.0, 0.0, 64.0, 64.0);
    let fine = engine
        .pull(
            las,
            WindowReq {
                bbox,
                resolution: base,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_points()
        .unwrap();
    assert_eq!(fine.points.len(), W * H, "level 0 keeps every point");

    let coarse = engine
        .pull(
            las,
            WindowReq {
                bbox,
                resolution: base * 4.0,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_points()
        .unwrap();
    assert!(
        coarse.points.len() < W * H / 4,
        "coarse level should thin, kept {}",
        coarse.points.len()
    );
    assert!(!coarse.points.is_empty());
}

/// identity transform that narrows the negotiated chunk size, forcing the
/// idw node onto small tiles so the seam test crosses chunk borders
struct SmallChunks;

impl Transform for SmallChunks {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Raster(RasterPattern {
            chunk_px: SetField::one(16),
            ..RasterPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        Ok(Chunk::Raster(input.raster()?.crop_to(&out.bbox)))
    }
}

#[tokio::test]
async fn chunked_idw_matches_the_whole_window_reference() {
    let cloud = lattice_cloud(plane);
    let mut g = Graph::new();
    let las = g.add_source(Box::new(LasSrc::new(cloud.clone(), Crs::WGS84).unwrap()));
    let grid = g.add_transform(las, Box::new(idw()));
    g.add_transform(grid, Box::new(SmallChunks));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(grid).raster().chunk_px, 16);

    let base = engine.grid(grid).base_resolution;
    let got = engine
        .pull(
            grid,
            WindowReq {
                bbox: Bbox::new(2.0, 2.0, 60.0, 60.0),
                resolution: base,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap();

    let res = got.resolution;
    let (cols, rows) = (got.width(), got.height());
    let params = idw();
    let reference = nubis_core::idw_window(
        &cloud,
        &nubis_core::GridWindow {
            origin_x: got.bbox.min_x + 0.5 * res,
            origin_y: got.bbox.min_y + 0.5 * res,
            width: cols,
            height: rows,
            cell_size: res,
        },
        params.power,
        params.radius_px * res,
        params.min_points,
    )
    .unwrap();

    let band = got.bands.band(0).unwrap();
    for row in 0..rows {
        for col in 0..cols {
            let a = band.data()[row * cols + col];
            let r = reference.data[(rows - 1 - row) * cols + col];
            let r = if r == reference.nodata { f64::NAN } else { r };
            assert!(
                (a - r).abs() < 1e-9 || (a.is_nan() && r.is_nan()),
                "cell ({col},{row}): chunked {a} vs whole-window {r}"
            );
        }
    }
}

#[tokio::test]
async fn constant_cloud_hillshades_flat() {
    let mut g = Graph::new();
    let las = g.add_source(Box::new(
        LasSrc::new(lattice_cloud(|_, _| 42.0), Crs::WGS84).unwrap(),
    ));
    let grid = g.add_transform(las, Box::new(idw()));
    let hs = g.add_transform(grid, Box::new(Hillshade::new(315.0, 45.0)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let base = engine.grid(hs).base_resolution;
    let got = engine
        .pull(
            hs,
            WindowReq {
                bbox: Bbox::new(8.0, 8.0, 56.0, 56.0),
                resolution: base,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let band = got.bands.band(0).unwrap();
    let flat: Vec<f64> = band
        .data()
        .iter()
        .copied()
        .filter(|v| !v.is_nan())
        .collect();
    assert!(!flat.is_empty());
    let first = flat[0];
    assert!(
        flat.iter().all(|v| (v - first).abs() < 1e-9),
        "flat terrain must shade uniformly"
    );
}

/// counts source reads so the spill test can tell a disk reload from a
/// recompute
struct CountingLas {
    inner: LasSrc,
    reads: Arc<AtomicUsize>,
}

impl Source for CountingLas {
    fn constraint(&self) -> Constraint {
        self.inner.constraint()
    }

    fn grid(&self) -> geoplumb::window::GridSpec {
        self.inner.grid()
    }

    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, geoplumb::Result<Chunk>> {
        self.reads.fetch_add(1, Ordering::SeqCst);
        self.inner.read(req)
    }
}

#[tokio::test]
async fn point_chunks_spill_to_disk_and_reload() {
    let reads = Arc::new(AtomicUsize::new(0));
    let mut g = Graph::new();
    let las = g.add_source(Box::new(CountingLas {
        inner: LasSrc::new(lattice_cloud(plane), Crs::WGS84).unwrap(),
        reads: reads.clone(),
    }));
    // level 0 chunk is ~128 KiB of points, the budget holds it alone but
    // not together with the next pull, forcing a demotion to disk
    let engine = Engine::with_disk_cache(g, 130 << 10, std::env::temp_dir(), 64 << 20).unwrap();
    let base = engine.grid(las).base_resolution;
    let bbox = Bbox::new(0.0, 0.0, 64.0, 64.0);
    let fine = WindowReq {
        bbox,
        resolution: base,
        time: None,
    };
    let coarse = WindowReq {
        bbox,
        resolution: base * 4.0,
        time: None,
    };

    let first = engine.pull(las, fine).await.unwrap().into_points().unwrap();
    engine.pull(las, coarse).await.unwrap();
    let after = reads.load(Ordering::SeqCst);
    let again = engine.pull(las, fine).await.unwrap().into_points().unwrap();
    assert_eq!(
        reads.load(Ordering::SeqCst),
        after,
        "spilled chunk must reload from disk, not recompute"
    );

    let key = |p: &Point3| (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
    let mut a: Vec<_> = first.points.points().iter().map(key).collect();
    let mut b: Vec<_> = again.points.points().iter().map(key).collect();
    a.sort_unstable();
    b.sort_unstable();
    assert_eq!(a, b, "reloaded points differ from the computed ones");
}

/// raster identity that insists on web mercator, downstream of the gridder
struct MercatorOnly;

impl Transform for MercatorOnly {
    fn constraint(&self) -> Constraint {
        Constraint::Identity(CapsSet::one(CapsPattern::Raster(RasterPattern {
            crs: SetField::one(Crs::WEB_MERCATOR),
            ..RasterPattern::default()
        })))
    }

    fn plan(&self, out: &WindowReq) -> WindowReq {
        *out
    }

    fn compute(&self, out: &WindowReq, input: &Chunk) -> geoplumb::Result<Chunk> {
        Ok(Chunk::Raster(input.raster()?.crop_to(&out.bbox)))
    }
}

#[test]
fn crs_demand_downstream_of_the_gridder_autoplugs_a_reproject() {
    let mut g = Graph::new();
    let las = g.add_source(Box::new(
        LasSrc::new(lattice_cloud(plane), Crs::WGS84).unwrap(),
    ));
    let grid = g.add_transform(las, Box::new(idw()));
    let t = g.add_transform(grid, Box::new(MercatorOnly));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert!(matches!(engine.caps(las), Caps::PointCloud(_)));
    assert_eq!(engine.caps(grid).raster().crs, Crs::WGS84);
    assert_eq!(engine.caps(t).raster().crs, Crs::WEB_MERCATOR);
}

fn brute_force(cloud: &PointCloud, bbox: &Bbox) -> Vec<Point3> {
    cloud
        .points()
        .iter()
        .filter(|p| tile_contains(bbox, p.x, p.y))
        .copied()
        .collect()
}

fn chunk_key(level: u8, ix: i64, iy: i64) -> ChunkKey {
    ChunkKey {
        level,
        ix,
        iy,
        time: None,
    }
}

#[test]
fn tile_index_candidates_keep_every_point_a_full_scan_keeps() {
    let cloud = lattice(1024, 1024, plane);
    let las = LasSrc::new(cloud.clone(), Crs::WGS84).unwrap();
    let grid = las.grid();
    let base_tile = grid.chunk_ground_size(0);
    let aligned = grid.chunk_bbox(chunk_key(0, 1, 1));
    let straddling_seam = Bbox::new(
        aligned.min_x + 0.5 * base_tile,
        aligned.min_y + 0.5 * base_tile,
        aligned.max_x + 0.5 * base_tile,
        aligned.max_y + 0.5 * base_tile,
    );
    let bboxes = [
        aligned,
        straddling_seam,
        grid.chunk_bbox(chunk_key(0, 3, 3)),
        grid.chunk_bbox(chunk_key(1, 1, 0)),
        grid.chunk_bbox(chunk_key(2, 0, 0)),
        grid.chunk_bbox(chunk_key(0, -2, -2)),
        grid.chunk_bbox(chunk_key(1, 5, 5)),
        Bbox::new(-1e12, aligned.min_y, 1e12, aligned.max_y),
    ];
    for bbox in bboxes {
        let indexed: Vec<Point3> = las
            .candidates(&bbox)
            .filter(|p| tile_contains(&bbox, p.x, p.y))
            .copied()
            .collect();
        assert_eq!(indexed, brute_force(&cloud, &bbox), "bbox {bbox:?}");
    }
}

#[test]
fn tile_index_scans_only_the_bins_a_tile_touches() {
    let cloud = lattice(2048, 2048, plane);
    let las = LasSrc::new(cloud.clone(), Crs::WGS84).unwrap();
    let grid = las.grid();
    let base_tile = grid.chunk_ground_size(0);
    let (min, max) = cloud.bounds().unwrap();
    assert!(max.x - min.x >= 8.0 * base_tile && max.y - min.y >= 8.0 * base_tile);

    let bbox = grid.chunk_bbox(chunk_key(0, 3, 4));
    let scanned = las.candidates(&bbox).count();
    let inside = brute_force(&cloud, &bbox).len();
    assert!(inside > 0);
    assert!(scanned >= inside, "scanned {scanned}, inside {inside}");
    assert!(
        scanned < cloud.len() / 4,
        "scanned {scanned} of {} points",
        cloud.len()
    );

    let row_spanning_more_bins_than_exist = Bbox::new(
        -1e12,
        bbox.min_y + 0.25 * base_tile,
        1e12,
        bbox.max_y - 0.25 * base_tile,
    );
    let scanned_row = las.candidates(&row_spanning_more_bins_than_exist).count();
    assert!(
        scanned_row < cloud.len() / 4,
        "scanned {scanned_row} of {} points",
        cloud.len()
    );
}

#[test]
fn inverted_bbox_yields_no_candidates_at_once() {
    let las = LasSrc::new(lattice_cloud(|_, _| 0.0), Crs::WGS84).unwrap();
    // min_x past max_x by several bins, with a y range that spans every row
    let bbox = Bbox::new(1.0e5, -1.0e300, 0.0, 1.0e300);
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let count = las.candidates(&bbox).count();
        let _ = sender.send(count);
    });
    let count = receiver
        .recv_timeout(std::time::Duration::from_secs(5))
        .expect("candidates did not return");
    assert_eq!(count, 0);
}
