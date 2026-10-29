mod gpkg_fixture;

use std::collections::HashMap;
use std::path::Path;

use geoplumb::element::Source;
use geoplumb::elements::{GpkgSrc, VecSrc};
use geoplumb::{Bbox, Crs, Engine, Graph, VectorChunk, WindowReq};
use gpkg_fixture::{FixtureFeature, Layer, Srs, TABLE, WGS84, envelope, write_geopackage};
use rusqlite::types::Value;
use topoi_core::geojson::{Feature, FeatureCollection, FeatureGeometry};
use topoi_core::{Coord, LineString, Polygon};

// 63 cells of side 2 every 33 units span 62 * 33 + 2 = 2048 units, so the
// extent's longer side over 1024 and the median segment length are both 2:
// GpkgSrc and VecSrc anchor the same grid
const CELLS_PER_SIDE: usize = 63;
const CELL_SIDE: f64 = 2.0;
const CELL_STRIDE: f64 = 33.0;
const ORIGIN_X: f64 = 500.0;
const ORIGIN_Y: f64 = 100.0;
const EXTENT_SIDE: f64 = 2048.0;
const BASE_RESOLUTION: f64 = 2.0;
const TILE_SIDE: f64 = 256.0 * BASE_RESOLUTION;
const LINE_EVERY: usize = 7;
const NULL_WEIGHT_EVERY: usize = 5;
const COLUMNS: &[(&str, &str)] = &[("name", "TEXT"), ("weight", "REAL")];

fn cell_geometry(index: usize) -> FeatureGeometry {
    let x = ORIGIN_X + (index % CELLS_PER_SIDE) as f64 * CELL_STRIDE;
    let y = ORIGIN_Y + (index / CELLS_PER_SIDE) as f64 * CELL_STRIDE;
    if index % LINE_EVERY == 0 {
        return FeatureGeometry::LineString(LineString::new(vec![
            Coord::new(x, y),
            Coord::new(x + CELL_SIDE, y),
            Coord::new(x + CELL_SIDE, y + CELL_SIDE),
        ]));
    }
    FeatureGeometry::Polygon(Polygon::from_coords(&[
        Coord::new(x, y),
        Coord::new(x + CELL_SIDE, y),
        Coord::new(x + CELL_SIDE, y + CELL_SIDE),
        Coord::new(x, y + CELL_SIDE),
        Coord::new(x, y),
    ]))
}

fn cell_weight(index: usize) -> Option<f64> {
    (index % NULL_WEIGHT_EVERY != 0).then_some(index as f64 * 0.25)
}

fn fixture_features() -> Vec<FixtureFeature> {
    (0..CELLS_PER_SIDE * CELLS_PER_SIDE)
        .map(|i| FixtureFeature {
            id: i as i64,
            geometry: cell_geometry(i),
            properties: vec![
                Value::Text(format!("cell {i}")),
                cell_weight(i).map_or(Value::Null, Value::Real),
            ],
        })
        .collect()
}

// the same features as a collection, in id order, so VecSrc numbers them
// the way the table's primary key does
fn fixture_collection() -> FeatureCollection {
    let features = (0..CELLS_PER_SIDE * CELLS_PER_SIDE)
        .map(|i| Feature {
            geometry: Some(cell_geometry(i)),
            properties: HashMap::from([
                ("name".to_string(), serde_json::json!(format!("cell {i}"))),
                ("weight".to_string(), serde_json::json!(cell_weight(i))),
            ]),
        })
        .collect();
    FeatureCollection { features }
}

fn write_fixture(dir: &Path, srs: Srs, rtree: bool) -> std::path::PathBuf {
    let path = dir.join("cells.gpkg");
    let features = fixture_features();
    write_geopackage(
        &path,
        &Layer {
            srs,
            columns: COLUMNS,
            features: &features,
            rtree,
        },
    );
    path
}

fn open_error(path: &Path, table: Option<&str>, crs: Option<Crs>) -> String {
    match GpkgSrc::open(path, table, crs) {
        Ok(_) => panic!("{} opened", path.display()),
        Err(e) => e.to_string(),
    }
}

fn request(bbox: Bbox) -> WindowReq {
    WindowReq {
        bbox,
        resolution: BASE_RESOLUTION,
        time: None,
    }
}

// the rtree's own test, inclusive on every side
fn touches(geometry: &FeatureGeometry, bbox: &Bbox) -> bool {
    let [min_x, max_x, min_y, max_y] = envelope(geometry);
    max_x >= bbox.min_x && min_x <= bbox.max_x && max_y >= bbox.min_y && min_y <= bbox.max_y
}

#[tokio::test]
async fn a_tile_read_decodes_only_the_rows_its_bbox_touches() {
    let dir = tempfile::tempdir().unwrap();
    let src = GpkgSrc::open(write_fixture(dir.path(), WGS84, true), None, None).unwrap();
    let grid = src.grid();
    assert_eq!(grid.base_resolution, BASE_RESOLUTION);
    let features = fixture_features();
    let tiles_per_side = (EXTENT_SIDE / TILE_SIDE) as usize;

    for row in 0..tiles_per_side {
        for column in 0..tiles_per_side {
            let min_x = grid.origin_x + column as f64 * TILE_SIDE;
            let max_y = grid.origin_y - row as f64 * TILE_SIDE;
            let bbox = Bbox::new(min_x, max_y - TILE_SIDE, min_x + TILE_SIDE, max_y);
            let expected = features
                .iter()
                .filter(|f| touches(&f.geometry, &bbox))
                .count() as u64;

            let before = src.rows_read();
            let chunk = src.read(&request(bbox)).await.unwrap();
            let rows = src.rows_read() - before;

            assert_eq!(rows, expected, "tile {column},{row}");
            assert!(
                rows * 10 < features.len() as u64,
                "tile {column},{row} read {rows} of {} rows",
                features.len()
            );
            assert!(!chunk.into_vector().unwrap().features.is_empty());
        }
    }
}

async fn whole_window(src: Box<dyn Source>) -> VectorChunk {
    let mut graph = Graph::new();
    let node = graph.add_source(src);
    let engine = Engine::new(graph, 64 << 20).unwrap();
    let grid = engine.grid(node);
    let bbox = Bbox::new(
        grid.origin_x,
        grid.origin_y - EXTENT_SIDE,
        grid.origin_x + EXTENT_SIDE,
        grid.origin_y,
    );
    engine
        .pull(node, request(bbox))
        .await
        .unwrap()
        .into_vector()
        .unwrap()
}

#[tokio::test]
async fn a_whole_window_pull_matches_vecsrc_fragment_for_fragment() {
    let dir = tempfile::tempdir().unwrap();
    let gpkg = GpkgSrc::open(write_fixture(dir.path(), WGS84, true), None, None).unwrap();
    let resident = VecSrc::new(fixture_collection(), Crs::WGS84).unwrap();
    assert_eq!(gpkg.grid(), resident.grid());

    let windowed = whole_window(Box::new(gpkg)).await;
    let resident = whole_window(Box::new(resident)).await;

    assert_eq!(windowed.features.len(), resident.features.len());
    for (w, r) in windowed.features.iter().zip(&resident.features) {
        assert_eq!(w.id, r.id);
        assert_eq!(w.geometry, r.geometry, "feature {}", w.id);
        assert_eq!(w.properties, r.properties, "feature {}", w.id);
    }

    let windowed = windowed.dissolve();
    let resident = resident.dissolve();
    let ids: Vec<u64> = windowed.features.iter().map(|f| f.id).collect();
    let expected: Vec<u64> = (0..(CELLS_PER_SIDE * CELLS_PER_SIDE) as u64).collect();
    assert_eq!(ids, expected, "every cell survives, once");
    for (w, r) in windowed.features.iter().zip(&resident.features) {
        assert_eq!(w.id, r.id);
        assert_eq!(w.geometry, r.geometry, "dissolved feature {}", w.id);
    }
}

#[test]
fn a_file_without_an_rtree_fails_at_open_naming_the_table() {
    let dir = tempfile::tempdir().unwrap();
    let error = open_error(&write_fixture(dir.path(), WGS84, false), None, None);
    assert!(error.contains(TABLE), "{error}");
    assert!(error.contains("rtree"), "{error}");
}

#[test]
fn a_table_not_in_gpkg_contents_fails_at_open() {
    let dir = tempfile::tempdir().unwrap();
    let error = open_error(&write_fixture(dir.path(), WGS84, true), Some("roads"), None);
    assert!(error.contains("roads"), "{error}");
}

#[tokio::test]
async fn a_non_epsg_srs_needs_a_crs_override() {
    let dir = tempfile::tempdir().unwrap();
    let local = Srs {
        id: 100_000,
        organization: "local",
        organization_coordsys_id: 1,
    };
    let path = write_fixture(dir.path(), local, true);

    let error = open_error(&path, None, None);
    assert!(error.contains("100000"), "{error}");

    let src = GpkgSrc::open(&path, None, Some(Crs::WEB_MERCATOR)).unwrap();
    let mut graph = Graph::new();
    let node = graph.add_source(Box::new(src));
    let engine = Engine::new(graph, 64 << 20).unwrap();
    assert_eq!(engine.caps(node).vector().crs, Crs::WEB_MERCATOR);
}

#[tokio::test]
async fn a_geometry_that_does_not_decode_fails_the_read_naming_the_row() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_fixture(dir.path(), WGS84, true);
    let broken_id = 1;
    rusqlite::Connection::open(&path)
        .unwrap()
        .execute(
            &format!("UPDATE \"{TABLE}\" SET geom = x'47500001' WHERE fid = ?1"),
            [broken_id],
        )
        .unwrap();

    let src = GpkgSrc::open(&path, None, None).unwrap();
    let grid = src.grid();
    let bbox = Bbox::new(
        grid.origin_x,
        grid.origin_y - EXTENT_SIDE,
        grid.origin_x + TILE_SIDE,
        grid.origin_y - EXTENT_SIDE + TILE_SIDE,
    );
    let error = src.read(&request(bbox)).await.unwrap_err().to_string();
    assert!(error.contains(TABLE), "{error}");
    assert!(error.contains(&format!("row {broken_id}")), "{error}");
}
