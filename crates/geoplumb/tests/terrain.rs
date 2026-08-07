//! aspect: the angle a known plane comes back with, flat ground coming back
//! as nodata, and seam equality against the whole-window reference

use geoplumb::elements::{Aspect, RasterSrc};
use geoplumb::{Bbox, Crs, Engine, Graph, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

/// the window every test pulls, spanning the chunk border at pixel 256 in
/// both axes so a chunked pull has seams to get wrong
const PX0: usize = 20;
const PY0: usize = 20;
const COLS: usize = 320;
const ROWS: usize = 210;

/// halo the aspect plan asks for, so a reference patch can carry the same
const PAD: usize = 4;

fn dem_from(elevation: impl Fn(usize, usize) -> f64) -> Raster {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            data.push(elevation(col, row));
        }
    }
    Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
}

fn src_from(elevation: impl Fn(usize, usize) -> f64) -> RasterSrc {
    RasterSrc::new(
        BandedRaster::new(vec![dem_from(elevation)]).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

/// an alpine-ish surface, aspect turning through every quadrant over it
fn rolling(col: usize, row: usize) -> f64 {
    let lon = ORIGIN_X + (col as f64 + 0.5) * CELL;
    let lat = ORIGIN_Y - (row as f64 + 0.5) * CELL;
    500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos()
}

fn test_window() -> Bbox {
    Bbox {
        min_x: ORIGIN_X + PX0 as f64 * CELL,
        max_x: ORIGIN_X + (PX0 + COLS) as f64 * CELL,
        max_y: ORIGIN_Y - PY0 as f64 * CELL,
        min_y: ORIGIN_Y - (PY0 + ROWS) as f64 * CELL,
    }
}

async fn pull_aspect(source: RasterSrc) -> Vec<f64> {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(source));
    let aspect = g.add_transform(s, Box::new(Aspect));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let chunk = engine
        .pull(
            aspect,
            WindowReq {
                bbox: test_window(),
                resolution: CELL,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let band = chunk.bands.band(0).unwrap();
    assert_eq!((band.width(), band.height()), (COLS, ROWS));
    band.data().to_vec()
}

async fn assert_uniform_aspect(elevation: fn(usize, usize) -> f64, want: f64) {
    for (i, got) in pull_aspect(src_from(elevation)).await.iter().enumerate() {
        assert_eq!(*got, want, "cell {i}");
    }
}

#[tokio::test]
async fn a_plane_faces_one_way_over_its_whole_window() {
    // terrano measures the angle counterclockwise from east, so a surface
    // rising eastward reads 180 and one rising northward reads 270
    assert_uniform_aspect(|col, _| col as f64 * 10.0, 180.0).await;
    assert_uniform_aspect(|_, row| (H - row) as f64 * 10.0, 270.0).await;
}

#[tokio::test]
async fn flat_ground_has_no_aspect_and_stays_nodata() {
    for (i, got) in pull_aspect(src_from(|_, _| 100.0)).await.iter().enumerate() {
        assert!(got.is_nan(), "flat cell {i} came back as {got}");
    }
}

#[tokio::test]
async fn aspect_is_seam_free_across_chunks() {
    let data = pull_aspect(src_from(rolling)).await;

    // reference: aspect over the padded window in one piece, pad cropped off
    let patch_cols = COLS + 2 * PAD;
    let patch_rows = ROWS + 2 * PAD;
    let mut patch = Vec::with_capacity(patch_cols * patch_rows);
    for row in 0..patch_rows {
        for col in 0..patch_cols {
            patch.push(rolling(PX0 - PAD + col, PY0 - PAD + row));
        }
    }
    let patch = Raster::from_vec(patch_cols, patch_rows, patch, CELL, f64::NAN).unwrap();
    let reference = terrano_core::aspect(&patch);

    for row in 0..ROWS {
        for col in 0..COLS {
            let got = data[row * COLS + col];
            let want = reference.data()[(row + PAD) * patch_cols + (col + PAD)];
            assert!(
                (got - want).abs() < 1e-9,
                "seam at ({col},{row}): chunked {got} vs reference {want}"
            );
        }
    }
}
