//! band math: an ndvi expression against a hand-computed reference, chunk
//! seam equality, nan propagation, parse failures and the band count the
//! expression demands of its input link

use geoplumb::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use geoplumb::element::Transform;
use geoplumb::elements::{BandMath, RasterSrc};
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 320;
const H: usize = 256;
// binary exact, so pixel-aligned windows survive outward alignment
const CELL: f64 = 1.0 / 1024.0;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;

/// red and near infrared, both well away from zero so an ndvi denominator
/// never vanishes
fn band_value(band: usize, col: usize, row: usize) -> f64 {
    let (x, y) = (col as f64, row as f64);
    match band {
        0 => 0.2 + 0.1 * (x * 0.11).sin() * (y * 0.07).cos(),
        _ => 0.6 + 0.2 * ((x + y) * 0.05).sin(),
    }
}

fn ndvi(col: usize, row: usize) -> f64 {
    let (red, nir) = (band_value(0, col, row), band_value(1, col, row));
    (nir - red) / (nir + red)
}

fn src() -> RasterSrc {
    src_with(|_, _, _| false)
}

/// two bands, with `hole` naming the cells written as nodata
fn src_with(hole: impl Fn(usize, usize, usize) -> bool) -> RasterSrc {
    let bands: Vec<Raster> = (0..2)
        .map(|b| {
            let mut data = Vec::with_capacity(W * H);
            for row in 0..H {
                for col in 0..W {
                    data.push(if hole(b, col, row) {
                        f64::NAN
                    } else {
                        band_value(b, col, row)
                    });
                }
            }
            Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap()
        })
        .collect();
    RasterSrc::new(
        BandedRaster::new(bands).unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )
}

fn window(px0: usize, py0: usize, px1: usize, py1: usize) -> Bbox {
    Bbox {
        min_x: ORIGIN_X + px0 as f64 * CELL,
        max_x: ORIGIN_X + px1 as f64 * CELL,
        max_y: ORIGIN_Y - py0 as f64 * CELL,
        min_y: ORIGIN_Y - py1 as f64 * CELL,
    }
}

async fn pull(expr: &str, source: RasterSrc, bbox: Bbox) -> RasterChunk {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(source));
    let bm = g.add_transform(s, Box::new(BandMath::new(expr).unwrap()));
    let engine = Engine::new(g, 64 << 20).unwrap();
    engine
        .pull(
            bm,
            WindowReq {
                bbox,
                resolution: CELL,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap()
}

#[tokio::test]
async fn ndvi_matches_the_hand_computed_reference() {
    let (px0, py0) = (16usize, 24usize);
    let (cols, rows) = (64usize, 48usize);
    let got = pull(
        "(b1 - b0) / (b1 + b0)",
        src(),
        window(px0, py0, px0 + cols, py0 + rows),
    )
    .await;

    assert_eq!(got.bands.band_count(), 1);
    assert_eq!((got.width(), got.height()), (cols, rows));
    let band = got.bands.band(0).unwrap();
    for row in 0..rows {
        for col in 0..cols {
            let want = ndvi(px0 + col, py0 + row);
            let a = band.data()[row * cols + col];
            assert!(
                (a - want).abs() < 1e-12,
                "cell ({col},{row}): got {a} vs reference {want}"
            );
        }
    }
}

#[test]
fn output_is_one_band_and_keeps_the_input_crs() {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(src()));
    let bm = g.add_transform(s, Box::new(BandMath::new("b0 * 2 - b1").unwrap()));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(s).raster().bands, 2);
    let caps = engine.caps(bm).raster();
    assert_eq!(caps.bands, 1);
    assert_eq!(caps.crs, Crs::WGS84);
}

/// raster identity demanding small tiles, so the same window comes back
/// stitched from many chunks instead of one
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
async fn chunked_pull_matches_the_whole_window() {
    const EXPR: &str = "sqrt(abs(b1 - b0)) + min(b0, b1) / max(b0, b1)";
    // 5x4 tiles of 16 px once the chunk demand lands, one piece without it
    let bbox = window(32, 32, 112, 96);
    let whole = pull(EXPR, src(), bbox).await;

    let mut g = Graph::new();
    let s = g.add_source(Box::new(src()));
    let bm = g.add_transform(s, Box::new(BandMath::new(EXPR).unwrap()));
    g.add_transform(bm, Box::new(SmallChunks));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(bm).raster().chunk_px, 16);
    let chunked = engine
        .pull(
            bm,
            WindowReq {
                bbox,
                resolution: CELL,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap();

    assert_eq!((chunked.width(), chunked.height()), (80, 64));
    let bits = |c: &RasterChunk| {
        c.bands
            .band(0)
            .unwrap()
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    assert_eq!(
        bits(&chunked),
        bits(&whole),
        "chunked band math differs from the whole-window computation"
    );
}

#[tokio::test]
async fn nan_propagates_through_every_operator() {
    // a nodata cell in the red band only, so min and max still have a
    // finite operand to prefer
    let hole = |band: usize, col: usize, row: usize| band == 0 && col == 40 && row == 30;
    let bbox = window(32, 24, 48, 40);
    let (hole_col, hole_row) = (8usize, 6usize);

    for expr in [
        "b0 + b1",
        "b0 * b1",
        "min(b0, b1)",
        "max(b0, b1)",
        "pow(b0, 0)",
        "-b0",
    ] {
        let got = pull(expr, src_with(hole), bbox).await;
        let band = got.bands.band(0).unwrap();
        let at = |col: usize, row: usize| band.data()[row * 16 + col];
        assert!(
            at(hole_col, hole_row).is_nan(),
            "{expr}: nodata cell came back as {}",
            at(hole_col, hole_row)
        );
        assert!(
            at(hole_col + 1, hole_row).is_finite(),
            "{expr}: nan leaked into the neighbouring cell"
        );
    }
}

#[test]
fn parse_failures_name_the_offending_token() {
    for (expr, needle) in [
        ("b0 % b1", "'%'"),
        ("(b0 + b1", "')'"),
        ("foo(b0)", "unknown function foo"),
        ("b0 + bx", "unknown name bx"),
        ("min(b0)", "min takes 2 arguments"),
        ("b0 b1", "trailing b1"),
    ] {
        let err = BandMath::new(expr)
            .err()
            .unwrap_or_else(|| panic!("{expr}"));
        let msg = err.to_string();
        assert!(msg.contains(needle), "{expr}: unexpected error {msg}");
    }
}

#[test]
fn a_band_the_source_lacks_fails_the_engine_build() {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(src()));
    g.add_transform(s, Box::new(BandMath::new("b0 + b2").unwrap()));
    let err = Engine::new(g, 64 << 20)
        .err()
        .expect("two bands cannot feed an expression naming b2");
    let msg = err.to_string();
    assert!(
        msg.contains("reads 3 bands") && msg.contains("carries 2"),
        "unexpected error: {msg}"
    );
}

#[test]
fn extra_source_bands_are_fine() {
    let three = BandedRaster::new(
        (0..3)
            .map(|_| Raster::from_vec(4, 4, vec![1.0; 16], CELL, f64::NAN).unwrap())
            .collect(),
    )
    .unwrap();
    let mut g = Graph::new();
    let s = g.add_source(Box::new(RasterSrc::new(
        three,
        ORIGIN_X,
        ORIGIN_Y,
        Crs::WGS84,
    )));
    let bm = g.add_transform(s, Box::new(BandMath::new("b0 + b1").unwrap()));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(bm).raster().bands, 1);
}
