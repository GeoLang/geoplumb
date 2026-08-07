//! focal statistics: every op against a hand-computed window, the band count
//! kept, nodata neighbours excluded and nodata centres held, and a chunked
//! pull equal to the whole-window pull so the halo is proven to cross seams

use geoplumb::caps::{CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use geoplumb::element::Transform;
use geoplumb::elements::{Focal, FocalOp, RasterSrc};
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 320;
const H: usize = 256;
// binary exact, so pixel-aligned windows survive outward alignment
const CELL: f64 = 1.0 / 1024.0;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;
const BANDS: usize = 2;

fn band_value(band: usize, col: usize, row: usize) -> f64 {
    let (x, y) = (col as f64, row as f64);
    match band {
        0 => 0.2 + 0.1 * (x * 0.11).sin() * (y * 0.07).cos(),
        _ => 12.0 + 3.0 * ((x + y) * 0.05).sin() * (x * 0.03).cos(),
    }
}

fn src() -> RasterSrc {
    src_with(|_, _, _| false)
}

/// two bands, with `hole` naming the cells written as nodata
fn src_with(hole: impl Fn(usize, usize, usize) -> bool) -> RasterSrc {
    let bands: Vec<Raster> = (0..BANDS)
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

async fn pull(op: FocalOp, radius: u32, source: RasterSrc, bbox: Bbox) -> RasterChunk {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(source));
    let focal = g.add_transform(s, Box::new(Focal::new(op, radius)));
    let engine = Engine::new(g, 64 << 20).unwrap();
    engine
        .pull(
            focal,
            WindowReq {
                bbox,
                resolution: CELL,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap()
}

/// the reference statistic, computed straight off the source function with no
/// help from the element under test
fn reference(op: FocalOp, radius: u32, band: usize, col: usize, row: usize) -> f64 {
    let radius = radius as isize;
    let mut values = Vec::new();
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let (x, y) = (col as isize + dx, row as isize + dy);
            values.push(band_value(band, x as usize, y as usize));
        }
    }
    match op {
        FocalOp::Mean => values.iter().sum::<f64>() / values.len() as f64,
        FocalOp::Median => {
            values.sort_by(f64::total_cmp);
            values[values.len() / 2]
        }
        FocalOp::Min => values.iter().copied().fold(f64::INFINITY, f64::min),
        FocalOp::Max => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
    }
}

#[tokio::test]
async fn every_op_matches_the_hand_computed_window() {
    let (px0, py0) = (40usize, 40usize);
    let (cols, rows) = (12usize, 10usize);
    for op in [FocalOp::Mean, FocalOp::Median, FocalOp::Min, FocalOp::Max] {
        for radius in [1u32, 3] {
            let got = pull(op, radius, src(), window(px0, py0, px0 + cols, py0 + rows)).await;
            assert_eq!(
                got.bands.band_count(),
                BANDS,
                "{op:?}: focal keeps the band count"
            );
            assert_eq!((got.width(), got.height()), (cols, rows));
            for band in 0..BANDS {
                let plane = got.bands.band(band).unwrap();
                for row in 0..rows {
                    for col in 0..cols {
                        let want = reference(op, radius, band, px0 + col, py0 + row);
                        let a = plane.data()[row * cols + col];
                        assert!(
                            (a - want).abs() < 1e-12,
                            "{op:?} radius {radius} band {band} cell ({col},{row}): got {a} vs reference {want}"
                        );
                    }
                }
            }
        }
    }
}

#[tokio::test]
async fn a_nodata_neighbour_drops_out_and_a_nodata_centre_holds() {
    // one blank cell in band 0 only, so band 1 shows the bands are independent
    let (hole_x, hole_y) = (40usize, 30usize);
    let hole =
        move |band: usize, col: usize, row: usize| band == 0 && col == hole_x && row == hole_y;
    let (px0, py0) = (32usize, 24usize);
    let (cols, rows) = (16usize, 16usize);
    let got = pull(
        FocalOp::Mean,
        1,
        src_with(hole),
        window(px0, py0, px0 + cols, py0 + rows),
    )
    .await;

    let cell = |band: usize, col: usize, row: usize| {
        got.bands.band(band).unwrap().data()[(row - py0) * cols + (col - px0)]
    };
    assert!(
        cell(0, hole_x, hole_y).is_nan(),
        "a nodata centre came back as {}",
        cell(0, hole_x, hole_y)
    );
    assert!(
        cell(1, hole_x, hole_y).is_finite(),
        "band 1 lost a pixel to band 0's nodata"
    );

    // the neighbour east of the hole: eight valid taps, the hole excluded
    let mut valid = Vec::new();
    for dy in -1i64..=1 {
        for dx in -1i64..=1 {
            let (x, y) = (hole_x as i64 + 1 + dx, hole_y as i64 + dy);
            if (x, y) != (hole_x as i64, hole_y as i64) {
                valid.push(band_value(0, x as usize, y as usize));
            }
        }
    }
    assert_eq!(valid.len(), 8);
    let want = valid.iter().sum::<f64>() / valid.len() as f64;
    let a = cell(0, hole_x + 1, hole_y);
    assert!(
        (a - want).abs() < 1e-12,
        "neighbour of a hole: got {a} vs the mean of the eight valid taps {want}"
    );
}

#[tokio::test]
async fn an_all_nodata_window_comes_back_nodata() {
    let (block_x, block_y) = (44usize, 30usize);
    let hole = move |band: usize, col: usize, row: usize| {
        band == 0 && col.abs_diff(block_x) <= 1 && row.abs_diff(block_y) <= 1
    };
    let (px0, py0) = (32usize, 24usize);
    let (cols, rows) = (16usize, 16usize);
    let got = pull(
        FocalOp::Max,
        1,
        src_with(hole),
        window(px0, py0, px0 + cols, py0 + rows),
    )
    .await;

    let plane = got.bands.band(0).unwrap();
    let cell = |col: usize, row: usize| plane.data()[(row - py0) * cols + (col - px0)];
    assert!(
        cell(block_x, block_y).is_nan(),
        "a window with no valid values came back as {}",
        cell(block_x, block_y)
    );
    assert!(
        cell(block_x + 2, block_y).is_finite(),
        "the blank block swallowed a cell whose window still has valid taps"
    );
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
    const RADIUS: u32 = 2;
    // 5x4 tiles of 16 px once the chunk demand lands, one piece without it,
    // so every interior cell of the chunked pull sits within the halo of a
    // seam
    let bbox = window(32, 32, 112, 96);
    let whole = pull(FocalOp::Mean, RADIUS, src(), bbox).await;

    let mut g = Graph::new();
    let s = g.add_source(Box::new(src()));
    let focal = g.add_transform(s, Box::new(Focal::new(FocalOp::Mean, RADIUS)));
    g.add_transform(focal, Box::new(SmallChunks));
    let engine = Engine::new(g, 64 << 20).unwrap();
    assert_eq!(engine.caps(focal).raster().chunk_px, 16);
    let chunked = engine
        .pull(
            focal,
            WindowReq {
                bbox,
                resolution: CELL,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap();

    assert_eq!((chunked.width(), chunked.height()), (80, 64));
    let bits = |c: &RasterChunk, band: usize| {
        c.bands
            .band(band)
            .unwrap()
            .data()
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    for band in 0..BANDS {
        assert_eq!(
            bits(&chunked, band),
            bits(&whole, band),
            "band {band}: chunked focal differs from the whole-window computation"
        );
    }
}
