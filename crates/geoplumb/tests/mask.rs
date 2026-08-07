//! quality masking: the keep-list against a hand-computed reference, every
//! band nulled at a rejected pixel, the band count and order kept, nodata in
//! the quality band treated as invalid, and the band count the mask demands
//! of its input link

use geoplumb::elements::{QualityMask, RasterSrc};
use geoplumb::{Bbox, Crs, Engine, Error, Graph, RasterChunk, WindowReq};
use terrano_core::{BandedRaster, Raster};

const W: usize = 320;
const H: usize = 256;
// binary exact, so pixel-aligned windows survive outward alignment
const CELL: f64 = 1.0 / 1024.0;
const ORIGIN_X: f64 = 7.0;
const ORIGIN_Y: f64 = 47.0;
const QUALITY_BAND: usize = 2;
/// the scl codes this fixture keeps: vegetation and bare soil
const KEEP: [f64; 2] = [4.0, 5.0];

fn reflectance(band: usize, col: usize, row: usize) -> f64 {
    let (x, y) = (col as f64, row as f64);
    match band {
        0 => 0.2 + 0.1 * (x * 0.11).sin() * (y * 0.07).cos(),
        _ => 0.6 + 0.2 * ((x + y) * 0.05).sin(),
    }
}

/// a spread of scl codes, only some of them in the keep-list
fn quality_code(col: usize, row: usize) -> f64 {
    ((col + 3 * row) % 11) as f64
}

fn src() -> RasterSrc {
    src_with(|_, _, _| false)
}

/// three bands, two reflectance and one quality, with `hole` naming the cells
/// written as nodata
fn src_with(hole: impl Fn(usize, usize, usize) -> bool) -> RasterSrc {
    let bands: Vec<Raster> = (0..3)
        .map(|b| {
            let mut data = Vec::with_capacity(W * H);
            for row in 0..H {
                for col in 0..W {
                    data.push(match () {
                        _ if hole(b, col, row) => f64::NAN,
                        _ if b == QUALITY_BAND => quality_code(col, row),
                        _ => reflectance(b, col, row),
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

async fn pull(source: RasterSrc, bbox: Bbox) -> RasterChunk {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(source));
    let masked = g.add_transform(s, Box::new(QualityMask::new(QUALITY_BAND, KEEP.to_vec())));
    let engine = Engine::new(g, 64 << 20).unwrap();
    engine
        .pull(
            masked,
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

#[tokio::test]
async fn listed_codes_pass_and_every_other_pixel_is_nan_in_all_bands() {
    let (px0, py0) = (16usize, 24usize);
    let (cols, rows) = (64usize, 48usize);
    let got = pull(src(), window(px0, py0, px0 + cols, py0 + rows)).await;

    assert_eq!(got.bands.band_count(), 3, "masking keeps the band count");
    assert_eq!((got.width(), got.height()), (cols, rows));
    let mut kept = 0usize;
    let mut dropped = 0usize;
    for row in 0..rows {
        for col in 0..cols {
            let (sx, sy) = (px0 + col, py0 + row);
            let cell = |band: usize| got.bands.band(band).unwrap().data()[row * cols + col];
            if KEEP.contains(&quality_code(sx, sy)) {
                kept += 1;
                for band in 0..2 {
                    let want = reflectance(band, sx, sy);
                    assert!(
                        (cell(band) - want).abs() < 1e-12,
                        "band {band} cell ({col},{row}): got {} vs source {want}",
                        cell(band)
                    );
                }
                assert_eq!(
                    cell(QUALITY_BAND),
                    quality_code(sx, sy),
                    "the quality band itself passes through at a kept pixel"
                );
            } else {
                dropped += 1;
                for band in 0..3 {
                    assert!(
                        cell(band).is_nan(),
                        "band {band} cell ({col},{row}): rejected code {} came back as {}",
                        quality_code(sx, sy),
                        cell(band)
                    );
                }
            }
        }
    }
    assert!(kept > 0 && dropped > 0, "window exercises both outcomes");
}

#[tokio::test]
async fn nodata_in_the_quality_band_masks_the_pixel() {
    // a cell the keep-list would otherwise pass, blanked in the quality band
    let (hole_x, hole_y) = (35usize, 30usize);
    assert!(KEEP.contains(&quality_code(hole_x, hole_y)));
    let hole = move |band: usize, col: usize, row: usize| {
        band == QUALITY_BAND && col == hole_x && row == hole_y
    };
    let (px0, py0) = (32usize, 24usize);
    let (cols, rows) = (16usize, 16usize);
    let got = pull(src_with(hole), window(px0, py0, px0 + cols, py0 + rows)).await;

    let cell = |band: usize, col: usize, row: usize| {
        got.bands.band(band).unwrap().data()[row * cols + col]
    };
    let (hole_col, hole_row) = (hole_x - px0, hole_y - py0);
    for band in 0..3 {
        assert!(
            cell(band, hole_col, hole_row).is_nan(),
            "band {band} kept a pixel whose quality code is nodata: {}",
            cell(band, hole_col, hole_row)
        );
    }
    // the nearest neighbour the keep-list passes is untouched
    let neighbour = (0..cols)
        .flat_map(|c| (0..rows).map(move |r| (c, r)))
        .find(|(c, r)| {
            (*c, *r) != (hole_col, hole_row) && KEEP.contains(&quality_code(px0 + c, py0 + r))
        })
        .expect("some kept pixel in the window");
    assert!(
        cell(0, neighbour.0, neighbour.1).is_finite(),
        "nodata leaked out of its own pixel"
    );
}

#[tokio::test]
async fn nodata_in_a_reflectance_band_stays_nan_where_the_code_passes() {
    // a cell the keep-list passes, so only the band's own nodata is in play
    let (hole_x, hole_y) = (35usize, 30usize);
    assert!(KEEP.contains(&quality_code(hole_x, hole_y)));
    let hole =
        move |band: usize, col: usize, row: usize| band == 0 && col == hole_x && row == hole_y;
    let (px0, py0) = (32usize, 24usize);
    let (cols, rows) = (16usize, 16usize);
    let got = pull(src_with(hole), window(px0, py0, px0 + cols, py0 + rows)).await;

    let (hole_col, hole_row) = (hole_x - px0, hole_y - py0);
    let cell = |band: usize| got.bands.band(band).unwrap().data()[hole_row * cols + hole_col];
    assert!(
        cell(0).is_nan(),
        "a nodata reflectance cell must stay nodata"
    );
    assert_eq!(
        cell(QUALITY_BAND),
        quality_code(hole_x, hole_y),
        "one band's nodata does not mask its neighbours"
    );
}

#[test]
fn negotiation_keeps_the_band_count_and_crs() {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(src()));
    let masked = g.add_transform(s, Box::new(QualityMask::new(QUALITY_BAND, KEEP.to_vec())));
    let engine = Engine::new(g, 64 << 20).unwrap();
    let caps = engine.caps(masked).raster();
    assert_eq!(caps.bands, 3);
    assert_eq!(caps.crs, Crs::WGS84);
}

#[test]
fn a_quality_band_the_source_lacks_fails_negotiation() {
    let mut g = Graph::new();
    let s = g.add_source(Box::new(src()));
    let masked = g.add_transform(s, Box::new(QualityMask::new(7, KEEP.to_vec())));
    let err = Engine::new(g, 64 << 20)
        .err()
        .expect("three bands cannot feed a mask reading band 7");
    match &err {
        Error::EmptyLink {
            upstream,
            downstream,
            detail,
        } => {
            assert_eq!((*upstream, *downstream), (s, masked));
            assert!(detail.contains("AtLeast(8)"), "unexpected detail: {detail}");
        }
        other => panic!("expected EmptyLink, got {other:?}"),
    }
}
