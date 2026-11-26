use std::path::PathBuf;

use geoplumb::caps::ValueKind;
use geoplumb::element::Source;
use geoplumb::elements::RasterSrc;
use geoplumb::{Bbox, RasterChunk, WindowReq};

const MICRODEGREES: i32 = 1_000_000;

struct Grid {
    template: u16,
    first_lat: i32,
    first_lon: i32,
    scanning_mode: u8,
}

const COLUMNS: usize = 4;
const ROWS: usize = 3;
const POINTS: usize = COLUMNS * ROWS;

struct Packing {
    template: u16,
    packed_count: usize,
    reference: f32,
    binary_scale: i16,
    decimal_scale: i16,
    bits: u8,
}

fn section(number: u8, body: &[u8]) -> Vec<u8> {
    let mut out = ((body.len() + 5) as u32).to_be_bytes().to_vec();
    out.push(number);
    out.extend(body);
    out
}

fn sign_magnitude_32(value: i32) -> [u8; 4] {
    // grib2 signed integers are a sign bit above a magnitude
    let magnitude = value.unsigned_abs();
    match value < 0 {
        true => (magnitude | 0x8000_0000).to_be_bytes(),
        false => magnitude.to_be_bytes(),
    }
}

fn sign_magnitude_16(value: i16) -> [u8; 2] {
    let magnitude = value.unsigned_abs();
    match value < 0 {
        true => (magnitude | 0x8000).to_be_bytes(),
        false => magnitude.to_be_bytes(),
    }
}

fn grid_section(grid: &Grid) -> Vec<u8> {
    // body index is the wmo octet number minus six
    let mut body = vec![0u8; 67];
    body[1..5].copy_from_slice(&(POINTS as u32).to_be_bytes());
    body[7..9].copy_from_slice(&grid.template.to_be_bytes());
    body[9] = 6;
    body[25..29].copy_from_slice(&(COLUMNS as u32).to_be_bytes());
    body[29..33].copy_from_slice(&(ROWS as u32).to_be_bytes());
    body[37..41].copy_from_slice(&u32::MAX.to_be_bytes());
    body[41..45].copy_from_slice(&sign_magnitude_32(grid.first_lat));
    body[45..49].copy_from_slice(&sign_magnitude_32(grid.first_lon));
    body[49] = 0x30;
    body[58..62].copy_from_slice(&(MICRODEGREES as u32).to_be_bytes());
    body[62..66].copy_from_slice(&(MICRODEGREES as u32).to_be_bytes());
    body[66] = grid.scanning_mode;
    section(3, &body)
}

fn packing_section(packing: &Packing) -> Vec<u8> {
    let mut body = Vec::new();
    body.extend((packing.packed_count as u32).to_be_bytes());
    body.extend(packing.template.to_be_bytes());
    body.extend(packing.reference.to_be_bytes());
    body.extend(sign_magnitude_16(packing.binary_scale));
    body.extend(sign_magnitude_16(packing.decimal_scale));
    body.push(packing.bits);
    body.push(0);
    section(5, &body)
}

fn pack(values: &[u64], bits: usize) -> Vec<u8> {
    let mut out = vec![0u8; (values.len() * bits).div_ceil(8)];
    for (index, value) in values.iter().enumerate() {
        for bit in 0..bits {
            if (value >> (bits - 1 - bit)) & 1 == 1 {
                let position = index * bits + bit;
                out[position / 8] |= 0x80 >> (position % 8);
            }
        }
    }
    out
}

fn message(sections: &[Vec<u8>]) -> Vec<u8> {
    let mut out = b"GRIB\0\0\0\x02".to_vec();
    out.extend([0u8; 8]);
    out.extend(section(1, &[0u8; 16]));
    for s in sections {
        out.extend(s);
    }
    out.extend(b"7777");
    let length = out.len() as u64;
    out[8..16].copy_from_slice(&length.to_be_bytes());
    out
}

const MISSING_POINTS: [usize; 2] = [5, 10];

fn field_one_packed(point: usize) -> u64 {
    point as u64 * 3
}

fn field_two_packed(present_index: usize) -> u64 {
    100 + present_index as u64
}

fn fixture() -> Vec<u8> {
    // message one holds two fields sharing a bitmap, message two a
    // complex-packed field and a lambert conformal one
    let present: Vec<usize> = (0..POINTS)
        .filter(|p| !MISSING_POINTS.contains(p))
        .collect();
    let mut bitmap = vec![0u8];
    let mut flags = [0u8; 2];
    for point in &present {
        flags[point / 8] |= 0x80 >> (point % 8);
    }
    bitmap.extend(flags);
    let simple = |reference, binary_scale, decimal_scale, bits| Packing {
        template: 0,
        packed_count: present.len(),
        reference,
        binary_scale,
        decimal_scale,
        bits,
    };
    let four = section(4, &[0u8; 29]);
    let first = message(&[
        grid_section(&Grid {
            template: 0,
            first_lat: -MICRODEGREES,
            first_lon: 200 * MICRODEGREES,
            scanning_mode: 0x40,
        }),
        four.clone(),
        packing_section(&simple(10.0, -1, 1, 12)),
        section(6, &bitmap),
        section(
            7,
            &pack(
                &present
                    .iter()
                    .map(|p| field_one_packed(*p))
                    .collect::<Vec<_>>(),
                12,
            ),
        ),
        grid_section(&Grid {
            template: 0,
            first_lat: MICRODEGREES,
            first_lon: 10 * MICRODEGREES,
            scanning_mode: 0x80,
        }),
        four.clone(),
        packing_section(&simple(0.0, 0, 0, 8)),
        section(6, &[254]),
        section(
            7,
            &pack(
                &(0..present.len()).map(field_two_packed).collect::<Vec<_>>(),
                8,
            ),
        ),
    ]);
    let regular = Grid {
        template: 0,
        first_lat: MICRODEGREES,
        first_lon: 0,
        scanning_mode: 0,
    };
    let second = message(&[
        grid_section(&regular),
        four.clone(),
        packing_section(&Packing {
            template: 3,
            ..simple(0.0, 0, 0, 8)
        }),
        section(6, &[255]),
        section(7, &[0u8; 12]),
        grid_section(&Grid {
            template: 30,
            ..regular
        }),
        four,
        packing_section(&simple(0.0, 0, 0, 8)),
        section(6, &[255]),
        section(7, &[0u8; 12]),
    ]);
    [first, second].concat()
}

fn fixture_path(dir: &tempfile::TempDir) -> PathBuf {
    let path = dir.path().join("fixture.grib2");
    std::fs::write(&path, fixture()).unwrap();
    path
}

async fn read_all(src: &RasterSrc) -> RasterChunk {
    let grid = src.grid();
    let bbox = Bbox::new(
        grid.origin_x,
        grid.origin_y - ROWS as f64,
        grid.origin_x + COLUMNS as f64,
        grid.origin_y,
    );
    let req = WindowReq {
        bbox,
        resolution: 1.0,
        time: None,
    };
    src.read(&req).await.unwrap().into_raster().unwrap()
}

fn assert_values(chunk: &RasterChunk, expected: &[f64]) {
    let actual = chunk.bands.bands()[0].data();
    assert_eq!(actual.len(), expected.len());
    for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a.is_nan() && e.is_nan()) || (a - e).abs() < 1e-9,
            "pixel {i}: read {a}, expected {e}"
        );
    }
}

#[tokio::test]
async fn northward_rows_flip_and_bitmap_holes_are_nan() {
    let dir = tempfile::tempdir().unwrap();
    let src = RasterSrc::from_grib(fixture_path(&dir), 1, ValueKind::Continuous).unwrap();
    let grid = src.grid();
    // stored rows run north from -1, columns from 200 east, which is -160
    assert_eq!(
        (grid.origin_x, grid.origin_y, grid.base_resolution),
        (-160.5, 1.5, 1.0)
    );
    let expected: Vec<f64> = (0..ROWS)
        .flat_map(|row| (0..COLUMNS).map(move |column| (ROWS - 1 - row) * COLUMNS + column))
        .map(|point| match MISSING_POINTS.contains(&point) {
            true => f64::NAN,
            false => (10.0 + field_one_packed(point) as f64 * 0.5) * 0.1,
        })
        .collect();
    assert_values(&read_all(&src).await, &expected);
}

#[tokio::test]
async fn second_field_reuses_the_bitmap_and_runs_west() {
    let dir = tempfile::tempdir().unwrap();
    let src = RasterSrc::from_grib(fixture_path(&dir), 2, ValueKind::Continuous).unwrap();
    let grid = src.grid();
    assert_eq!((grid.origin_x, grid.origin_y), (6.5, 1.5));
    let mut stored = Vec::new();
    let mut present_index = 0;
    for point in 0..POINTS {
        if MISSING_POINTS.contains(&point) {
            stored.push(f64::NAN);
            continue;
        }
        stored.push(field_two_packed(present_index) as f64);
        present_index += 1;
    }
    let expected: Vec<f64> = (0..ROWS)
        .flat_map(|row| (0..COLUMNS).map(move |column| row * COLUMNS + COLUMNS - 1 - column))
        .map(|point| stored[point])
        .collect();
    assert_values(&read_all(&src).await, &expected);
}

#[test]
fn unsupported_fields_fail_naming_what_they_use() {
    let dir = tempfile::tempdir().unwrap();
    let path = fixture_path(&dir);
    let failure = |message: usize| {
        RasterSrc::from_grib(&path, message, ValueKind::Continuous)
            .err()
            .expect("the field is refused")
            .to_string()
    };
    let complex = failure(3);
    assert!(complex.contains("5.3"), "{complex}");
    let lambert = failure(4);
    assert!(lambert.contains("3.30"), "{lambert}");
    let past_end = failure(5);
    assert!(past_end.contains("holds 4 messages"), "{past_end}");
    let zero = failure(0);
    assert!(zero.contains("count from 1"), "{zero}");
}
