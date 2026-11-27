use geoplumb::RasterBands;
use geoplumb::caps::{BandKinds, ValueKind};
use geoplumb::elements::{FlowAccumulation, FlowDirection, RasterSrc, StreamOrder, Watershed};
use geoplumb::{Bbox, Chunk, Crs, Engine, Graph, RasterChunk, Transform, WindowReq};
use terrano_core::{BandedRaster, Raster};

const UTM_32N: Crs = Crs(32632);
const ORIGIN_X: f64 = 380_000.0;
const ORIGIN_Y: f64 = 5_170_000.0;
const CELL: f64 = 30.0;
const CELL_AREA: f64 = CELL * CELL;

fn cells_bbox(column0: usize, row0: usize, columns: usize, rows: usize) -> Bbox {
    Bbox::new(
        ORIGIN_X + column0 as f64 * CELL,
        ORIGIN_Y - (row0 + rows) as f64 * CELL,
        ORIGIN_X + (column0 + columns) as f64 * CELL,
        ORIGIN_Y - row0 as f64 * CELL,
    )
}

fn grid(columns: usize, rows: usize, value: impl Fn(usize, usize) -> f64) -> Vec<f64> {
    let mut values = Vec::with_capacity(columns * rows);
    for row in 0..rows {
        for column in 0..columns {
            values.push(value(row, column));
        }
    }
    values
}

fn computed(element: &dyn Transform, dem: Vec<f64>, columns: usize, rows: usize) -> Vec<f64> {
    let bbox = cells_bbox(0, 0, columns, rows);
    let input = RasterChunk {
        bands: RasterBands::from_rasters(vec![
            Raster::from_vec(columns, rows, dem, CELL, f64::NAN).unwrap(),
        ])
        .unwrap(),
        bbox,
        resolution: CELL,
        crs: UTM_32N,
    };
    let out = WindowReq {
        bbox,
        resolution: CELL,
        time: None,
    };
    let output = element
        .compute(&out, &Chunk::Raster(input))
        .unwrap()
        .into_raster()
        .unwrap();
    assert_eq!(output.bbox, bbox);
    output.bands.band(0).unwrap().to_raster().data().to_vec()
}

const PLANE_SIDE: usize = 4;

#[test]
fn flow_directions_follow_a_plane_falling_south_east() {
    let plane = grid(PLANE_SIDE, PLANE_SIDE, |row, column| {
        20.0 - row as f64 - column as f64
    });
    let element = FlowDirection {
        crs: UTM_32N,
        region: cells_bbox(0, 0, PLANE_SIDE, PLANE_SIDE),
    };
    // esri d8 codes: 1 east, 2 south east, 4 south, 0 where no neighbour is lower
    #[rustfmt::skip]
    let expected = vec![
        2.0, 2.0, 2.0, 4.0,
        2.0, 2.0, 2.0, 4.0,
        2.0, 2.0, 2.0, 4.0,
        1.0, 1.0, 1.0, 0.0,
    ];
    assert_eq!(computed(&element, plane, PLANE_SIDE, PLANE_SIDE), expected);
}

const BOWL_SIDE: usize = 7;
const BOWL_NOTCH: (usize, usize) = (6, 3);

fn bowl() -> Vec<f64> {
    grid(BOWL_SIDE, BOWL_SIDE, |row, column| {
        if (row, column) == BOWL_NOTCH {
            0.0
        } else {
            10.0 + (row as f64 - 3.0).hypot(column as f64 - 3.0)
        }
    })
}

#[test]
fn a_bowl_drains_through_its_notch_carrying_its_whole_area() {
    let region = cells_bbox(0, 0, BOWL_SIDE, BOWL_SIDE);
    let notch = BOWL_NOTCH.0 * BOWL_SIDE + BOWL_NOTCH.1;

    let directions = computed(
        &FlowDirection {
            crs: UTM_32N,
            region,
        },
        bowl(),
        BOWL_SIDE,
        BOWL_SIDE,
    );
    let pits: Vec<usize> = (0..directions.len())
        .filter(|&cell| directions[cell] == 0.0)
        .collect();
    assert_eq!(pits, vec![notch], "{directions:?}");

    let area = computed(
        &FlowAccumulation {
            crs: UTM_32N,
            region,
        },
        bowl(),
        BOWL_SIDE,
        BOWL_SIDE,
    );
    assert_eq!(area[notch], (BOWL_SIDE * BOWL_SIDE) as f64 * CELL_AREA);
    assert_eq!(area[0], CELL_AREA, "a corner takes in nothing but itself");
    assert!(area.iter().all(|value| *value >= CELL_AREA), "{area:?}");

    let basins = computed(
        &Watershed {
            crs: UTM_32N,
            region,
        },
        bowl(),
        BOWL_SIDE,
        BOWL_SIDE,
    );
    assert!(
        basins.iter().all(|label| *label == basins[notch]),
        "{basins:?}"
    );
}

#[test]
fn a_ridge_splits_two_basins() {
    let (columns, rows) = (7, 5);
    let ridge = grid(columns, rows, |row, column| {
        10.0 - (column as f64 - 3.0).abs() - 0.1 * row as f64
    });
    let basins = computed(
        &Watershed {
            crs: UTM_32N,
            region: cells_bbox(0, 0, columns, rows),
        },
        ridge,
        columns,
        rows,
    );
    let west = basins[0];
    let east = basins[columns - 1];
    assert!(west >= 1.0 && east >= 1.0, "{basins:?}");
    assert_ne!(west, east);
    for row in 0..rows {
        for column in 0..columns {
            // terrano's flow_direction picks east on a tie
            let expected = if column < 3 { west } else { east };
            assert_eq!(
                basins[row * columns + column],
                expected,
                "row {row} column {column}"
            );
        }
    }
}

const FORK_SIDE: usize = 7;

fn fork() -> Vec<f64> {
    let channel = [
        ((6, 3), 0.0),
        ((5, 3), 1.0),
        ((4, 3), 2.0),
        ((3, 3), 3.0),
        ((2, 2), 4.0),
        ((1, 1), 5.0),
        ((0, 0), 6.0),
        ((2, 4), 4.0),
        ((1, 5), 5.0),
        ((0, 6), 6.0),
    ];
    grid(FORK_SIDE, FORK_SIDE, |row, column| {
        if let Some((_, height)) = channel.iter().find(|(cell, _)| *cell == (row, column)) {
            return *height;
        }
        let cells_to_channel = channel
            .iter()
            .map(|((r, c), _)| row.abs_diff(*r).max(column.abs_diff(*c)))
            .min()
            .unwrap();
        10.0 + 2.0 * cells_to_channel as f64 + 0.1 * (FORK_SIDE - 1 - row) as f64
    })
}

#[test]
fn strahler_order_rises_once_where_two_streams_meet() {
    let element = StreamOrder {
        crs: UTM_32N,
        region: cells_bbox(0, 0, FORK_SIDE, FORK_SIDE),
        // between the three cells a slope cell gathers at most and the six at each channel's second cell
        threshold: 4.5 * CELL_AREA,
    };
    let orders = computed(&element, fork(), FORK_SIDE, FORK_SIDE);
    let n = f64::NAN;
    #[rustfmt::skip]
    let expected = [
        n,   n,   n,   n,   n,   n,   n,
        n,   1.0, n,   n,   n,   1.0, n,
        n,   n,   1.0, n,   1.0, n,   n,
        n,   n,   n,   2.0, n,   n,   n,
        n,   n,   n,   2.0, n,   n,   n,
        n,   n,   n,   2.0, n,   n,   n,
        n,   n,   n,   2.0, n,   n,   n,
    ];
    for (cell, (got, want)) in orders.iter().zip(expected).enumerate() {
        assert!(
            *got == want || (got.is_nan() && want.is_nan()),
            "cell {cell}: got {got}, want {want}, all {orders:?}"
        );
    }
}

fn source(crs: Crs, origin: (f64, f64), cell: f64, columns: usize, rows: usize) -> RasterSrc {
    let dem = grid(columns, rows, |row, column| {
        100.0 + row as f64 + column as f64
    });
    RasterSrc::new(
        BandedRaster::new(vec![
            Raster::from_vec(columns, rows, dem, cell, f64::NAN).unwrap(),
        ])
        .unwrap(),
        origin.0,
        origin.1,
        crs,
        ValueKind::Continuous,
    )
}

#[test]
fn each_element_reprojects_a_lon_lat_dem_into_its_crs_and_pins_its_value_kind() {
    let region = cells_bbox(0, 0, 10, 10);
    let elements: Vec<(Box<dyn Transform>, ValueKind)> = vec![
        (
            Box::new(FlowDirection {
                crs: UTM_32N,
                region,
            }),
            ValueKind::Categorical,
        ),
        (
            Box::new(FlowAccumulation {
                crs: UTM_32N,
                region,
            }),
            ValueKind::Peak,
        ),
        (
            Box::new(Watershed {
                crs: UTM_32N,
                region,
            }),
            ValueKind::Categorical,
        ),
        (
            Box::new(StreamOrder {
                crs: UTM_32N,
                region,
                threshold: 10.0 * CELL_AREA,
            }),
            ValueKind::Categorical,
        ),
    ];
    for (element, value_kind) in elements {
        let identity = element.identity();
        let mut g = Graph::new();
        let src = g.add_source(Box::new(source(Crs::WGS84, (7.0, 47.0), 0.001, 64, 64)));
        let node = g.add_transform(src, element);
        let engine = Engine::new(g, 64 << 20).unwrap_or_else(|e| panic!("{identity}: {e}"));
        let caps = engine.caps(node).raster();
        assert_eq!(caps.crs, UTM_32N, "{identity}");
        assert_eq!(caps.value_kinds, BandKinds::from(value_kind), "{identity}");
        assert_eq!(engine.caps(src).raster().crs, Crs::WGS84);
    }
}

#[test]
fn a_lon_lat_crs_is_refused() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(source(Crs::WGS84, (7.0, 47.0), 0.001, 64, 64)));
    g.add_transform(
        src,
        Box::new(FlowAccumulation {
            crs: Crs::WGS84,
            region: Bbox::new(7.0, 46.95, 7.05, 47.0),
        }),
    );
    let error = Engine::new(g, 64 << 20)
        .err()
        .expect("flow accumulation solved in degrees");
    assert!(error.to_string().contains("projected crs"), "{error}");
}

#[test]
fn a_stream_threshold_that_is_no_area_is_refused() {
    let mut g = Graph::new();
    let src = g.add_source(Box::new(source(
        UTM_32N,
        (ORIGIN_X, ORIGIN_Y),
        CELL,
        16,
        16,
    )));
    g.add_transform(
        src,
        Box::new(StreamOrder {
            crs: UTM_32N,
            region: cells_bbox(0, 0, 16, 16),
            threshold: 0.0,
        }),
    );
    let error = Engine::new(g, 64 << 20)
        .err()
        .expect("stream order took a threshold of 0");
    assert!(error.to_string().contains("threshold"), "{error}");
}

#[tokio::test]
async fn a_pull_cuts_its_window_out_of_the_one_region_compute() {
    let (side, offset) = (16, 4);
    let in_bowl = |row: usize, column: usize| {
        (offset..offset + BOWL_SIDE).contains(&row)
            && (offset..offset + BOWL_SIDE).contains(&column)
    };
    let bowl = bowl();
    let dem = grid(side, side, |row, column| {
        if in_bowl(row, column) {
            bowl[(row - offset) * BOWL_SIDE + column - offset]
        } else {
            100.0
        }
    });
    let src = RasterSrc::new(
        BandedRaster::new(vec![
            Raster::from_vec(side, side, dem, CELL, f64::NAN).unwrap(),
        ])
        .unwrap(),
        ORIGIN_X,
        ORIGIN_Y,
        UTM_32N,
        ValueKind::Continuous,
    );
    let mut g = Graph::new();
    let src = g.add_source(Box::new(src));
    let node = g.add_transform(
        src,
        Box::new(FlowAccumulation {
            crs: UTM_32N,
            region: cells_bbox(offset, offset, BOWL_SIDE, BOWL_SIDE),
        }),
    );
    let engine = Engine::new(g, 64 << 20).unwrap();
    let pulled = engine
        .pull(
            node,
            WindowReq {
                bbox: cells_bbox(0, 0, side, side),
                resolution: CELL,
                time: None,
            },
        )
        .await
        .unwrap()
        .into_raster()
        .unwrap();
    let area = pulled.bands.band(0).unwrap().to_raster();
    let area = area.data();
    assert_eq!(area.len(), side * side);
    for row in 0..side {
        for column in 0..side {
            let value = area[row * side + column];
            if in_bowl(row, column) {
                assert!(value >= CELL_AREA, "row {row} column {column}: {value}");
            } else {
                assert!(value.is_nan(), "row {row} column {column}: {value}");
            }
        }
    }
    let notch = (offset + BOWL_NOTCH.0) * side + offset + BOWL_NOTCH.1;
    assert_eq!(area[notch], (BOWL_SIDE * BOWL_SIDE) as f64 * CELL_AREA);
}
