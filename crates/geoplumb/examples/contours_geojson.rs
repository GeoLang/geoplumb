use geoplumb::caps::ValueKind;
use geoplumb::elements::{CogSrc, Contours};
use geoplumb::{Bbox, Engine, Graph, Source, WindowReq};
use topoi_core::geojson::{Feature, FeatureCollection, write_geojson};

const USAGE: &str =
    "usage: contours_geojson <dem.tif> <column> <row> <columns> <rows> <interval> <out.geojson>";

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, column, row, columns, rows, interval, out] = args.as_slice() else {
        panic!("{USAGE}");
    };
    let number = |text: &String| -> f64 { text.parse().expect(USAGE) };
    let file = std::fs::File::open(path).expect("open the dem");
    let source = CogSrc::open(file, ValueKind::Continuous).expect("read the cog header");
    let grid = source.grid();
    let resolution = grid.base_resolution;
    let window = Bbox {
        min_x: grid.origin_x + number(column) * resolution,
        max_x: grid.origin_x + (number(column) + number(columns)) * resolution,
        max_y: grid.origin_y - number(row) * resolution,
        min_y: grid.origin_y - (number(row) + number(rows)) * resolution,
    };

    let mut graph = Graph::new();
    let dem = graph.add_source(Box::new(source));
    let contours = Contours {
        interval: number(interval),
        base: 0.0,
    };
    let lines = graph.add_transform(dem, Box::new(contours));
    let engine = Engine::new(graph, 256 << 20).expect("negotiation");
    let pulled = engine
        .pull(
            lines,
            WindowReq {
                bbox: window,
                resolution,
                time: None,
            },
        )
        .await
        .expect("pull")
        .into_vector()
        .expect("contours are features");

    // gdal_contour's lines end at the outermost pixel centers, not the window edge
    let centers = window.expand(-resolution / 2.0);
    let dissolved = pulled.crop_to(&centers).dissolve();
    let collection = FeatureCollection {
        features: dissolved
            .features
            .into_iter()
            .map(|f| Feature {
                geometry: Some(f.geometry),
                properties: f.properties,
            })
            .collect(),
    };
    std::fs::write(out, write_geojson(&collection)).expect("write the geojson");
    println!(
        "{}",
        serde_json::json!({
            "min_x": centers.min_x,
            "min_y": centers.min_y,
            "max_x": centers.max_x,
            "max_y": centers.max_y,
            "resolution": resolution,
            "chunk_px": engine.grid(lines).chunk_px,
        })
    );
}
