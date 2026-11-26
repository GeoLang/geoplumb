use geoplumb::caps::ValueKind;
use geoplumb::elements::{CogSrc, Contours, Polygonize};
use geoplumb::{Bbox, Engine, Graph, Source, Transform, WindowReq};
use topoi_core::geojson::{Feature, FeatureCollection, write_geojson};

const USAGE: &str = "usage: raster_vector_geojson <raster.tif> <column> <row> <columns> <rows> <out.geojson> (contours <interval> | polygonize)";

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [path, column, row, columns, rows, out, operation @ ..] = args.as_slice() else {
        panic!("{USAGE}");
    };
    let number = |text: &String| -> f64 { text.parse().expect(USAGE) };
    let (value_kind, transform, inset_px): (ValueKind, Box<dyn Transform>, f64) = match operation {
        // gdal_contour's lines end at the outermost pixel centers, not the window edge
        [name, interval] if name == "contours" => (
            ValueKind::Continuous,
            Box::new(Contours {
                interval: number(interval),
                base: 0.0,
            }),
            0.5,
        ),
        [name] if name == "polygonize" => (ValueKind::Categorical, Box::new(Polygonize), 0.0),
        _ => panic!("{USAGE}"),
    };
    let file = std::fs::File::open(path).expect("open the raster");
    let source = CogSrc::open(file, value_kind).expect("read the cog header");
    let grid = source.grid();
    let resolution = grid.base_resolution;
    let window = Bbox {
        min_x: grid.origin_x + number(column) * resolution,
        max_x: grid.origin_x + (number(column) + number(columns)) * resolution,
        max_y: grid.origin_y - number(row) * resolution,
        min_y: grid.origin_y - (number(row) + number(rows)) * resolution,
    };

    let mut graph = Graph::new();
    let raster = graph.add_source(Box::new(source));
    let features = graph.add_transform(raster, transform);
    let engine = Engine::new(graph, 256 << 20).expect("negotiation");
    let pulled = engine
        .pull(
            features,
            WindowReq {
                bbox: window,
                resolution,
                time: None,
            },
        )
        .await
        .expect("pull")
        .into_vector()
        .expect("raster to vector output is features");

    let extent = window.expand(-inset_px * resolution);
    let dissolved = pulled.crop_to(&extent).dissolve();
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
            "min_x": extent.min_x,
            "min_y": extent.min_y,
            "max_x": extent.max_x,
            "max_y": extent.max_y,
            "resolution": resolution,
            "chunk_px": engine.grid(features).chunk_px,
        })
    );
}
