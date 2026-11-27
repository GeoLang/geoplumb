//! batch driver: materialize an xyz tile pyramid from the same graph the
//! live server pulls, proving batch is a full-extent pull.
//!
//! usage: pyramid [max_z] [out_dir], defaults 12 and ./tiles

use geoplumb::caps::ValueKind;
use geoplumb::elements::{Hillshade, RasterSrc, Reproject};
use geoplumb::tile::XyzTile;
use geoplumb::{Bbox, Crs, Engine, Graph};
use terrano_core::{BandedRaster, Raster};

const W: usize = 600;
const H: usize = 400;
const CELL: f64 = 0.001;
const ORIGIN: (f64, f64) = (7.0, 47.0);

fn synthetic_dem() -> RasterSrc {
    let mut data = Vec::with_capacity(W * H);
    for row in 0..H {
        for col in 0..W {
            let lon = ORIGIN.0 + (col as f64 + 0.5) * CELL;
            let lat = ORIGIN.1 - (row as f64 + 0.5) * CELL;
            data.push(500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos());
        }
    }
    let dem = Raster::from_vec(W, H, data, CELL, f64::NAN).unwrap();
    RasterSrc::new(
        BandedRaster::new(vec![dem]).unwrap(),
        ORIGIN.0,
        ORIGIN.1,
        Crs::WGS84,
        ValueKind::Continuous,
    )
}

#[tokio::main]
async fn main() {
    let mut args = std::env::args().skip(1);
    let max_z: u8 = args.next().and_then(|a| a.parse().ok()).unwrap_or(12);
    let out = args.next().unwrap_or_else(|| "tiles".into());

    let mut g = Graph::new();
    let s = g.add_source(Box::new(synthetic_dem()));
    let hs = g.add_transform(s, Box::new(Hillshade::new(315.0, 45.0)));
    let rp = g.add_transform(hs, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
    let engine = Engine::new(g, 256 << 20).expect("negotiation");

    let extent = Bbox::new(
        ORIGIN.0,
        ORIGIN.1 - H as f64 * CELL,
        ORIGIN.0 + W as f64 * CELL,
        ORIGIN.1,
    );
    let started = std::time::Instant::now();
    let mut count = 0u32;
    for z in 0..=max_z {
        for tile in XyzTile::covering(z, &extent) {
            let chunk = geoplumb::tile::render_tile(&engine, rp, tile)
                .await
                .expect("render");
            let png = geoplumb::encode::png_gray(&chunk, 0.0, 255.0).expect("encode");
            let dir = format!("{out}/{z}/{}", tile.x);
            std::fs::create_dir_all(&dir).expect("mkdir");
            std::fs::write(format!("{dir}/{}.png", tile.y), png).expect("write");
            count += 1;
        }
    }
    println!(
        "{count} tiles to ./{out} in {:.2}s",
        started.elapsed().as_secs_f64()
    );
}
