//! batch driver: materialize an xyz tile pyramid from the same graph the
//! live server pulls, proving batch is a full-extent pull.
//!
//! usage: pyramid [max_z] [out_dir], defaults 12 and ./tiles

use geoplumb::elements::{Hillshade, RasterSrc, Reproject};
use geoplumb::tile::XyzTile;
use geoplumb::{Crs, Engine, Graph};
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
    )
}

fn tile_range(z: u8, lon0: f64, lat0: f64, lon1: f64, lat1: f64) -> (u32, u32, u32, u32) {
    let n = f64::from(1u32 << z);
    let tx = |lon: f64| (((lon + 180.0) / 360.0 * n).floor().max(0.0) as u32).min(n as u32 - 1);
    let ty = |lat: f64| {
        let r = lat.to_radians();
        ((((1.0 - (r.tan() + 1.0 / r.cos()).ln() / std::f64::consts::PI) / 2.0 * n).floor())
            .max(0.0) as u32)
            .min(n as u32 - 1)
    };
    (tx(lon0), tx(lon1), ty(lat0), ty(lat1))
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

    let (lon0, lat0) = ORIGIN;
    let (lon1, lat1) = (ORIGIN.0 + W as f64 * CELL, ORIGIN.1 - H as f64 * CELL);
    let started = std::time::Instant::now();
    let mut count = 0u32;
    for z in 0..=max_z {
        let (x0, x1, y0, y1) = tile_range(z, lon0, lat0, lon1, lat1);
        for x in x0..=x1 {
            for y in y0..=y1 {
                let tile = XyzTile { z, x, y };
                let chunk = geoplumb::tile::render_tile(&engine, rp, tile)
                    .await
                    .expect("render");
                let png = geoplumb::encode::png_gray(&chunk, 0.0, 255.0).expect("encode");
                let dir = format!("{out}/{z}/{x}");
                std::fs::create_dir_all(&dir).expect("mkdir");
                std::fs::write(format!("{dir}/{y}.png"), png).expect("write");
                count += 1;
            }
        }
    }
    println!(
        "{count} tiles to ./{out} in {:.2}s",
        started.elapsed().as_secs_f64()
    );
}
