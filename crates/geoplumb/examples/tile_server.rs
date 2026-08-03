//! live driver: hillshade xyz tiles over the pull engine.
//!
//! GEOPLUMB_DEM=/path/to/dem.tif serves a real geotiff, otherwise a
//! synthetic alpine dem. open the printed url in a slippy-map viewer.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use geoplumb::elements::{Hillshade, RasterSrc, Reproject};
use geoplumb::tile::XyzTile;
use geoplumb::{Crs, Engine, Graph, NodeId};
use terrano_core::{BandedRaster, Raster};

fn synthetic_dem() -> RasterSrc {
    let (w, h, cell) = (600, 400, 0.001);
    let (ox, oy) = (7.0, 47.0);
    let mut data = Vec::with_capacity(w * h);
    for row in 0..h {
        for col in 0..w {
            let lon = ox + (col as f64 + 0.5) * cell;
            let lat = oy - (row as f64 + 0.5) * cell;
            data.push(500.0 + 200.0 * (lon * 8.0).sin() * (lat * 8.0).cos());
        }
    }
    let dem = Raster::from_vec(w, h, data, cell, f64::NAN).unwrap();
    RasterSrc::new(BandedRaster::new(vec![dem]).unwrap(), ox, oy, Crs::WGS84)
}

fn build_engine() -> (Engine, NodeId) {
    let src = match std::env::var("GEOPLUMB_DEM") {
        Ok(path) => {
            let bytes = std::fs::read(&path).expect("read GEOPLUMB_DEM");
            RasterSrc::from_geotiff(&bytes).expect("parse geotiff")
        }
        Err(_) => synthetic_dem(),
    };
    let mut g = Graph::new();
    let s = g.add_source(Box::new(src));
    let hs = g.add_transform(s, Box::new(Hillshade::new(315.0, 45.0)));
    let rp = g.add_transform(hs, Box::new(Reproject::new(Crs::WEB_MERCATOR)));
    (Engine::new(g, 256 << 20).expect("negotiation"), rp)
}

async fn tile(
    State((engine, node)): State<(Arc<Engine>, NodeId)>,
    Path((z, x, y)): Path<(u8, u32, String)>,
) -> impl IntoResponse {
    let Ok(y) = y.trim_end_matches(".png").parse::<u32>() else {
        return (StatusCode::BAD_REQUEST, "bad y").into_response();
    };
    match geoplumb::tile::render_tile(&engine, node, XyzTile { z, x, y }).await {
        Ok(chunk) => match geoplumb::encode::png_gray(&chunk, 0.0, 255.0) {
            Ok(png) => ([(header::CONTENT_TYPE, "image/png")], png).into_response(),
            Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
        },
        Err(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()).into_response(),
    }
}

#[tokio::main]
async fn main() {
    let (engine, node) = build_engine();
    let app = axum::Router::new()
        .route("/{z}/{x}/{y}", get(tile))
        .with_state((Arc::new(engine), node));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8477")
        .await
        .unwrap();
    println!("hillshade tiles at http://127.0.0.1:8477/{{z}}/{{x}}/{{y}}.png");
    axum::serve(listener, app).await.unwrap();
}
