//! live hillshade tiles from a public stac collection, no local data:
//! Copernicus DEM cogs stream in over http range requests as tiles are
//! viewed. needs network. override the area with GEOPLUMB_BBOX
//! (minlon,minlat,maxlon,maxlat), the api and collection with
//! GEOPLUMB_STAC_API / GEOPLUMB_STAC_COLLECTION / GEOPLUMB_STAC_ASSET.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::{StatusCode, header};
use axum::response::IntoResponse;
use axum::routing::get;
use geoplumb::elements::{Hillshade, Reproject, StacSearch, StacSrc};
use geoplumb::tile::XyzTile;
use geoplumb::{Crs, Engine, Graph, NodeId};

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).unwrap_or_else(|_| default.into())
}

fn build_engine() -> (Engine, NodeId) {
    let bbox: Vec<f64> = env_or("GEOPLUMB_BBOX", "7.0,46.3,8.0,46.9")
        .split(',')
        .map(|v| v.trim().parse().expect("GEOPLUMB_BBOX numbers"))
        .collect();
    let search = StacSearch::new(
        &env_or(
            "GEOPLUMB_STAC_API",
            "https://earth-search.aws.element84.com/v1",
        ),
        &env_or("GEOPLUMB_STAC_COLLECTION", "cop-dem-glo-30"),
        &env_or("GEOPLUMB_STAC_ASSET", "data"),
        [bbox[0], bbox[1], bbox[2], bbox[3]],
    );
    let src = StacSrc::open(&search).expect("stac search");
    println!(
        "{} items in {} ({})",
        src.item_count(),
        search.collection,
        src.crs()
    );
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
    let (engine, node) = tokio::task::spawn_blocking(build_engine)
        .await
        .expect("engine build");
    let app = axum::Router::new()
        .route("/{z}/{x}/{y}", get(tile))
        .with_state((Arc::new(engine), node));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:8478")
        .await
        .unwrap();
    println!("live cop-dem hillshade at http://127.0.0.1:8478/{{z}}/{{x}}/{{y}}.png");
    axum::serve(listener, app).await.unwrap();
}
