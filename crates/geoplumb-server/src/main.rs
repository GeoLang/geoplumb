use std::path::PathBuf;

use geoplumb_server::config::Config;
use geoplumb_server::{Layer, router};

const DEFAULT_CACHE_BYTES: usize = 256 << 20;
const DEFAULT_PORT: u16 = 3000;

#[tokio::main]
async fn main() {
    if let Err(message) = run().await {
        eprintln!("geoplumb-server: {message}");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let path = std::env::var("GEOPLUMB_LAYERS")
        .map_err(|_| "set GEOPLUMB_LAYERS to a layer toml file".to_string())?;
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{path}: {e}"))?;
    let config = Config::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    let budget = env_number("GEOPLUMB_CACHE_BYTES", DEFAULT_CACHE_BYTES)?;
    let port = env_number("PORT", DEFAULT_PORT)?;
    let disk = std::env::var_os("GEOPLUMB_DISK_CACHE").map(PathBuf::from);

    // opening sources is blocking: a stac layer searches its anchor bbox
    // and a cog layer reads a file header
    let layers =
        tokio::task::spawn_blocking(move || Layer::build_all(&config, budget, disk.as_deref()))
            .await
            .expect("layer build panicked")?;

    let bind = format!("0.0.0.0:{port}");
    let names: Vec<&str> = layers.iter().map(|l| l.info().name.as_str()).collect();
    println!(
        "geoplumb-server listening on {bind}, layers: {}",
        names.join(", ")
    );
    let listener = tokio::net::TcpListener::bind(&bind)
        .await
        .map_err(|e| format!("{bind}: {e}"))?;
    axum::serve(listener, router(layers))
        .await
        .map_err(|e| e.to_string())
}

fn env_number<T: std::str::FromStr>(key: &str, default: T) -> Result<T, String> {
    match std::env::var(key) {
        Err(_) => Ok(default),
        Ok(value) => value
            .parse()
            .map_err(|_| format!("{key} is not a number: {value:?}")),
    }
}
