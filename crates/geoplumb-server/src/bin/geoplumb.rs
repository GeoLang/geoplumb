use std::path::PathBuf;

use clap::{Parser, Subcommand};
use geoplumb_server::ExportQuery;
use geoplumb_server::cli::{PyramidQuery, cog_info, write_layer, write_pyramid};

#[derive(Parser)]
#[command(
    name = "geoplumb",
    about = "Write a geoplumb layer to disk, or print what a COG declares"
)]
struct Arguments {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    #[command(
        about = "Write one layer of a layer TOML over a window: a raster layer as a COG, a vector layer as GeoJSON"
    )]
    Run {
        #[arg(help = "the layer TOML geoplumb-server reads")]
        layers: PathBuf,
        #[arg(help = "the name of the layer to write")]
        layer: String,
        // clap parses a value starting with a hyphen, like a western bbox, as a flag
        #[arg(
            long,
            allow_hyphen_values = true,
            help = "minx,miny,maxx,maxy in the units of --crs"
        )]
        bbox: String,
        #[arg(long, help = "EPSG:<code> of the output")]
        crs: String,
        #[arg(long, help = "pixel size in the units of --crs")]
        resolution: f64,
        #[arg(
            long,
            help = "rfc 3339 start/end interval, the source's own when absent"
        )]
        t: Option<String>,
        #[arg(short = 'o', long = "output", help = "a .tif or a .geojson path")]
        output: PathBuf,
    },
    #[command(
        about = "Write the xyz tiles of one layer over a lon/lat bbox as <output>/z/x/y.png, or .mvt for a vector layer, as geoplumb-server serves them"
    )]
    Pyramid {
        #[arg(help = "the layer TOML geoplumb-server reads")]
        layers: PathBuf,
        #[arg(help = "the name of the layer to write")]
        layer: String,
        #[arg(
            long,
            allow_hyphen_values = true,
            help = "west,south,east,north in degrees"
        )]
        bbox: String,
        #[arg(long, default_value_t = 0, help = "the first zoom written")]
        min_zoom: u8,
        #[arg(long, help = "the last zoom written")]
        max_zoom: u8,
        #[arg(
            long,
            help = "rfc 3339 start/end interval, the source's own when absent"
        )]
        t: Option<String>,
        #[arg(
            short = 'o',
            long = "output",
            help = "the directory the tiles go under"
        )]
        output: PathBuf,
    },
    #[command(
        about = "Print the size, bands, sample format, CRS, geotransform, compression, nodata and overviews of a COG"
    )]
    Info {
        #[arg(help = "a COG terrano reads")]
        file: PathBuf,
    },
}

#[tokio::main]
async fn main() {
    let result = match Arguments::parse().command {
        Command::Run {
            layers,
            layer,
            bbox,
            crs,
            resolution,
            t,
            output,
        } => {
            let query = ExportQuery {
                bbox,
                crs,
                resolution,
                t,
            };
            write_layer(&layers, &layer, query, &output).await
        }
        Command::Pyramid {
            layers,
            layer,
            bbox,
            min_zoom,
            max_zoom,
            t,
            output,
        } => {
            let query = PyramidQuery {
                bbox,
                min_zoom,
                max_zoom,
                t,
            };
            write_pyramid(&layers, &layer, query, &output)
                .await
                .map(|written| println!("{written} tiles under {}", output.display()))
        }
        Command::Info { file } => cog_info(&file).map(|text| print!("{text}")),
    };
    if let Err(message) = result {
        eprintln!("geoplumb: {message}");
        std::process::exit(1);
    }
}
