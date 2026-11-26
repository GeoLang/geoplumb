pub mod algebra;
pub mod band_math;
pub mod cast;
pub mod cog;
pub mod copc;
#[cfg(feature = "gdal")]
pub mod gdal;
pub mod gpkg;
mod grib;
pub mod hydrology;
pub mod mosaic;
pub mod netcdf;
pub mod points;
pub mod raster_vector;
pub mod reproject;
pub mod source;
pub mod stac;
mod stored_grid;
pub mod tensor;
pub mod terrain;
pub mod vec_ops;
pub mod vector;
pub mod viewshed;
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "gdal")]
pub use self::gdal::GdalSrc;
#[cfg(feature = "wgpu")]
pub use self::wgpu::{Download, Upload, WgpuConvolve};
pub use algebra::{Combine, MapAlgebra};
pub use band_math::BandMath;
pub use cast::Cast;
pub use cog::{CogSrc, HttpRange};
pub use copc::{CopcHttpRange, CopcSrc};
pub use gpkg::GpkgSrc;
pub use hydrology::{FlowAccumulation, FlowDirection, StreamOrder, Watershed};
pub use mosaic::Mosaic;
pub use netcdf::NetcdfSrc;
pub use points::{IdwGrid, LasSrc};
pub use raster_vector::{Contours, Polygonize};
pub use reproject::Reproject;
pub use source::RasterSrc;
pub use stac::{Composite, ItemMask, StacSearch, StacSrc};
pub use tensor::{TensorConv, ToRaster, ToTensor};
pub use vec_ops::{VecClip, VecFilter, VecSchema};
pub use vector::{Burn, Rasterize, VecSrc};
pub use viewshed::Viewshed;

pub(crate) fn default_adapters() -> Vec<crate::element::Adapter> {
    vec![Reproject::adapter(), Cast::adapter()]
}
pub use terrain::{Aspect, Hillshade, Slope};
pub mod mask;
pub use mask::QualityMask;
pub mod focal;
pub use focal::{Focal, FocalOp};

/// length and modification time of a file a source reads by path, so a
/// rewritten file gets a new identity
pub(crate) fn file_stamp(path: &std::path::Path) -> String {
    let metadata = match std::fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(e) => return format!("file_error={:?}", e.to_string()),
    };
    let modified = metadata
        .modified()
        .map_err(|e| e.to_string())
        .and_then(|t| {
            t.duration_since(std::time::UNIX_EPOCH)
                .map_err(|e| e.to_string())
        });
    match modified {
        Ok(since_epoch) => format!(
            "file_bytes={} file_modified_ns={}",
            metadata.len(),
            since_epoch.as_nanos()
        ),
        Err(e) => format!("file_bytes={} file_modified_error={e:?}", metadata.len()),
    }
}
