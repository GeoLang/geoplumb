pub mod algebra;
pub mod band_math;
pub mod cog;
pub mod copc;
#[cfg(feature = "gdal")]
pub mod gdal;
pub mod gpkg;
pub mod mosaic;
pub mod points;
pub mod reproject;
pub mod source;
pub mod stac;
pub mod tensor;
pub mod terrain;
pub mod vec_ops;
pub mod vector;
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(feature = "gdal")]
pub use self::gdal::GdalSrc;
#[cfg(feature = "wgpu")]
pub use self::wgpu::{Download, Upload, WgpuConvolve};
pub use algebra::{Combine, MapAlgebra};
pub use band_math::BandMath;
pub use cog::{CogSrc, HttpRange};
pub use copc::{CopcHttpRange, CopcSrc};
pub use gpkg::GpkgSrc;
pub use mosaic::Mosaic;
pub use points::{IdwGrid, LasSrc};
pub use reproject::{Reproject, VecReproject};
pub use source::RasterSrc;
pub use stac::{Composite, StacSearch, StacSrc};
pub use tensor::{TensorConv, ToRaster, ToTensor};
pub use vec_ops::{VecClip, VecFilter, VecSchema};
pub use vector::{Burn, Rasterize, VecSrc};

/// adapters every graph starts with, see `Graph::register_adapter`
pub(crate) fn default_adapters() -> Vec<crate::element::Adapter> {
    vec![Reproject::adapter(), VecReproject::adapter()]
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
