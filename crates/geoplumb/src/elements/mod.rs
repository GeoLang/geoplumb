pub mod algebra;
pub mod band_math;
pub mod cog;
pub mod mosaic;
pub mod points;
pub mod reproject;
pub mod source;
pub mod stac;
pub mod tensor;
pub mod terrain;
pub mod vec_ops;
pub mod vector;

pub use algebra::{Combine, MapAlgebra};
pub use band_math::BandMath;
pub use cog::{CogSrc, HttpRange};
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
