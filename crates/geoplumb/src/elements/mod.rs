pub mod algebra;
pub mod cog;
pub mod mosaic;
pub mod reproject;
pub mod source;
pub mod terrain;

pub use algebra::{Combine, MapAlgebra};
pub use cog::{CogSrc, HttpRange};
pub use mosaic::Mosaic;
pub use reproject::Reproject;
pub use source::RasterSrc;

/// adapters every graph starts with, see `Graph::register_adapter`
pub(crate) fn default_adapters() -> Vec<crate::element::Adapter> {
    vec![Reproject::adapter()]
}
pub use terrain::{Hillshade, Slope};
