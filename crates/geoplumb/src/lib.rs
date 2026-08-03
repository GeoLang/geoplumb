//! geoplumb: a pull-based geospatial compute pipeline.
//!
//! build a dag of sources and transforms, negotiate caps once with the
//! solver, then pull windows: a demand (bbox + resolution) flows sink to
//! source, each element rewriting it (halo, inverse projection), and
//! chunks flow back, cached and coalesced per node. batch is a full-extent
//! pull, live is invalidation plus re-pull. see DESIGN.md

pub mod caps;
pub mod chunk;
pub mod element;
pub mod elements;
pub mod encode;
pub mod engine;
pub mod error;
pub mod graph;
pub mod resample;
pub mod solver;
mod spill;
pub mod tile;
pub mod window;

pub use caps::{Caps, CapsSet, Constraint, Crs};
pub use chunk::{Chunk, RasterChunk};
pub use element::{Adapter, Fanin, Source, Transform};
pub use engine::{Engine, Invalidation, materialize};
pub use error::{Error, Result};
pub use graph::{Graph, NodeId};
pub use window::{Bbox, WindowReq};
