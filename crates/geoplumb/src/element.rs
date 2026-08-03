//! element traits. sources answer window reads, transforms declare a caps
//! constraint, rewrite the upstream request (halo, inverse projection) and
//! compute chunks. compute is synchronous by contract, the engine offloads
//! it to a blocking pool when a tokio runtime is present

use crate::caps::{Caps, Constraint};
use crate::chunk::RasterChunk;
use crate::error::Result;
use crate::window::{Bbox, GridSpec, WindowReq};
use futures::future::BoxFuture;

pub trait Source: Send + Sync {
    /// must be `Constraint::Produces`
    fn constraint(&self) -> Constraint;

    /// native pixel grid anchoring this source's resolution ladder
    fn grid(&self) -> GridSpec;

    /// req is chunk-aligned to `grid()` by the engine, the response must
    /// cover exactly that grid window
    fn read<'a>(&'a self, req: &'a WindowReq) -> BoxFuture<'a, Result<RasterChunk>>;
}

pub trait Transform: Send + Sync {
    fn constraint(&self) -> Constraint;

    /// called once after the solve with the fixated link caps
    fn configure(&mut self, input: &Caps, output: &Caps) -> Result<()> {
        let _ = (input, output);
        Ok(())
    }

    /// this node's grid given the upstream grid, identity unless the
    /// transform changes crs or resolution
    fn output_grid(&self, input: &GridSpec) -> GridSpec {
        *input
    }

    /// the upstream window needed to compute `out`: widen by kernel halo,
    /// inverse-project across a crs change
    fn plan(&self, out: &WindowReq) -> WindowReq;

    /// forward image of a dirty region at a given chunk resolution, for
    /// invalidation walks. widen by the halo, project across a crs change
    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        let _ = resolution;
        *dirty
    }

    /// produce exactly the `out` grid from an input chunk covering at
    /// least `plan(out)`
    fn compute(&self, out: &WindowReq, input: &RasterChunk) -> Result<RasterChunk>;
}
