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

pub trait Fanin: Send + Sync {
    /// applies to every input link and the output link alike, so all pins
    /// negotiate to matching caps. `Identity` is the usual choice
    fn constraint(&self) -> Constraint;

    /// called once after the solve with the fixated caps of every input
    /// link, wiring order, and the output link
    fn configure(&mut self, inputs: &[Caps], output: &Caps) -> Result<()> {
        let _ = (inputs, output);
        Ok(())
    }

    /// output grid given the input grids, the finest input by default
    fn output_grid(&self, inputs: &[GridSpec]) -> GridSpec {
        *inputs
            .iter()
            .min_by(|a, b| a.base_resolution.total_cmp(&b.base_resolution))
            .expect("fanin has inputs")
    }

    /// the window needed from input `k` to compute `out`
    fn plan(&self, out: &WindowReq, k: usize) -> WindowReq {
        let _ = k;
        *out
    }

    /// forward image of a dirty region arriving from any input
    fn spread(&self, dirty: &Bbox, resolution: f64) -> Bbox {
        let _ = resolution;
        *dirty
    }

    /// produce exactly the `out` grid from one chunk per input, wiring
    /// order, each covering at least `plan(out, k)` on its own alignment
    fn compute(&self, out: &WindowReq, inputs: &[RasterChunk]) -> Result<RasterChunk>;
}
