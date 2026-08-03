use crate::caps::CapsSet;
use crate::graph::NodeId;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("negotiation failed between node {upstream:?} and node {downstream:?}: {detail}")]
    EmptyLink {
        upstream: NodeId,
        downstream: NodeId,
        detail: String,
    },
    #[error("link between {upstream:?} and {downstream:?} cannot fixate: {remaining:?}")]
    Unfixable {
        upstream: NodeId,
        downstream: NodeId,
        remaining: CapsSet,
    },
    #[error("graph is not a valid pull dag: {0}")]
    InvalidGraph(String),
    #[error("source read failed: {0}")]
    Source(String),
    #[error("compute failed at node {node:?}: {detail}")]
    Compute { node: NodeId, detail: String },
    #[error("reprojection failed: {0}")]
    Projection(String),
    #[error(transparent)]
    Terrano(#[from] terrano_core::Error),
}

pub type Result<T> = core::result::Result<T, Error>;
