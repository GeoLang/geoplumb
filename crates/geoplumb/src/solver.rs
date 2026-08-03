// adapted from glass2glass g2g-core/src/runtime/solver.rs (MPL-2.0)
//! caps solver over the pull dag: arc consistency, then fixation.
//!
//! forward sweep narrows every node's output link through its constraint,
//! backward sweep intersects each node's link with what all consumers can
//! still accept, then a second forward pass fixates source-first so every
//! child fixates against its parent's concrete caps. single-input nodes
//! make the constraint graph a forest, where one sweep each way is
//! complete. fan-in will bring back g2g's backtracking fixation

use crate::caps::{Caps, CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use crate::error::{Error, Result};
use crate::graph::{Graph, Node, NodeId};

fn constraint_of(graph: &Graph, id: NodeId) -> Constraint {
    match &graph.nodes[id.0] {
        Node::Source(s) => s.constraint(),
        Node::Transform { element, .. } => element.constraint(),
    }
}

/// one fixated caps per node's output link
pub fn solve(graph: &Graph) -> Result<Vec<Caps>> {
    let n = graph.len();
    let constraints: Vec<Constraint> = (0..n).map(|i| constraint_of(graph, NodeId(i))).collect();

    // forward: narrow each output link through the node's constraint
    let mut links: Vec<CapsSet> = Vec::with_capacity(n);
    for (i, constraint) in constraints.iter().enumerate() {
        let set = match graph.parent(NodeId(i)) {
            None => constraint.output_set(&CapsSet::any_raster()),
            Some(p) => {
                let upstream = links[p.0].intersect(&constraint.input_set());
                if upstream.is_empty() {
                    return Err(Error::EmptyLink {
                        upstream: p,
                        downstream: NodeId(i),
                        detail: format!(
                            "producer offers {:?}, consumer accepts {:?}",
                            links[p.0].alternatives,
                            constraint.input_set().alternatives
                        ),
                    });
                }
                constraint.output_set(&upstream)
            }
        };
        if set.is_empty() {
            let up = graph.parent(NodeId(i)).unwrap_or(NodeId(i));
            return Err(Error::EmptyLink {
                upstream: up,
                downstream: NodeId(i),
                detail: "constraint derives no producible output".into(),
            });
        }
        links.push(set);
    }

    // backward: every consumer narrows its parent's link
    for i in (0..n).rev() {
        for child in graph.children(NodeId(i)) {
            let narrowed = constraints[child.0].narrow_input(
                &links[i].intersect(&constraints[child.0].input_set()),
                &links[child.0],
            );
            if narrowed.is_empty() {
                return Err(Error::EmptyLink {
                    upstream: NodeId(i),
                    downstream: child,
                    detail: "backward narrowing left no common caps".into(),
                });
            }
            links[i] = narrowed;
        }
    }

    // fixate source-first so children see concrete parent caps
    let mut fixed: Vec<Caps> = Vec::with_capacity(n);
    for i in 0..n {
        let set = match graph.parent(NodeId(i)) {
            None => links[i].clone(),
            Some(p) => {
                let parent_pattern = pattern_of(&fixed[p.0]);
                constraints[i]
                    .output_set(&CapsSet::one(CapsPattern::Raster(parent_pattern)))
                    .intersect(&links[i])
            }
        };
        let caps = set.fixate().ok_or_else(|| Error::Unfixable {
            upstream: graph.parent(NodeId(i)).unwrap_or(NodeId(i)),
            downstream: NodeId(i),
            remaining: set.clone(),
        })?;
        fixed.push(caps);
    }
    Ok(fixed)
}

/// concrete caps as a single-alternative pattern for downstream derivation
fn pattern_of(caps: &Caps) -> RasterPattern {
    let r = caps.raster();
    RasterPattern {
        dtype: SetField::one(r.dtype),
        bands: SetField::one(r.bands),
        crs: SetField::one(r.crs),
        resolution: r.resolution,
        chunk_px: SetField::one(r.chunk_px),
    }
}
