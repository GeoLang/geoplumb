// adapted from glass2glass g2g-core/src/runtime/solver.rs (MPL-2.0)
//! caps solver over the pull dag: arc consistency, then fixation.
//!
//! forward sweeps narrow every node's output link through its constraint,
//! backward sweeps intersect each link with what all consumers can still
//! accept. fanin couples siblings through their shared consumer, so the
//! sweeps repeat until the links stop changing. fixation then assigns
//! concrete caps source-first by backtracking search: a diamond can leave
//! preference orders that disagree between branches, where greedy per-link
//! fixation picks a jointly impossible combination. the greedy choice is
//! tried first, so chains and plain fan-out fixate as before

use crate::caps::{Caps, CapsPattern, CapsSet, Constraint, RasterPattern, SetField};
use crate::error::{Error, Result};
use crate::graph::{Graph, Node, NodeId};

fn constraint_of(graph: &Graph, id: NodeId) -> Constraint {
    match &graph.nodes[id.0] {
        Node::Source(s) => s.constraint(),
        Node::Transform { element, .. } => element.constraint(),
        Node::Fanin { element, .. } => element.constraint(),
    }
}

/// one fixated caps per node's output link
pub fn solve(graph: &Graph) -> Result<Vec<Caps>> {
    let n = graph.len();
    let constraints: Vec<Constraint> = (0..n).map(|i| constraint_of(graph, NodeId(i))).collect();
    let parents: Vec<Vec<usize>> = (0..n)
        .map(|i| graph.parents(NodeId(i)).iter().map(|p| p.0).collect())
        .collect();

    let mut links: Vec<CapsSet> = vec![CapsSet::any_raster(); n];
    let max_iters = 8 * n + 4;
    for _ in 0..max_iters {
        let snapshot = links.clone();

        // forward: narrow each output link through the node's constraint.
        // intersecting with the current link keeps earlier backward
        // narrowing, so the sweep stays monotone
        for i in 0..n {
            let set = match parents[i].as_slice() {
                [] => constraints[i].output_set(&CapsSet::any_raster()),
                ps => {
                    let mut upstream = links[ps[0]].clone();
                    for &p in &ps[1..] {
                        upstream = upstream.intersect(&links[p]);
                    }
                    let upstream = upstream.intersect(&constraints[i].input_set());
                    if upstream.is_empty() {
                        return Err(Error::EmptyLink {
                            upstream: NodeId(ps[0]),
                            downstream: NodeId(i),
                            detail: format!(
                                "producers offer nothing the consumer accepts, accepts {:?}",
                                constraints[i].input_set().alternatives
                            ),
                        });
                    }
                    constraints[i].output_set(&upstream)
                }
            };
            let narrowed = links[i].intersect(&set);
            if narrowed.is_empty() {
                let up = parents[i].first().copied().unwrap_or(i);
                return Err(Error::EmptyLink {
                    upstream: NodeId(up),
                    downstream: NodeId(i),
                    detail: "constraint derives no producible output".into(),
                });
            }
            links[i] = narrowed;
        }

        // backward: every consumer narrows each of its parents' links. a
        // fanin pin also carries the sibling links, coupling the branches
        for i in (0..n).rev() {
            for child in graph.children(NodeId(i)) {
                let c = child.0;
                let mut pin = links[c].clone();
                for &q in &parents[c] {
                    if q != i {
                        pin = pin.intersect(&links[q]);
                    }
                }
                let narrowed = constraints[c]
                    .narrow_input(&links[i].intersect(&constraints[c].input_set()), &pin);
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

        if links == snapshot {
            break;
        }
    }

    // fixate source-first, backtracking on a fanin whose parents' greedy
    // choices disagree
    let mut fixed: Vec<Option<Caps>> = vec![None; n];
    let mut deepest_fail = 0usize;
    if !fixate(
        &constraints,
        &parents,
        &links,
        &mut fixed,
        &mut deepest_fail,
        0,
    ) {
        let up = parents[deepest_fail]
            .first()
            .copied()
            .unwrap_or(deepest_fail);
        return Err(Error::Unfixable {
            upstream: NodeId(up),
            downstream: NodeId(deepest_fail),
            remaining: links[deepest_fail].clone(),
        });
    }
    Ok(fixed
        .into_iter()
        .map(|c| c.expect("all assigned"))
        .collect())
}

fn fixate(
    constraints: &[Constraint],
    parents: &[Vec<usize>],
    links: &[CapsSet],
    fixed: &mut Vec<Option<Caps>>,
    deepest_fail: &mut usize,
    i: usize,
) -> bool {
    if i == links.len() {
        return true;
    }
    let candidates = candidates_of(constraints, parents, links, fixed, i);
    if candidates.is_empty() {
        *deepest_fail = (*deepest_fail).max(i);
    }
    for cand in candidates {
        fixed[i] = Some(cand);
        if fixate(constraints, parents, links, fixed, deepest_fail, i + 1) {
            return true;
        }
    }
    fixed[i] = None;
    false
}

/// concrete choices for node `i` against its already-fixed parents, in the
/// link's preference order so the first candidate is the greedy pick
fn candidates_of(
    constraints: &[Constraint],
    parents: &[Vec<usize>],
    links: &[CapsSet],
    fixed: &[Option<Caps>],
    i: usize,
) -> Vec<Caps> {
    let set = match parents[i].as_slice() {
        [] => links[i].clone(),
        ps => {
            let mut pat = pattern_of(fixed[ps[0]].as_ref().expect("parents fixed first"));
            for &q in &ps[1..] {
                pat = pat.intersect(&pattern_of(fixed[q].as_ref().expect("parents fixed first")));
            }
            if pat.is_empty() {
                return Vec::new();
            }
            constraints[i]
                .output_set(&CapsSet::one(CapsPattern::Raster(pat)))
                .intersect(&links[i])
        }
    };
    let mut out: Vec<Caps> = Vec::new();
    for alt in &set.alternatives {
        if let Some(c) = CapsSet::one(alt.clone()).fixate() {
            if !out.contains(&c) {
                out.push(c);
            }
        }
    }
    out
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
