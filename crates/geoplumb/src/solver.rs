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
//! tried first, so chains and plain fan-out fixate as before.
//!
//! a link empty only because of crs is not a failure: the solver splices a
//! reproject onto the offending edge and re-solves, so mixed-crs graphs
//! negotiate without explicit wiring

use crate::caps::{Caps, CapsPattern, CapsSet, Constraint, Crs, RasterPattern, SetField};
use crate::elements::Reproject;
use crate::error::{Error, Result};
use crate::graph::{Graph, Node, NodeId};

fn constraint_of(graph: &Graph, id: NodeId) -> Constraint {
    match &graph.nodes[id.0] {
        Node::Source(s) => s.constraint(),
        Node::Transform { element, .. } => element.constraint(),
        Node::Fanin { element, .. } => element.constraint(),
    }
}

/// one fixated caps per node's output link. splices reproject nodes into
/// the graph wherever a link is empty only because of crs
pub fn solve(graph: &mut Graph) -> Result<Vec<Caps>> {
    let edges: usize = (0..graph.len())
        .map(|i| graph.parents(NodeId(i)).len())
        .sum();
    for _ in 0..=edges {
        match try_solve(graph)? {
            Outcome::Solved(caps) => return Ok(caps),
            Outcome::NeedsPlug {
                parent,
                child,
                target,
            } => splice(graph, parent, child, target),
        }
    }
    unreachable!("every splice satisfies one edge's crs for good");
}

enum Outcome {
    Solved(Vec<Caps>),
    /// the child-input link is empty only because of crs: reprojecting
    /// `parent` to `target` lets negotiation proceed
    NeedsPlug {
        parent: usize,
        child: usize,
        target: Crs,
    },
}

fn splice(graph: &mut Graph, parent: usize, child: usize, target: Crs) {
    let plug = graph.add_transform(NodeId(parent), Box::new(Reproject::new(target)));
    match &mut graph.nodes[child] {
        Node::Transform { parent: p, .. } => *p = plug,
        Node::Fanin { parents, .. } => {
            for p in parents.iter_mut() {
                if p.0 == parent {
                    *p = plug;
                }
            }
        }
        Node::Source(_) => unreachable!("sources have no inputs"),
    }
}

fn try_solve(graph: &Graph) -> Result<Outcome> {
    let n = graph.len();
    let topo = graph.topo_order();
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
        for &i in &topo {
            let set = match parents[i].as_slice() {
                [] => constraints[i].output_set(&CapsSet::any_raster()),
                ps => {
                    let mut upstream = links[ps[0]].clone();
                    for &p in &ps[1..] {
                        let joined = upstream.intersect(&links[p]);
                        if joined.is_empty() {
                            return plug_or_fail(&upstream, &links[p], p, i);
                        }
                        upstream = joined;
                    }
                    let accepted = upstream.intersect(&constraints[i].input_set());
                    if accepted.is_empty() {
                        return plug_or_fail(&constraints[i].input_set(), &upstream, ps[0], i);
                    }
                    constraints[i].output_set(&accepted)
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
        for &i in topo.iter().rev() {
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

    // fixate parents-first, backtracking on a fanin whose parents' greedy
    // choices disagree
    let mut fixed: Vec<Option<Caps>> = vec![None; n];
    let mut deepest_fail = 0usize;
    if !fixate(
        &topo,
        &constraints,
        &parents,
        &links,
        &mut fixed,
        &mut deepest_fail,
        0,
    ) {
        let node = topo[deepest_fail];
        let up = parents[node].first().copied().unwrap_or(node);
        return Err(Error::Unfixable {
            upstream: NodeId(up),
            downstream: NodeId(node),
            remaining: links[node].clone(),
        });
    }
    Ok(Outcome::Solved(
        fixed
            .into_iter()
            .map(|c| c.expect("all assigned"))
            .collect(),
    ))
}

/// a link empty only because of crs asks for a plug, anything else fails
fn plug_or_fail(demand: &CapsSet, offer: &CapsSet, parent: usize, child: usize) -> Result<Outcome> {
    if !without_crs(demand)
        .intersect(&without_crs(offer))
        .is_empty()
    {
        if let Some(target) = first_crs(demand) {
            return Ok(Outcome::NeedsPlug {
                parent,
                child,
                target,
            });
        }
    }
    Err(Error::EmptyLink {
        upstream: NodeId(parent),
        downstream: NodeId(child),
        detail: format!(
            "producer offers {:?}, consumer side needs {:?}",
            offer.alternatives, demand.alternatives
        ),
    })
}

fn without_crs(set: &CapsSet) -> CapsSet {
    CapsSet {
        alternatives: set
            .alternatives
            .iter()
            .map(|p| {
                let mut r = p.raster().clone();
                r.crs = SetField::Any;
                CapsPattern::Raster(r)
            })
            .collect(),
    }
}

fn first_crs(set: &CapsSet) -> Option<Crs> {
    set.alternatives.iter().find_map(|p| match &p.raster().crs {
        SetField::OneOf(v) => v.first().copied(),
        SetField::Any => None,
    })
}

#[allow(clippy::too_many_arguments)]
fn fixate(
    topo: &[usize],
    constraints: &[Constraint],
    parents: &[Vec<usize>],
    links: &[CapsSet],
    fixed: &mut Vec<Option<Caps>>,
    deepest_fail: &mut usize,
    pos: usize,
) -> bool {
    if pos == topo.len() {
        return true;
    }
    let i = topo[pos];
    let candidates = candidates_of(constraints, parents, links, fixed, i);
    if candidates.is_empty() {
        *deepest_fail = (*deepest_fail).max(pos);
    }
    for cand in candidates {
        fixed[i] = Some(cand);
        if fixate(
            topo,
            constraints,
            parents,
            links,
            fixed,
            deepest_fail,
            pos + 1,
        ) {
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
