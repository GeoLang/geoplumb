// adapted from glass2glass g2g-core/src/runtime/solver.rs (MPL-2.0)
//! caps solver over the pull dag: arc consistency, then fixation.
//!
//! forward sweeps narrow every node's output link through its constraint,
//! backward sweeps intersect each link with what all consumers can still
//! accept. fanin couples siblings through their shared consumer, on the
//! fields its constraint couples, so the sweeps repeat until the links stop
//! changing. fixation then assigns concrete caps source-first by
//! backtracking search: a diamond can leave
//! preference orders that disagree between branches, where greedy per-link
//! fixation picks a jointly impossible combination. the greedy choice is
//! tried first, so chains and plain fan-out fixate as before.
//!
//! an empty link is not always a failure: registered adapters (transforms
//! whose template constraint leaves their retargeted fields free) are
//! tried against the failing link, and one whose bridged output meets the
//! demand is spliced in before re-solving. the solver never knows which
//! caps field an adapter fixes, the element's constraint declares it

use crate::caps::{Caps, CapsSet, Constraint, FieldMask, conflicting_fields, meet};
use crate::element::Adapter;
use crate::error::{Error, Result};
use crate::graph::{Graph, Node, NodeId};

fn constraint_of(graph: &Graph, id: NodeId) -> Constraint {
    match &graph.nodes[id.0] {
        Node::Source(s) => s.constraint(),
        Node::Transform { element, .. } => element.constraint(),
        Node::Fanin { element, .. } => element.constraint(),
    }
}

/// one fixated caps per node's output link. splices adapter elements into
/// the graph wherever they bridge a link that fails negotiation
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
                element,
            } => splice(graph, parent, child, element),
        }
    }
    Err(Error::InvalidGraph(
        "auto-plug did not converge, an adapter keeps failing its own link".into(),
    ))
}

enum Outcome {
    Solved(Vec<Caps>),
    /// the child-input link is empty but `element` bridges it when spliced
    /// between `parent` and `child`
    NeedsPlug {
        parent: usize,
        child: usize,
        element: Box<dyn crate::element::Transform>,
    },
}

fn splice(
    graph: &mut Graph,
    parent: usize,
    child: usize,
    element: Box<dyn crate::element::Transform>,
) {
    let plug = graph.add_transform(NodeId(parent), element);
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

    let mut links: Vec<CapsSet> = vec![CapsSet::any(); n];
    let max_iters = 8 * n + 4;
    for _ in 0..max_iters {
        let snapshot = links.clone();

        // forward: narrow each output link through the node's constraint.
        // intersecting with the current link keeps earlier backward
        // narrowing, so the sweep stays monotone
        for &i in &topo {
            let set = match parents[i].as_slice() {
                [] => constraints[i].output_set(&CapsSet::any()),
                ps => {
                    let coupling = constraints[i].coupled_fields();
                    let mut upstream = links[ps[0]].clone();
                    for &p in &ps[1..] {
                        let joined = meet(&upstream, &links[p], coupling);
                        if joined.is_empty() {
                            return plug_or_fail(
                                &graph.adapters,
                                &upstream,
                                &links[p],
                                coupling,
                                p,
                                i,
                            );
                        }
                        upstream = joined;
                    }
                    let accepted = upstream.intersect(&constraints[i].input_set());
                    if accepted.is_empty() {
                        return plug_or_fail(
                            &graph.adapters,
                            &constraints[i].input_set(),
                            &upstream,
                            None,
                            ps[0],
                            i,
                        );
                    }
                    constraints[i].output_set(&accepted)
                }
            };
            let narrowed = links[i].intersect(&set);
            if narrowed.is_empty() {
                let up = parents[i].first().copied().unwrap_or(i);
                return Err(empty_link(
                    up,
                    i,
                    "constraint derives no producible output",
                    &links[i],
                    &set,
                    None,
                ));
            }
            links[i] = narrowed;
        }

        // backward: every consumer narrows each of its parents' links. a
        // fanin also meets each parent link with its siblings on the
        // fields its constraint couples
        for &i in topo.iter().rev() {
            for child in graph.children(NodeId(i)) {
                let c = child.0;
                let input_set = constraints[c].input_set();
                let accepted = links[i].intersect(&input_set);
                if accepted.is_empty() {
                    return Err(empty_link(
                        i,
                        c,
                        "the consumer accepts none of the producer's caps",
                        &links[i],
                        &input_set,
                        None,
                    ));
                }
                let coupling = constraints[c].coupled_fields();
                let mut narrowed = constraints[c].narrow_input(&accepted, &links[c]);
                if narrowed.is_empty() {
                    return Err(empty_link(
                        i,
                        c,
                        "backward narrowing left no common caps",
                        &accepted,
                        &links[c],
                        coupling,
                    ));
                }
                for &q in &parents[c] {
                    if q == i {
                        continue;
                    }
                    let joined = meet(&narrowed, &links[q], coupling);
                    if joined.is_empty() {
                        return Err(empty_link(
                            i,
                            c,
                            "fan-in inputs share no caps",
                            &narrowed,
                            &links[q],
                            coupling,
                        ));
                    }
                    narrowed = joined;
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

fn empty_link(
    upstream: usize,
    downstream: usize,
    what: &str,
    onto: &CapsSet,
    from: &CapsSet,
    coupling: Option<FieldMask>,
) -> Error {
    Error::EmptyLink {
        upstream: NodeId(upstream),
        downstream: NodeId(downstream),
        detail: format!(
            "{what}{}",
            no_common_value(&conflicting_fields(onto, from, coupling))
        ),
    }
}

fn no_common_value(fields: &[&str]) -> String {
    match fields {
        [] => String::new(),
        fields => format!(", no common value for {}", fields.join(", ")),
    }
}

/// try each registered adapter against the failing link: one whose bridged
/// output meets the demand gets spliced, in registration order, taking the
/// bridged set's first pattern. otherwise the link fails as before
fn plug_or_fail(
    adapters: &[Adapter],
    demand: &CapsSet,
    offer: &CapsSet,
    coupling: Option<FieldMask>,
    parent: usize,
    child: usize,
) -> Result<Outcome> {
    for adapter in adapters {
        let bridged = meet(&adapter.template.output_set(offer), demand, coupling);
        for alt in &bridged.alternatives {
            if let Some(element) = (adapter.build)(alt) {
                return Ok(Outcome::NeedsPlug {
                    parent,
                    child,
                    element,
                });
            }
        }
    }
    Err(Error::EmptyLink {
        upstream: NodeId(parent),
        downstream: NodeId(child),
        detail: format!(
            "producer offers {:?}, consumer side needs {:?}, no adapter bridges it{}",
            offer.alternatives,
            demand.alternatives,
            no_common_value(&conflicting_fields(demand, offer, coupling))
        ),
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
            let fixed_pattern =
                |p: usize| CapsSet::one(fixed[p].as_ref().expect("parents fixed first").pattern());
            let coupling = constraints[i].coupled_fields();
            let mut joined = fixed_pattern(ps[0]);
            for &q in &ps[1..] {
                joined = meet(&joined, &fixed_pattern(q), coupling);
                if joined.is_empty() {
                    return Vec::new();
                }
            }
            constraints[i].output_set(&joined).intersect(&links[i])
        }
    };
    let mut out: Vec<Caps> = Vec::new();
    for alt in &set.alternatives {
        if let Some(c) = CapsSet::one(alt.clone()).fixate()
            && !out.contains(&c)
        {
            out.push(c);
        }
    }
    out
}
