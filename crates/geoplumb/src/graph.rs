//! pull dag: transforms have one upstream, fanin nodes several, fan-out is
//! several nodes sharing one parent. nodes are added parents-first, but
//! auto-plugged reprojects are appended after their consumers, so walk in
//! `topo_order`, not index order

use crate::element::{Fanin, Source, Transform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

pub(crate) enum Node {
    Source(Box<dyn Source>),
    Transform {
        parent: NodeId,
        element: Box<dyn Transform>,
    },
    Fanin {
        parents: Vec<NodeId>,
        element: Box<dyn Fanin>,
    },
}

#[derive(Default)]
pub struct Graph {
    pub(crate) nodes: Vec<Node>,
}

impl Graph {
    pub fn new() -> Self {
        Graph::default()
    }

    pub fn add_source(&mut self, source: Box<dyn Source>) -> NodeId {
        self.nodes.push(Node::Source(source));
        NodeId(self.nodes.len() - 1)
    }

    /// panics if `parent` is not a node of this graph
    pub fn add_transform(&mut self, parent: NodeId, element: Box<dyn Transform>) -> NodeId {
        assert!(parent.0 < self.nodes.len(), "unknown parent node");
        self.nodes.push(Node::Transform { parent, element });
        NodeId(self.nodes.len() - 1)
    }

    /// panics on fewer than two parents or a parent not of this graph
    pub fn add_fanin(&mut self, parents: &[NodeId], element: Box<dyn Fanin>) -> NodeId {
        assert!(parents.len() >= 2, "fanin needs at least two inputs");
        assert!(
            parents.iter().all(|p| p.0 < self.nodes.len()),
            "unknown parent node"
        );
        self.nodes.push(Node::Fanin {
            parents: parents.to_vec(),
            element,
        });
        NodeId(self.nodes.len() - 1)
    }

    pub(crate) fn parents(&self, id: NodeId) -> Vec<NodeId> {
        match &self.nodes[id.0] {
            Node::Source(_) => Vec::new(),
            Node::Transform { parent, .. } => vec![*parent],
            Node::Fanin { parents, .. } => parents.clone(),
        }
    }

    pub(crate) fn children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match n {
                Node::Transform { parent, .. } if *parent == id => Some(NodeId(i)),
                Node::Fanin { parents, .. } if parents.contains(&id) => Some(NodeId(i)),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// parents-first order. the graph is acyclic by construction: an added
    /// node only points at existing nodes, and a splice points the new node
    /// at the old parent
    pub(crate) fn topo_order(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut order = Vec::with_capacity(n);
        let mut placed = vec![false; n];
        while order.len() < n {
            let before = order.len();
            for i in 0..n {
                if !placed[i] && self.parents(NodeId(i)).iter().all(|p| placed[p.0]) {
                    placed[i] = true;
                    order.push(i);
                }
            }
            assert!(order.len() > before, "graph has a cycle");
        }
        order
    }
}
