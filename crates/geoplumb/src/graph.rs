//! single-input pull dag: every node has at most one upstream, fan-out is
//! several nodes sharing one parent. fan-in (mosaic, multi-input algebra)
//! is a later milestone and brings back the backtracking fixation from the
//! g2g solver

use crate::element::{Source, Transform};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub(crate) usize);

pub(crate) enum Node {
    Source(Box<dyn Source>),
    Transform {
        parent: NodeId,
        element: Box<dyn Transform>,
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

    pub(crate) fn parent(&self, id: NodeId) -> Option<NodeId> {
        match &self.nodes[id.0] {
            Node::Source(_) => None,
            Node::Transform { parent, .. } => Some(*parent),
        }
    }

    pub(crate) fn children(&self, id: NodeId) -> Vec<NodeId> {
        self.nodes
            .iter()
            .enumerate()
            .filter_map(|(i, n)| match n {
                Node::Transform { parent, .. } if *parent == id => Some(NodeId(i)),
                _ => None,
            })
            .collect()
    }

    pub(crate) fn len(&self) -> usize {
        self.nodes.len()
    }
}
