use crate::prelude::*;
use anyhow::Result;
use dsi_progress_logger::ProgressLog;
use std::{collections::VecDeque, marker::PhantomData};
use sux::bits::BitVec;
use webgraph::traits::RandomAccessGraph;

/// An iterator on nodes that visits a graph with a Breadth First strategy.
pub struct SingleThreadedBreadthFirstIterator<'a, G: RandomAccessGraph, N, F: NodeFactory<Node = N>>
{
    graph: &'a G,
    start: usize,
    cursor: usize,
    node_factory: &'a F,
    visited: BitVec,
    queue: VecDeque<usize>,
    _node_type: PhantomData<N>,
}

/// A simple sequential Breadth First visit on a graph.
///
/// It also implements [`IntoIterator`], so it can be used in `for ... in Visit`.
pub struct SingleThreadedBreadthFirstVisit<
    'a,
    G: RandomAccessGraph,
    N: NodeVisit,
    F: NodeFactory<Node = N>,
> {
    graph: &'a G,
    start: usize,
    node_factory: &'a F,
    _node_type: PhantomData<N>,
}

impl<'a, G: RandomAccessGraph, N, F: NodeFactory<Node = N>> Iterator
    for SingleThreadedBreadthFirstIterator<'a, G, N, F>
{
    type Item = N;
    fn next(&mut self) -> Option<N> {
        let current_node_index = match self.queue.pop_front() {
            None => {
                while self.visited[self.cursor] {
                    self.cursor = (self.cursor + 1) % self.graph.num_nodes();
                    if self.cursor == self.start {
                        return None;
                    }
                }
                self.visited.set(self.cursor, true);
                self.cursor
            }
            Some(node) => node,
        };

        for successor in self.graph.successors(current_node_index) {
            if !self.visited[successor] {
                self.queue.push_back(successor);
                self.visited.set(successor, true);
            }
        }

        Some(self.node_factory.node_from_index(current_node_index))
    }
}

impl<'a, G: RandomAccessGraph, N: NodeVisit, F: NodeFactory<Node = N>>
    SingleThreadedBreadthFirstVisit<'a, G, N, F>
{
    /// Constructs a sequential BFV for the specified graph using the provided node factory.
    ///
    /// # Arguments:
    /// - `graph`: An immutable reference to the graph to visit.
    /// - `node_factory`: An immutable reference to the node factory that produces nodes to visit
    /// from their index.
    pub fn new(graph: &'a G, node_factory: &'a F) -> SingleThreadedBreadthFirstVisit<'a, G, N, F> {
        Self::with_start(graph, node_factory, 0)
    }

    /// Constructs a sequential BFV starting from the node with the specified index in the
    /// provided graph using the provided node factory.
    ///
    /// # Arguments:
    /// - `graph`: An immutable reference to the graph to visit.
    /// - `node_factory`: An immutable reference to the node factory that produces nodes to visit
    /// from their index.
    pub fn with_start(
        graph: &'a G,
        node_factory: &'a F,
        start: usize,
    ) -> SingleThreadedBreadthFirstVisit<'a, G, N, F> {
        SingleThreadedBreadthFirstVisit {
            graph,
            start,
            node_factory,
            _node_type: PhantomData,
        }
    }
}

impl<'a, G: RandomAccessGraph, N: NodeVisit, F: NodeFactory<Node = N>> IntoIterator
    for SingleThreadedBreadthFirstVisit<'a, G, N, F>
{
    type Item = N;
    type IntoIter = SingleThreadedBreadthFirstIterator<'a, G, N, F>;
    fn into_iter(self) -> SingleThreadedBreadthFirstIterator<'a, G, N, F> {
        SingleThreadedBreadthFirstIterator {
            graph: self.graph,
            start: self.start,
            cursor: self.start,
            visited: BitVec::new(self.graph.num_nodes()),
            queue: VecDeque::new(),
            node_factory: self.node_factory,
            _node_type: PhantomData,
        }
    }
}

impl<'a, G: RandomAccessGraph, N: NodeVisit, F: NodeFactory<Node = N>> IntoIterator
    for &SingleThreadedBreadthFirstVisit<'a, G, N, F>
{
    type Item = N;
    type IntoIter = SingleThreadedBreadthFirstIterator<'a, G, N, F>;
    fn into_iter(self) -> SingleThreadedBreadthFirstIterator<'a, G, N, F> {
        SingleThreadedBreadthFirstIterator {
            graph: self.graph,
            start: self.start,
            cursor: self.start,
            visited: BitVec::new(self.graph.num_nodes()),
            queue: VecDeque::new(),
            node_factory: self.node_factory,
            _node_type: PhantomData,
        }
    }
}

impl<'a, G: RandomAccessGraph, N: NodeVisit, F: NodeFactory<Node = N>> IntoIterator
    for &mut SingleThreadedBreadthFirstVisit<'a, G, N, F>
{
    type Item = N;
    type IntoIter = SingleThreadedBreadthFirstIterator<'a, G, N, F>;
    fn into_iter(self) -> SingleThreadedBreadthFirstIterator<'a, G, N, F> {
        SingleThreadedBreadthFirstIterator {
            graph: self.graph,
            start: self.start,
            cursor: self.start,
            visited: BitVec::new(self.graph.num_nodes()),
            queue: VecDeque::new(),
            node_factory: self.node_factory,
            _node_type: PhantomData,
        }
    }
}

impl<'a, G: RandomAccessGraph, N: NodeVisit, F: NodeFactory<Node = N>> GraphVisit<N>
    for SingleThreadedBreadthFirstVisit<'a, G, N, F>
{
    fn visit(self, mut pl: impl ProgressLog) -> Result<N::AccumulatedResult> {
        pl.expected_updates(Some(self.graph.num_nodes()));
        pl.start("Visiting graph with a sequential BFV...");
        let mut result = N::init_result();
        for node in self {
            pl.light_update();
            N::accumulate_result(&mut result, node.visit());
        }
        pl.done();
        Ok(result)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::Context;
    use webgraph::graphs::BVGraph;

    struct Node {
        index: usize,
    }

    struct Factory {}

    impl NodeVisit for Node {
        type VisitResult = usize;
        type AccumulatedResult = Vec<usize>;

        fn init_result() -> Self::AccumulatedResult {
            Vec::new()
        }

        fn accumulate_result(
            partial_result: &mut Self::AccumulatedResult,
            visit_result: Self::VisitResult,
        ) {
            partial_result.push(visit_result)
        }

        fn visit(self) -> Self::VisitResult {
            self.index
        }
    }

    impl NodeFactory for Factory {
        type Node = Node;

        fn node_from_index(&self, node_index: usize) -> Self::Node {
            Node { index: node_index }
        }
    }

    #[test]
    fn test_sequential_bfv_with_start() -> Result<()> {
        let graph = BVGraph::with_basename("tests/graphs/cnr-2000")
            .load()
            .with_context(|| "Cannot load graph")?;
        let factory = Factory {};
        let visit = SingleThreadedBreadthFirstVisit::with_start(&graph, &factory, 10);

        assert_eq!(visit.start, 10);

        Ok(())
    }

    #[test]
    fn test_sequential_bfv_new() -> Result<()> {
        let graph = BVGraph::with_basename("tests/graphs/cnr-2000")
            .load()
            .with_context(|| "Cannot load graph")?;
        let factory = Factory {};
        let visit = SingleThreadedBreadthFirstVisit::new(&graph, &factory);

        assert_eq!(visit.start, 0);

        Ok(())
    }
}
