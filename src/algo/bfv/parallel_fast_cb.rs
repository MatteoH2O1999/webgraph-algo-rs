use crate::{prelude::*, utils::Threads};
use anyhow::Result;
use dsi_progress_logger::ProgressLog;
use parallel_frontier::prelude::{Frontier, ParallelIterator};
use rayon::prelude::*;
use std::{borrow::Borrow, sync::atomic::Ordering};
use sux::bits::AtomicBitVec;
use webgraph::traits::RandomAccessGraph;

/// Builder for [`ParallelBreadthFirstVisitFastCB`].
#[derive(Clone)]
pub struct ParallelBreadthFirstVisitFastCBBuilder<'a, G: RandomAccessGraph, T = Threads> {
    graph: &'a G,
    start: usize,
    granularity: usize,
    threads: T,
}

impl<'a, G: RandomAccessGraph> ParallelBreadthFirstVisitFastCBBuilder<'a, G, Threads> {
    /// Constructs a new builder with default parameters for specified graph.
    pub fn new(graph: &'a G) -> Self {
        Self {
            graph,
            start: 0,
            granularity: 1,
            threads: Threads::Default,
        }
    }
}

impl<'a, G: RandomAccessGraph, T> ParallelBreadthFirstVisitFastCBBuilder<'a, G, T> {
    /// Sets the starting node for full visits.
    /// It does nothing for single visits using [`BreadthFirstGraphVisit::visit_from_node`].
    pub fn start(mut self, start: usize) -> Self {
        self.start = start;
        self
    }

    /// Sets the number of nodes in each chunk of the frontier to explore.
    ///
    /// High granularity reduces overhead, but may lead to decreased performance
    /// on graphs with skewed outdegrees.
    pub fn granularity(mut self, granularity: usize) -> Self {
        self.granularity = granularity;
        self
    }

    /// Sets the visit to use the default threadpool.
    pub fn default_threadpool(self) -> ParallelBreadthFirstVisitFastCBBuilder<'a, G> {
        ParallelBreadthFirstVisitFastCBBuilder {
            graph: self.graph,
            start: self.start,
            granularity: self.granularity,
            threads: Threads::Default,
        }
    }

    /// Sets the visit to use a [`rayon::ThreadPool`] with the specified number of threads.
    pub fn num_threads(self, num_threads: usize) -> ParallelBreadthFirstVisitFastCBBuilder<'a, G> {
        ParallelBreadthFirstVisitFastCBBuilder {
            graph: self.graph,
            start: self.start,
            granularity: self.granularity,
            threads: Threads::NumThreads(num_threads),
        }
    }

    /// Sets the visit to use the custop [`rayon::ThreadPool`].
    pub fn threadpool<T2: Borrow<rayon::ThreadPool>>(
        self,
        threadpool: T2,
    ) -> ParallelBreadthFirstVisitFastCBBuilder<'a, G, T2> {
        ParallelBreadthFirstVisitFastCBBuilder {
            graph: self.graph,
            start: self.start,
            granularity: self.granularity,
            threads: threadpool,
        }
    }
}

impl<'a, G: RandomAccessGraph> ParallelBreadthFirstVisitFastCBBuilder<'a, G, Threads> {
    /// Builds the parallel BFV with the builder parameters and consumes the builder.
    pub fn build(self) -> ParallelBreadthFirstVisitFastCB<'a, G> {
        let builder = ParallelBreadthFirstVisitFastCBBuilder {
            graph: self.graph,
            start: self.start,
            granularity: self.granularity,
            threads: self.threads.build(),
        };
        builder.build()
    }
}

impl<'a, G: RandomAccessGraph, T: Borrow<rayon::ThreadPool>>
    ParallelBreadthFirstVisitFastCBBuilder<'a, G, T>
{
    /// Builds the parallel BFV with the builder parameters and consumes the builder.
    pub fn build(self) -> ParallelBreadthFirstVisitFastCB<'a, G, T> {
        ParallelBreadthFirstVisitFastCB {
            graph: self.graph,
            start: self.start,
            granularity: self.granularity,
            visited: AtomicBitVec::new(self.graph.num_nodes()),
            threads: self.threads,
        }
    }
}

/// A simple parallel Breadth First visit on a graph with low memory consumption but with a smaller
/// frontier.
pub struct ParallelBreadthFirstVisitFastCB<
    'a,
    G: RandomAccessGraph,
    T: Borrow<rayon::ThreadPool> = rayon::ThreadPool,
> {
    graph: &'a G,
    start: usize,
    granularity: usize,
    visited: AtomicBitVec,
    threads: T,
}

impl<'a, G: RandomAccessGraph + Sync, T: Borrow<rayon::ThreadPool>> BreadthFirstGraphVisit
    for ParallelBreadthFirstVisitFastCB<'a, G, T>
{
    fn visit_from_node_filtered<
        C: Fn(usize, usize, usize, usize) + Sync,
        F: Fn(usize, usize, usize, usize) -> bool + Sync,
    >(
        &mut self,
        callback: C,
        filter: F,
        visit_root: usize,
        pl: &mut impl ProgressLog,
    ) -> Result<()> {
        if self.visited.get(visit_root, Ordering::Relaxed)
            || !filter(visit_root, visit_root, visit_root, 0)
        {
            return Ok(());
        }

        let mut next_frontier = Frontier::with_threads(self.threads.borrow(), Some(4));

        self.threads.borrow().join(
            || next_frontier.push(visit_root),
            || {
                self.visited.set(visit_root, true, Ordering::Relaxed);
                callback(visit_root, visit_root, visit_root, 0);
            },
        );

        let mut distance = 1;

        // Visit the connected component
        while !next_frontier.is_empty() {
            let current_frontier = next_frontier;
            let current_len = current_frontier.len();
            next_frontier = Frontier::with_threads(self.threads.borrow(), Some(4));
            self.threads.borrow().install(|| {
                current_frontier
                    .par_iter()
                    .chunks(self.granularity)
                    .for_each(|chunk| {
                        chunk.into_iter().for_each(|&node| {
                            self.graph.successors(node).into_iter().for_each(|succ| {
                                if filter(succ, node, visit_root, distance)
                                    && !self.visited.swap(succ, true, Ordering::Relaxed)
                                {
                                    callback(succ, node, visit_root, distance);
                                    next_frontier.push(succ);
                                }
                            })
                        })
                    });
            });
            pl.update_with_count(current_len);
            distance += 1;
        }

        Ok(())
    }

    fn visit_graph_filtered<
        C: Fn(usize, usize, usize, usize) + Sync,
        F: Fn(usize, usize, usize, usize) -> bool + Sync,
    >(
        &mut self,
        callback: C,
        filter: F,
        pl: &mut impl ProgressLog,
    ) -> Result<()> {
        let num_threads = self.threads.borrow().current_num_threads();

        pl.expected_updates(Some(self.graph.num_nodes()));
        pl.start("Visiting graph with a parallel BFV...");
        pl.info(format_args!(
            "Using {} threads with block size {}...",
            num_threads, self.granularity
        ));

        for i in 0..self.graph.num_nodes() {
            let index = (i + self.start) % self.graph.num_nodes();
            self.visit_from_node_filtered(&callback, &filter, index, pl)?;
        }

        pl.done();

        Ok(())
    }
}

impl<'a, G: RandomAccessGraph + Sync, T: Borrow<rayon::ThreadPool>> ReusableBreadthFirstGraphVisit
    for ParallelBreadthFirstVisitFastCB<'a, G, T>
{
    #[inline(always)]
    fn reset(&mut self) -> Result<()> {
        self.visited.fill(false, Ordering::Relaxed);
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use anyhow::Context;
    use webgraph::prelude::BvGraph;

    #[test]
    fn test_parallel_low_mem_bfv_with_parameters() -> Result<()> {
        let graph = BvGraph::with_basename("tests/graphs/cnr-2000")
            .load()
            .with_context(|| "Cannot load graph")?;
        let visit = ParallelBreadthFirstVisitFastCBBuilder::new(&graph)
            .start(10)
            .granularity(2)
            .build();

        assert_eq!(visit.start, 10);
        assert_eq!(visit.granularity, 2);

        Ok(())
    }

    #[test]
    fn test_parallel_bfv_low_mem_with_start() -> Result<()> {
        let graph = BvGraph::with_basename("tests/graphs/cnr-2000")
            .load()
            .with_context(|| "Cannot load graph")?;
        let visit = ParallelBreadthFirstVisitFastCBBuilder::new(&graph)
            .start(10)
            .build();

        assert_eq!(visit.start, 10);
        assert_eq!(visit.granularity, 1);

        Ok(())
    }

    #[test]
    fn test_parallel_bfv_low_mem_with_granularity() -> Result<()> {
        let graph = BvGraph::with_basename("tests/graphs/cnr-2000")
            .load()
            .with_context(|| "Cannot load graph")?;
        let visit = ParallelBreadthFirstVisitFastCBBuilder::new(&graph)
            .granularity(10)
            .build();

        assert_eq!(visit.start, 0);
        assert_eq!(visit.granularity, 10);

        Ok(())
    }

    #[test]
    fn test_parallel_low_mem_bfv_new() -> Result<()> {
        let graph = BvGraph::with_basename("tests/graphs/cnr-2000")
            .load()
            .with_context(|| "Cannot load graph")?;
        let visit = ParallelBreadthFirstVisitFastCBBuilder::new(&graph).build();

        assert_eq!(visit.start, 0);
        assert_eq!(visit.granularity, 1);

        Ok(())
    }
}
