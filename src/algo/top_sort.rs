use crate::{algo::visits::dfv::*, algo::visits::SeqVisit};
use dsi_progress_logger::ProgressLog;
use std::mem::MaybeUninit;
use webgraph::traits::RandomAccessGraph;

/// Returns the node of the graph in topological-sort order, if the graph is acyclic.
///
/// Otherwise, the order reflects the exit times from a depth-first visit of the graph.
pub fn run(graph: impl RandomAccessGraph, pl: &mut impl ProgressLog) -> Box<[usize]> {
    let mut visit =
        SingleThreadedDepthFirstVisit::<TwoState, std::convert::Infallible, _>::new(&graph);
    let num_nodes = graph.num_nodes();
    pl.item_name("node");
    pl.expected_updates(Some(num_nodes));
    pl.start("Computing topological sort");

    let mut topol_sort = vec![MaybeUninit::uninit(); num_nodes];
    let mut pos = num_nodes;

    visit
        .visit(
            |&Args {
                 node,
                 pred: _pred,
                 root: _root,
                 depth: _depth,
                 event,
             }| {
                if event == Event::Completed {
                    pos -= 1;
                    topol_sort[pos].write(node);
                }

                Ok(())
            },
            |_| true,
            pl,
        )
        .unwrap(); // Safe as infallible

    pl.done();
    // SAFETY: we write in each element of top_sort
    unsafe { std::mem::transmute::<Vec<MaybeUninit<usize>>, Vec<usize>>(topol_sort) }
        .into_boxed_slice()
}