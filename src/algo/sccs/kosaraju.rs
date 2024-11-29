use std::ops::ControlFlow::Continue;

use super::BasicSccs;
use crate::{
    algo::{
        top_sort,
        visits::{Done, Sequential},
    },
    prelude::depth_first::*,
};
use dsi_progress_logger::ProgressLog;
use webgraph::traits::RandomAccessGraph;

/// Computes the strongly connected components of a graph using Kosaraju's algorithm.
///
/// # Arguments
/// * `graph`: the graph.
/// * `transpose`: the transposed of `graph`.
/// * `pl`: a progress logger.
pub fn kosaraju(
    graph: impl RandomAccessGraph,
    transpose: impl RandomAccessGraph,
    pl: &mut impl ProgressLog,
) -> BasicSccs {
    let num_nodes = graph.num_nodes();
    pl.item_name("node");
    pl.expected_updates(Some(num_nodes));
    pl.start("Computing strongly connected components...");

    let top_sort = top_sort(&graph, pl);
    let mut number_of_components = 0;
    let mut visit = SeqNoPred::new(&transpose);
    let mut components = vec![0; num_nodes].into_boxed_slice();

    for &node in &top_sort {
        visit
            .visit(
                node,
                |event| {
                    match event {
                        EventNoPred::Previsit { curr, .. } => {
                            components[curr] = number_of_components;
                        }
                        EventNoPred::Done { .. } => {
                            number_of_components += 1;
                        }
                        _ => (),
                    }
                    Continue(())
                },
                pl,
            )
            .done();
    }

    pl.done();

    BasicSccs::new(number_of_components, components)
}
