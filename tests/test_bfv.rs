use anyhow::Result;
use dsi_progress_logger::prelude::*;
use no_break::NoBreak;
use std::ops::ControlFlow::Continue;
use std::sync::atomic::{AtomicUsize, Ordering};
use webgraph::{
    labels::Left,
    prelude::{BvGraph, VecGraph},
    traits::{RandomAccessGraph, SequentialLabeling},
};
use webgraph_algo::{algo::visits::*, threads};

fn correct_dists<G: RandomAccessGraph>(graph: &G, start: usize) -> Vec<usize> {
    let mut dists = Vec::new();
    let mut visits = vec![-1; graph.num_nodes()];
    let mut current_frontier = Vec::new();
    let mut next_frontier = Vec::new();

    for i in 0..graph.num_nodes() {
        let start_node = (i + start) % graph.num_nodes();
        if visits[start_node] != -1 {
            continue;
        }
        let mut distance = 1;
        visits[start_node] = 0;
        current_frontier.push(start_node);

        while !current_frontier.is_empty() {
            for node in current_frontier {
                for succ in graph.successors(node) {
                    if visits[succ] == -1 {
                        next_frontier.push(succ);
                        visits[succ] = distance;
                    }
                }
            }
            current_frontier = next_frontier;
            next_frontier = Vec::new();
            distance += 1;
        }
    }

    for dist in visits {
        dists.push(dist.try_into().unwrap());
    }

    dists
}

fn into_non_atomic(v: Vec<AtomicUsize>) -> Vec<usize> {
    let mut res = Vec::new();
    for element in v {
        res.push(element.load(Ordering::Relaxed));
    }
    res
}

macro_rules! test_bfv_algo_seq {
    ($bfv:expr, $name:ident) => {
        mod $name {
            use super::*;

            #[test]
            fn test_simple_graph() -> Result<()> {
                let arcs = vec![
                    (0, 0),
                    (1, 0),
                    (1, 2),
                    (2, 1),
                    (2, 3),
                    (2, 4),
                    (2, 5),
                    (3, 4),
                    (4, 3),
                    (5, 5),
                    (5, 6),
                    (5, 7),
                    (5, 8),
                    (6, 7),
                    (8, 7),
                ];
                let mut g = VecGraph::new();
                for i in 0..9 {
                    g.add_node(i);
                }
                for arc in arcs {
                    g.add_arc(arc.0, arc.1);
                }
                let graph = Left(g);
                let mut visit = $bfv(&graph);
                let dists: Vec<AtomicUsize> = (0..graph.num_nodes())
                    .map(|_| AtomicUsize::new(0))
                    .collect();
                let expected_dists = correct_dists(&graph, 0);

                for node in 0..graph.num_nodes() {
                    visit
                        .visit(
                            node,
                            |event| {
                                if let breadth_first::EventPred::Unknown {
                                    curr, distance, ..
                                } = event
                                {
                                    dists[curr].store(distance, Ordering::Relaxed);
                                }
                                Continue(())
                            },
                            no_logging![],
                        )
                        .continue_value_no_break();
                }

                let actual_dists = into_non_atomic(dists);

                assert_eq!(actual_dists, expected_dists);

                Ok(())
            }

            #[test]
            fn test_cnr_2000() -> Result<()> {
                let graph = BvGraph::with_basename("tests/graphs/cnr-2000").load()?;
                let mut visit = $bfv(&graph);
                let dists: Vec<AtomicUsize> = (0..graph.num_nodes())
                    .map(|_| AtomicUsize::new(0))
                    .collect();
                let expected_dists = correct_dists(&graph, 10000);

                for i in 0..graph.num_nodes() {
                    let node = (i + 10000) % graph.num_nodes();
                    visit
                        .visit(
                            node,
                            |event| {
                                if let breadth_first::EventPred::Unknown {
                                    curr, distance, ..
                                } = event
                                {
                                    dists[curr].store(distance, Ordering::Relaxed);
                                }
                                Continue(())
                            },
                            no_logging![],
                        )
                        .continue_value_no_break();
                }

                let actual_dists = into_non_atomic(dists);

                assert_eq!(actual_dists, expected_dists);

                Ok(())
            }
        }
    };
}

macro_rules! test_bfv_algo_par {
    ($bfv:expr, $name:ident) => {
        mod $name {
            use super::*;

            #[test]
            fn test_simple_graph() -> Result<()> {
                let arcs = vec![
                    (0, 0),
                    (1, 0),
                    (1, 2),
                    (2, 1),
                    (2, 3),
                    (2, 4),
                    (2, 5),
                    (3, 4),
                    (4, 3),
                    (5, 5),
                    (5, 6),
                    (5, 7),
                    (5, 8),
                    (6, 7),
                    (8, 7),
                ];
                let mut g = VecGraph::new();
                for i in 0..9 {
                    g.add_node(i);
                }
                for arc in arcs {
                    g.add_arc(arc.0, arc.1);
                }
                let graph = Left(g);
                let mut visit = $bfv(&graph);
                let dists: Vec<AtomicUsize> = (0..graph.num_nodes())
                    .map(|_| AtomicUsize::new(0))
                    .collect();
                let expected_dists = correct_dists(&graph, 0);

                let t = threads![];

                for node in 0..graph.num_nodes() {
                    visit
                        .par_visit(
                            node,
                            |event| {
                                if let breadth_first::EventPred::Unknown {
                                    curr, distance, ..
                                } = event
                                {
                                    dists[curr].store(distance, Ordering::Relaxed);
                                }
                                Continue(())
                            },
                            &t,
                            no_logging![],
                        )
                        .continue_value_no_break();
                }

                let actual_dists = into_non_atomic(dists);

                assert_eq!(actual_dists, expected_dists);

                Ok(())
            }

            #[test]
            fn test_cnr_2000() -> Result<()> {
                let graph = BvGraph::with_basename("tests/graphs/cnr-2000").load()?;
                let mut visit = $bfv(&graph);
                let dists: Vec<AtomicUsize> = (0..graph.num_nodes())
                    .map(|_| AtomicUsize::new(0))
                    .collect();
                let expected_dists = correct_dists(&graph, 10000);
                let t = threads![];

                for i in 0..graph.num_nodes() {
                    let node = (i + 10000) % graph.num_nodes();
                    visit
                        .par_visit(
                            node,
                            |event| {
                                if let breadth_first::EventPred::Unknown {
                                    curr, distance, ..
                                } = event
                                {
                                    dists[curr].store(distance, Ordering::Relaxed);
                                }
                                Continue(())
                            },
                            &t,
                            no_logging![],
                        )
                        .continue_value_no_break();
                }

                let actual_dists = into_non_atomic(dists);

                assert_eq!(actual_dists, expected_dists);

                Ok(())
            }
        }
    };
}

test_bfv_algo_seq!(
    webgraph_algo::prelude::breadth_first::Seq::<_>::new,
    sequential
);
test_bfv_algo_par!(
    |g| { webgraph_algo::prelude::breadth_first::ParFairPred::<_>::new(g, 32,) },
    parallel_fair_pred
);
test_bfv_algo_par!(
    |g| { webgraph_algo::prelude::breadth_first::ParLowMem::<_>::new(g, 32,) },
    parallel_fast_callback
);
