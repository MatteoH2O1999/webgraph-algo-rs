//! Module containing traits and implementations of both depth-first and breadth-first
//! visits on graphs.
//!
//! Breadth-first visits come in 3 flavours:
//! * [`Sequential`](bfv::SingleThreadedBreadthFirstVisit): a single threaded visit.
//! * [`Parallel`](bfv::ParallelBreadthFirstVisit): a parallel visit where at each iteration
//!   the frontier is divided in chunks for the threads in order to call the callback and perform
//!   the visit logic. In order to do so both the node and its parent must be enqued in the frontier.
//! * [`Parallel with fast callbacks`](bfv::ParallelBreadthFirstVisitFastCB): a parallel visit where the
//!   callback is called during successor enumeration, allowing to store only the nodes without their parents.
//!   This leads to slowdowns and less parallelization in the case where the callback is not trascurable relative
//!   to the visit logic but to performance improvements in case it is.

pub mod bfv;
pub mod dfv;

use dsi_progress_logger::ProgressLog;

/// A sequential visit.
///
/// Implementation of this trait must provide the
/// [`visit_from_node`](SeqVisit::visit_from_node) method, which should
/// perform a visit of a graph starting from a given node, and the
/// [`visit`](SeqVisit::visit) method, which should perform a
/// visit of the whole graph.
///
/// For each node, the visit should invoke a callback with argument of type
/// `A`. The callback may be called multiple times depending on the implementation
/// as long as information on the type of event is passed bundled with `A`.
///
/// In addition, before discovering a node the visit should invoke a filter
/// callback with argument of type `&A` that returns a boolean.
/// If the filter returns `false`, the node is not discovered but may be
/// discovered in the future with a different arc.
/// The filer can also be used to avoid calling the callback in other events
/// or to implement other behavior that should be specified by the single implementation.
///
pub trait SeqVisit<A> {
    /// Visits the graph from the specified node.
    ///
    /// # Arguments:
    /// * `root`: The node to start the visit from.
    /// * `callback`: The callback function.
    /// * `filter`: The filter function
    /// * `pl`: A progress logger that implements
    ///   [`dsi_progress_logger::ProgressLog`] may be passed to the method to
    ///   log the progress of the visit. If
    ///   `Option::<dsi_progress_logger::ProgressLogger>::None` is passed,
    ///   logging code should be optimized away by the compiler.
    fn visit_from_node<C: FnMut(A), F: Fn(&A) -> bool>(
        &mut self,
        root: usize,
        callback: C,
        filter: F,
        pl: &mut impl ProgressLog,
    );

    /// Visits the whole graph.
    ///
    /// See [`visit_from_node`](SeqVisit::visit_from_node) for more
    /// details.
    fn visit<C: FnMut(A), F: Fn(&A) -> bool>(
        &mut self,
        callback: C,
        filter: F,
        pl: &mut impl ProgressLog,
    );

    /// Resets the visit status, making it possible to reuse it.
    fn reset(&mut self);
}

/// A parallel visit.
///
/// Implementation of this trait must provide the
/// [`visit_from_node`](ParVisit::visit_from_node) method, which should
/// perform a visit of a graph starting from a given node, and the
/// [`visit`](ParVisit::visit) method, which should perform a
/// visit of the whole graph.
///
/// For each node, the visit should invoke a callback with argument of type
/// `A`. The callback may be called multiple times depending on the implementation
/// as long as information on the type of event is passed bundled with `A`.
///
/// In addition, before discovering a node the visit should invoke a filter
/// callback with argument of type `&A` that returns a boolean.
/// If the filter returns `false`, the node is not discovered but may be
/// discovered in the future with a different arc.
/// The filer can also be used to avoid calling the callback in other events
/// or to implement other behavior that should be specified by the single implementation.
///
pub trait ParVisit<A> {
    /// Visits the graph from the specified node.
    ///
    /// # Arguments:
    /// * `root`: The node to start the visit from.
    /// * `callback`: The callback function.
    /// * `filter`: The filter function
    /// * `pl`: A progress logger that implements
    ///   [`dsi_progress_logger::ProgressLog`] may be passed to the method to
    ///   log the progress of the visit. If
    ///   `Option::<dsi_progress_logger::ProgressLogger>::None` is passed,
    ///   logging code should be optimized away by the compiler.
    fn visit_from_node<C: Fn(A) + Sync, F: Fn(&A) -> bool + Sync>(
        &mut self,
        root: usize,
        callback: C,
        filter: F,
        pl: &mut impl ProgressLog,
    );

    /// Visits the whole graph.
    ///
    /// See [`visit_from_node`](ParVisit::visit_from_node) for more
    /// details.
    fn visit<C: Fn(A) + Sync, F: Fn(&A) -> bool + Sync>(
        &mut self,
        callback: C,
        filter: F,
        pl: &mut impl ProgressLog,
    );

    /// Resets the visit status, making it possible to reuse it.
    fn reset(&mut self);
}
