use crate::{
    algo::{
        bfv::BFV,
        diameter::{scc_graph::SccGraph, SumSweepOutputLevel},
        strongly_connected_components::TarjanStronglyConnectedComponents,
    },
    prelude::*,
    utils::{closure_vec, math, MmapSlice},
};
use anyhow::{Context, Result};
use dsi_progress_logger::*;
use nonmax::NonMaxUsize;
use rayon::prelude::*;
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    RwLock,
};
use sux::bits::AtomicBitVec;
use webgraph::traits::RandomAccessGraph;

const VISIT_GRANULARITY: usize = 32;

pub struct SumSweepDirectedDiameterRadius<
    'a,
    G: RandomAccessGraph + Sync,
    C: StronglyConnectedComponents<G>,
> {
    graph: &'a G,
    reversed_graph: &'a G,
    number_of_nodes: usize,
    output: SumSweepOutputLevel,
    radial_vertices: AtomicBitVec,
    diameter_lower_bound: usize,
    radius_upper_bound: usize,
    /// A vertex whose eccentricity equals the diameter.
    diameter_vertex: usize,
    /// A vertex whose eccentrivity equals the radius.
    radius_vertex: usize,
    /// Number of iterations performed until now.
    iterations: usize,
    lower_bound_forward_eccentricities: MmapSlice<usize>,
    upper_bound_forward_eccentricities: MmapSlice<usize>,
    lower_bound_backward_eccentricities: MmapSlice<usize>,
    upper_bound_backward_eccentricities: MmapSlice<usize>,
    /// Number of iterations before the radius is found.
    radius_iterations: Option<NonMaxUsize>,
    /// Number of iterations before the diameter is found.
    diameter_iterations: Option<NonMaxUsize>,
    /// Number of iterations before all forward eccentricities are found.
    forward_eccentricities_iterations: Option<NonMaxUsize>,
    /// Number of iterations before all eccentricities are found.
    all_eccentricities_iterations: Option<NonMaxUsize>,
    strongly_connected_components: C,
    /// The strongly connected components diagram.
    strongly_connected_components_graph: SccGraph<G, C>,
    /// Total forward distance from already processed vertices (used as tie-break for the choice
    /// of the next vertex to process).
    total_forward_distance: MmapSlice<usize>,
    /// Total backward distance from already processed vertices (used as tie-break for the choice
    /// of the next vertex to process).
    total_backward_distance: MmapSlice<usize>,
    compute_radial_vertices: bool,
}

impl<'a, G: RandomAccessGraph + Sync>
    SumSweepDirectedDiameterRadius<'a, G, TarjanStronglyConnectedComponents<G>>
{
    /// Creates a new instance for computing diameter and/or radius and/or all eccentricities.
    ///
    /// # Arguments
    /// - `graph`: An immutable reference to the graph.
    /// - `reversed_graph`: An immutable reference to `graph` transposed.
    /// - `output`: Which output is requested: radius, diameter, radius and diameter, or all eccentricities.
    /// - `radial_verticies`: The set of radial vertices. If [`None`], the set is automatically chosen
    /// as the set of vertices that are in the biggest strongly connected component, or that are able
    /// to reach the biggest strongly connected component.
    /// - `options`: the options for the [`crate::utils::MmapSlice`].
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    pub fn new(
        graph: &'a G,
        reversed_graph: &'a G,
        output: SumSweepOutputLevel,
        radial_vertices: Option<AtomicBitVec>,
        options: TempMmapOptions,
        pl: impl ProgressLog,
    ) -> Result<Self> {
        let nn = graph.num_nodes();

        let scc =
            TarjanStronglyConnectedComponents::compute(graph, false, options.clone(), pl.clone())
                .with_context(|| "Cannot compute strongly connected components")?;

        let compute_radial_vertices = radial_vertices.is_none();
        let acc_radial = if let Some(r) = radial_vertices {
            debug_assert_eq!(r.len(), nn);
            r
        } else {
            AtomicBitVec::new(nn)
        };

        let scc_graph = SccGraph::new(graph, reversed_graph, &scc, options.clone(), pl.clone())
            .with_context(|| "Cannot create strongly connected components graph")?;

        debug_assert_eq!(graph.num_nodes(), reversed_graph.num_nodes());
        debug_assert_eq!(graph.num_arcs(), reversed_graph.num_arcs());

        pl.info(format_args!("Initializing data structure"));

        let lower_forward = MmapSlice::from_vec(vec![0; nn], options.clone())
            .with_context(|| "Cannot create lower bound forward eccentricities slice")?;
        let lower_backward = MmapSlice::from_vec(vec![0; nn], options.clone())
            .with_context(|| "Cannot create lower bound backwards eccentricities slice")?;
        let upper_forward = MmapSlice::from_vec(vec![nn + 1; nn], options.clone())
            .with_context(|| "Cannot create upper bound forward eccentricities slice")?;
        let upper_backward = MmapSlice::from_vec(vec![nn + 1; nn], options.clone())
            .with_context(|| "Cannot create upper bound backwards eccentricities slice")?;
        let total_forward = MmapSlice::from_vec(vec![0; nn], options.clone())
            .with_context(|| "Cannot create total forward distances slice")?;
        let total_backward = MmapSlice::from_vec(vec![0; nn], options.clone())
            .with_context(|| "Cannot create total backards distances slice")?;

        Ok(SumSweepDirectedDiameterRadius {
            graph,
            reversed_graph,
            number_of_nodes: nn,
            total_forward_distance: total_forward,
            total_backward_distance: total_backward,
            lower_bound_forward_eccentricities: lower_forward,
            upper_bound_forward_eccentricities: upper_forward,
            lower_bound_backward_eccentricities: lower_backward,
            upper_bound_backward_eccentricities: upper_backward,
            strongly_connected_components_graph: scc_graph,
            strongly_connected_components: scc,
            diameter_lower_bound: 0,
            radius_upper_bound: usize::MAX,
            output,
            radius_iterations: None,
            diameter_iterations: None,
            all_eccentricities_iterations: None,
            forward_eccentricities_iterations: None,
            iterations: 0,
            radial_vertices: acc_radial,
            radius_vertex: 0,
            diameter_vertex: 0,
            compute_radial_vertices,
        })
    }
}

impl<'a, G: RandomAccessGraph + Sync, C: StronglyConnectedComponents<G> + Sync>
    SumSweepDirectedDiameterRadius<'a, G, C>
{
    fn delta_debug(&self, pl: impl ProgressLog, message: std::fmt::Arguments) {
        pl.info(format_args!("Delta debug: {}", message));
        pl.info(format_args!(
            "Delta debug: forwards upper bounds -> {:?}",
            self.upper_bound_forward_eccentricities.as_slice()
        ));
        pl.info(format_args!(
            "Delta debug: forwards lower bounds -> {:?}",
            self.lower_bound_forward_eccentricities.as_slice()
        ));
        pl.info(format_args!(
            "Delta debug: backward upper bounds -> {:?}",
            self.upper_bound_backward_eccentricities.as_slice()
        ));
        pl.info(format_args!(
            "Delta debug: backward lower bounds -> {:?}",
            self.lower_bound_backward_eccentricities.as_slice()
        ));
        pl.info(format_args!(
            "Delta debug: forwards total distance -> {:?}",
            self.total_forward_distance.as_slice()
        ));
        pl.info(format_args!(
            "Delta debug: backward total distance -> {:?}",
            self.total_backward_distance.as_slice()
        ));

        let to_complete_f = (0..self.number_of_nodes)
            .map(|v| self.incomplete_forward_vertex(v))
            .collect::<Vec<_>>();
        let to_complete_b = (0..self.number_of_nodes)
            .map(|v| self.incomplete_backward_vertex(v))
            .collect::<Vec<_>>();

        pl.info(format_args!(
            "Delta debug: incomplete forwards vertices -> {:?}",
            to_complete_f
        ));
        pl.info(format_args!(
            "Delta debug: incomplete backward vertices -> {:?}",
            to_complete_b
        ));
    }

    fn incomplete_forward_vertex(&self, index: usize) -> bool {
        self.lower_bound_forward_eccentricities[index]
            != self.upper_bound_forward_eccentricities[index]
    }

    fn incomplete_backward_vertex(&self, index: usize) -> bool {
        self.lower_bound_backward_eccentricities[index]
            != self.upper_bound_backward_eccentricities[index]
    }

    fn forward_eccentricity(&self, index: usize) -> Option<usize> {
        if self.incomplete_forward_vertex(index) {
            None
        } else {
            debug_assert_eq!(
                self.lower_bound_forward_eccentricities[index],
                self.upper_bound_forward_eccentricities[index]
            );
            Some(self.lower_bound_forward_eccentricities[index])
        }
    }

    fn backward_eccentricity(&self, index: usize) -> Option<usize> {
        if self.incomplete_backward_vertex(index) {
            None
        } else {
            debug_assert_eq!(
                self.lower_bound_backward_eccentricities[index],
                self.upper_bound_backward_eccentricities[index]
            );
            Some(self.lower_bound_backward_eccentricities[index])
        }
    }

    /// Performs `iterations` steps of the SumSweep heuristic, starting from vertex `start`.
    ///
    /// # Arguments
    /// - `start`: The starting vertex.
    /// - `iterations`: The number of iterations.
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    fn sum_sweep_heuristic(
        &mut self,
        start: usize,
        iterations: usize,
        pl: impl ProgressLog,
    ) -> Result<()> {
        pl.info(format_args!(
            "Performing initial SumSweep visit from {}.",
            start
        ));
        self.step_sum_sweep(Some(start), true, pl.clone())
            .with_context(|| "Could not perform initial SumSweep visit")?;

        for i in 2..=iterations {
            if i % 2 == 0 {
                let v = math::filtered_argmax(
                    &self.total_backward_distance,
                    &self.lower_bound_backward_eccentricities,
                    |i, _| self.incomplete_backward_vertex(i),
                );
                pl.info(format_args!(
                    "Performing backwards SumSweep visit from {:?}",
                    v
                ));
                self.step_sum_sweep(v, false, pl.clone())
                    .with_context(|| format!("Could not perform backwards visit from {:?}", v))?;
            } else {
                let v = math::filtered_argmax(
                    &self.total_forward_distance,
                    &self.lower_bound_forward_eccentricities,
                    |i, _| self.incomplete_forward_vertex(i),
                );
                pl.info(format_args!(
                    "Performing forward SumSweep visit from {:?}.",
                    v
                ));
                self.step_sum_sweep(v, true, pl.clone())
                    .with_context(|| format!("Could not perform forward visit from {:?}", v))?;
            }
        }

        Ok(())
    }

    /// Computes diameter, radius, and/or all eccentricities.
    ///
    /// Results can be accessed by methods [`Self::radius`], [`Self::diameter`], [`Self::radial_vertex`],
    /// [`Self::diametral_vertex`], [`Self::eccentricity`].
    ///
    /// # Arguments
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    pub fn compute(&mut self, mut pl: impl ProgressLog) -> Result<()> {
        if self.number_of_nodes == 0 {
            return Ok(());
        }
        pl.start("Computing SumSweep...");

        if self.compute_radial_vertices {
            self.compute_radial_vertices(pl.clone())
                .with_context(|| "Could not compute radial vertices")?;
        }

        let max_outdegree_vertex = AtomicUsize::new(0);

        (0..self.number_of_nodes).into_par_iter().for_each(|v| {
            let outdegree = self.graph.outdegree(v);
            let mut current_max = max_outdegree_vertex.load(Ordering::Relaxed);
            while outdegree > self.graph.outdegree(current_max) {
                let result = max_outdegree_vertex.compare_exchange(
                    current_max,
                    v,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                );
                if result.is_ok() {
                    break;
                }
                current_max = max_outdegree_vertex.load(Ordering::Relaxed);
            }
        });

        self.sum_sweep_heuristic(max_outdegree_vertex.load(Ordering::Relaxed), 6, pl.clone())
            .with_context(|| "Could not perform first 6 iterations of SumSweep heuristic.")?;

        let mut points = [self.graph.num_nodes() as f64; 5];
        let mut missing_nodes = self
            .find_missing_nodes(pl.clone())
            .with_context(|| "Could not compute missing nodes")?;
        let mut old_missing_nodes;

        pl.info(format_args!(
            "Missing nodes: {} out of {}",
            missing_nodes,
            self.number_of_nodes * 2
        ));

        while missing_nodes > 0 {
            let step_to_perform =
                math::argmax(&points).with_context(|| "Could not find step to perform")?;

            match step_to_perform {
                0 => {
                    pl.info(format_args!("Performing all_cc_upper_bound."));
                    let pivot = self
                        .find_best_pivot(pl.clone())
                        .with_context(|| "Could not find best pivot for allCCUpperBound")?;
                    self.all_cc_upper_bound(pivot, pl.clone())
                        .with_context(|| "Could not perform allCCUpperBound")?
                }
                1 => {
                    pl.info(format_args!(
                        "Performing a forward BFS, from a vertex maximizing the upper bound."
                    ));
                    let v = math::filtered_argmax(
                        &self.upper_bound_forward_eccentricities,
                        &self.total_forward_distance,
                        |i, _| self.incomplete_forward_vertex(i),
                    );
                    self.step_sum_sweep(v, true, pl.clone())
                        .with_context(|| format!("Could not perform forward visit from {:?}", v))?
                }
                2 => {
                    pl.info(format_args!(
                        "Performing a forward BFS, from a vertex minimizing the lower bound."
                    ));
                    let v = math::filtered_argmin(
                        &self.lower_bound_forward_eccentricities,
                        &self.total_forward_distance,
                        |i, _| self.radial_vertices[i],
                    );
                    self.step_sum_sweep(v, true, pl.clone())
                        .with_context(|| format!("Could not perform forward visit from {:?}", v))?
                }
                3 => {
                    pl.info(format_args!(
                        "Performing a backward BFS from a vertex maximizing the upper bound."
                    ));
                    let v = math::filtered_argmax(
                        &self.upper_bound_backward_eccentricities,
                        &self.total_backward_distance,
                        |i, _| self.incomplete_backward_vertex(i),
                    );
                    self.step_sum_sweep(v, false, pl.clone()).with_context(|| {
                        format!("Could not perform backwards visit from {:?}", v)
                    })?
                }
                4 => {
                    pl.info(format_args!(
                        "Performing a backward BFS, from a vertex maximizing the distance sum."
                    ));
                    let v = math::filtered_argmax(
                        &self.total_backward_distance,
                        &self.upper_bound_backward_eccentricities,
                        |i, _| self.incomplete_backward_vertex(i),
                    );
                    self.step_sum_sweep(v, false, pl.clone()).with_context(|| {
                        format!("Could not perform backwards visit from {:?}", v)
                    })?
                }
                5.. => panic!(),
            }

            old_missing_nodes = missing_nodes;
            missing_nodes = self
                .find_missing_nodes(pl.clone())
                .with_context(|| "Could not compute missing nodes")?;
            points[step_to_perform] = (old_missing_nodes - missing_nodes) as f64;

            // This is to make rust-analyzer happy as it cannot recognize mut reference
            #[allow(clippy::needless_range_loop)]
            for i in 0..points.len() {
                if i != step_to_perform && points[i] >= 0.0 {
                    points[i] += 2.0 / self.iterations as f64;
                }
            }

            pl.info(format_args!(
                "Missing nodes: {} out of {}",
                missing_nodes,
                self.number_of_nodes * 2
            ));
        }

        if self.output == SumSweepOutputLevel::Radius
            || self.output == SumSweepOutputLevel::RadiusDiameter
        {
            pl.info(format_args!(
                "Radius: {} ({} iterations).",
                self.radius_upper_bound,
                self.radius_iterations
                    .expect("radius iterations should not be None")
            ));
        }
        if self.output == SumSweepOutputLevel::Diameter
            || self.output == SumSweepOutputLevel::RadiusDiameter
        {
            pl.info(format_args!(
                "Diameter: {} ({} iterations).",
                self.diameter_lower_bound,
                self.diameter_iterations
                    .expect("radius iterations should not be None"),
            ));
        }
        pl.done();

        Ok(())
    }

    /// Returns the radius of the graph if it has already been computed, [`None`] otherwise.
    pub fn radius(&self) -> Option<usize> {
        if self.radius_iterations.is_none() {
            None
        } else {
            Some(self.radius_upper_bound)
        }
    }

    /// Returns the diameter of the graph if is has already been computed, [`None`] otherwise.
    pub fn diameter(&self) -> Option<usize> {
        if self.diameter_iterations.is_none() {
            None
        } else {
            Some(self.diameter_lower_bound)
        }
    }

    /// Returns a radial vertex if it has already been computed, [`None`] otherwise.
    pub fn radial_vertex(&self) -> Option<usize> {
        if self.radius_iterations.is_none() {
            None
        } else {
            Some(self.radius_vertex)
        }
    }

    /// Returns a diametral vertex if it has already been computed, [`None`] otherwise.
    pub fn diametral_vertex(&self) -> Option<usize> {
        if self.diameter_iterations.is_none() {
            None
        } else {
            Some(self.diameter_vertex)
        }
    }

    /// Returns the eccentricity of a vertex if it has already been computed, [`None`] otherwise.
    ///
    /// # Arguments
    /// - `vertex`: The vertex.
    /// - `forward`: Whether to return the forward eccentricity (if `true`) or the backward
    /// eccentricity (if `false`).
    pub fn eccentricity(&self, vertex: usize, forward: bool) -> Option<usize> {
        if forward {
            self.forward_eccentricity(vertex)
        } else {
            self.backward_eccentricity(vertex)
        }
    }

    /// Returns the number of iterations needed to compute the radius if it has already
    /// been computed, [`None`] otherwise.
    pub fn radius_iterations(&self) -> Option<usize> {
        self.radius_iterations.map(|v| v.into())
    }

    /// Returns the number of iterations needed to compute the diameter if it has already
    /// been computed, [`None`] otherwise.
    pub fn diameter_iterations(&self) -> Option<usize> {
        self.diameter_iterations.map(|v| v.into())
    }

    /// Returns the number of iterations needed to compute all forward eccentricities
    /// if they have already been computed, [`None`] otherwise.
    pub fn all_forward_iterations(&self) -> Option<usize> {
        self.forward_eccentricities_iterations.map(|v| v.into())
    }

    /// Returns the number of iterations needed to compute all eccentricities if they
    /// have already been computed, [`None`] otherwise.
    pub fn all_iterations(&self) -> Option<usize> {
        self.all_eccentricities_iterations.map(|v| v.into())
    }

    /// Uses a heuristic to decide which is the best pivot to choose in each strongly connected
    /// component, in order to perform the [`Self::all_cc_upper_bound`] method.
    ///
    /// # Arguments
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    fn find_best_pivot(&self, mut pl: impl ProgressLog) -> Result<Vec<usize>> {
        let mut pivot = vec![None; self.strongly_connected_components.number_of_components()];
        let components = self.strongly_connected_components.component();
        pl.expected_updates(Some(components.len()));
        pl.item_name("nodes");
        pl.display_memory(false);
        pl.start("Computing best pivot");

        for (v, &component) in components.iter().enumerate().rev() {
            if let Some(p) = pivot[component] {
                let current = self.lower_bound_backward_eccentricities[v]
                    + self.lower_bound_forward_eccentricities[v]
                    + if self.incomplete_forward_vertex(v) {
                        0
                    } else {
                        self.number_of_nodes
                    }
                    + if self.incomplete_backward_vertex(v) {
                        0
                    } else {
                        self.number_of_nodes
                    };

                let best = self.lower_bound_backward_eccentricities[p]
                    + self.lower_bound_forward_eccentricities[p]
                    + if self.incomplete_forward_vertex(p) {
                        0
                    } else {
                        self.number_of_nodes
                    }
                    + if self.incomplete_backward_vertex(p) {
                        0
                    } else {
                        self.number_of_nodes
                    };

                if current < best
                    || (current == best
                        && self.total_forward_distance[v] + self.total_backward_distance[v]
                            <= self.total_forward_distance[p] + self.total_backward_distance[p])
                {
                    pivot[component] = Some(v);
                }
            } else {
                pivot[component] = Some(v);
            }
            pl.light_update();
        }

        pl.done();

        Ok(pivot.into_iter().map(|x| x.unwrap()).collect())
    }

    /// Computes and stores in variable [`Self::radial_vertices`] the set of vertices that are
    /// either in the biggest strongly connected component or that are able to reach vertices in
    /// the biggest strongly connected component.
    ///
    /// # Arguments
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    fn compute_radial_vertices(&mut self, mut pl: impl ProgressLog) -> Result<()> {
        if self.number_of_nodes == 0 {
            return Ok(());
        }

        pl.expected_updates(None);
        pl.item_name("nodes");
        pl.display_memory(false);
        pl.start("Computing radial vertices...");

        let component = self.strongly_connected_components.component();
        let scc_sizes = self.strongly_connected_components.compute_sizes();
        let max_size_scc =
            math::argmax(&scc_sizes).with_context(|| "Could not find max size scc.")?;

        pl.info(format_args!(
            "Searching for biggest strongly connected component"
        ));

        let mut v = self.number_of_nodes;

        while v > 0 {
            v -= 1;
            if component[v] == max_size_scc {
                break;
            }
        }

        pl.info(format_args!("Computing radial vertices set"));

        let mut bfs = BFV::new_parallel(self.reversed_graph)
            .with_granularity(VISIT_GRANULARITY)
            .build();
        bfs.visit_from_node(
            |node, _, _, _| self.radial_vertices.set(node, true, Ordering::Relaxed),
            v,
            &mut pl,
        )
        .with_context(|| format!("Could not perform BFS from {}", v))?;

        pl.done();

        Ok(())
    }

    /// Performs a (forward or backward) BFS, updating lower bounds on the eccentricities
    /// of all visited vertices.
    ///
    /// # Arguments
    /// - `start`: The starting vertex of the BFS. If [`None`], no visit happens.
    /// - `forward`: Whether the BFS is performed following the direction of edges or
    /// in the opposite direction.
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    fn step_sum_sweep(
        &mut self,
        start: Option<usize>,
        forward: bool,
        mut pl: impl ProgressLog,
    ) -> Result<()> {
        if start.is_none() {
            return Ok(());
        }
        let start = start.unwrap();

        self.delta_debug(pl.clone(), format_args!("BFS from {}", start));

        pl.item_name("nodes");
        pl.display_memory(false);
        pl.expected_updates(None);

        if forward {
            pl.start(format!("Performing forward BFS starting from {}", start));
        } else {
            pl.start(format!("Performing backwards BFS starting from {}", start));
        }

        let lower_bound;
        let other_lower_bound;
        let upper_bound;
        let other_upper_bound;
        let other_total_distance;
        let graph;

        if forward {
            other_lower_bound = &self.lower_bound_backward_eccentricities;
            other_upper_bound = &self.upper_bound_backward_eccentricities;
            other_total_distance = &self.total_backward_distance;
            graph = self.graph;
        } else {
            other_lower_bound = &self.lower_bound_forward_eccentricities;
            other_upper_bound = &self.upper_bound_forward_eccentricities;
            other_total_distance = &self.total_forward_distance;
            graph = self.reversed_graph;
        }

        let max_dist = AtomicUsize::new(0);
        let radius = RwLock::new((self.radius_upper_bound, self.radius_vertex));

        let mut bfs = BFV::new_parallel(graph)
            .with_granularity(VISIT_GRANULARITY)
            .build();

        bfs.visit_from_node(
            |node, _, _, distance| {
                // Safety for unsafe blocks: each node gets accessed exactly once, so no data races can happen
                let incomplete = if forward {
                    self.incomplete_backward_vertex(node)
                } else {
                    self.incomplete_forward_vertex(node)
                };
                max_dist.fetch_max(distance, Ordering::Relaxed);

                unsafe {
                    *other_total_distance.get_mut_unsafe(node) += distance;
                }
                if incomplete && other_lower_bound[node] < distance {
                    unsafe {
                        other_lower_bound.write_once(node, distance);
                    }

                    if distance == other_upper_bound[node] && !forward && self.radial_vertices[node]
                    {
                        let mut update_radius = false;
                        {
                            let radius_lock = radius.read().unwrap();
                            if distance < radius_lock.0 {
                                update_radius = true;
                            }
                        }

                        if update_radius {
                            let mut radius_lock = radius.write().unwrap();
                            if distance < radius_lock.0 {
                                radius_lock.0 = distance;
                                radius_lock.1 = node;
                            }
                        }
                    }
                }
            },
            start,
            &mut pl,
        )
        .with_context(|| format!("Could not perform BFS from {}", start))?;

        if forward {
            lower_bound = &mut self.lower_bound_forward_eccentricities;
            upper_bound = &mut self.upper_bound_forward_eccentricities;
        } else {
            lower_bound = &mut self.lower_bound_backward_eccentricities;
            upper_bound = &mut self.upper_bound_backward_eccentricities;
        }

        let ecc_start = max_dist.load(Ordering::Relaxed);

        lower_bound[start] = ecc_start;
        upper_bound[start] = ecc_start;

        (self.radius_upper_bound, self.radius_vertex) = radius.into_inner().unwrap();

        if self.diameter_lower_bound < ecc_start {
            self.diameter_lower_bound = ecc_start;
            self.diameter_vertex = start;
        }
        if forward && self.radial_vertices[start] && self.radius_upper_bound > ecc_start {
            self.radius_upper_bound = ecc_start;
            self.radius_vertex = start;
        }

        self.iterations += 1;

        pl.done();

        self.delta_debug(pl.clone(), format_args!("BFS from {} completed", start));

        Ok(())
    }

    /// Performs a (forward or backward) BFS inside each strongly connected component, starting
    /// from the pivot.
    ///
    /// # Arguments
    /// - `pivot`: An array containing in position `i` the pivot of the `i`-th strongly connected
    /// component.
    /// - `forward`: Whether the BFS is performed following the direction of edges or
    /// in the opposite direction.
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    ///
    /// # Return
    /// Two arrays.
    /// The first one contains the distance of each vertex from the pivot of its strongly connected
    /// component, while the second one contains in position `i` the eccentricity of the pivot of the
    /// `i`-th strongly connected component.
    fn compute_dist_pivot(
        &self,
        pivot: &[usize],
        forward: bool,
        mut pl: impl ProgressLog,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        pl.expected_updates(None);
        pl.display_memory(false);

        if forward {
            pl.start("Computing forward dist pivots");
        } else {
            pl.start("Computing backwards dist pivots");
        }

        let components = self.strongly_connected_components.component();
        let ecc_pivot = closure_vec(
            || AtomicUsize::new(0),
            self.strongly_connected_components.number_of_components(),
        );
        let dist_pivot: Vec<usize> = vec![0; self.number_of_nodes];
        let graph = if forward {
            self.graph
        } else {
            self.reversed_graph
        };
        let current_index = AtomicUsize::new(0);

        rayon::broadcast(|_| {
            let mut bfs = BFV::new_parallel(graph)
                .with_granularity(VISIT_GRANULARITY)
                .build();
            let mut current_pivot_index = current_index.fetch_add(1, Ordering::Relaxed);

            while let Some(&p) = pivot.get(current_pivot_index) {
                let pivot_component = components[p];
                let component_ecc_pivot = &ecc_pivot[pivot_component];

                bfs.visit_from_node_filtered(
                    |node, _, _, distance| {
                        // Safety: each node is accessed exaclty once
                        unsafe {
                            dist_pivot.write_once(node, distance);
                        }
                        component_ecc_pivot.store(distance, Ordering::Relaxed);
                    },
                    |node, _, _, _| components[node] == pivot_component,
                    p,
                    &mut Option::<ProgressLogger>::None,
                )
                .unwrap_or_else(|_| panic!("Could not perform visit from {}", p));

                current_pivot_index = current_index.fetch_add(1, Ordering::Relaxed);
            }
        });

        let usize_ecc_pivot = unsafe {
            let mut clone = std::mem::ManuallyDrop::new(ecc_pivot);
            Vec::from_raw_parts(
                clone.as_mut_ptr() as *mut usize,
                clone.len(),
                clone.capacity(),
            )
        };

        pl.done();

        Ok((dist_pivot, usize_ecc_pivot))
    }

    /// Performs a step of the ExactSumSweep algorithm.
    ///
    /// # Arguments
    /// - `pivot`: An array containing in position `i` the pivot of the `i`-th strongly connected component.
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    fn all_cc_upper_bound(&mut self, pivot: Vec<usize>, mut pl: impl ProgressLog) -> Result<()> {
        self.delta_debug(
            pl.clone(),
            format_args!("AllCCUpperBound with pivots {:?}", pivot),
        );

        pl.item_name("elements");
        pl.display_memory(false);
        pl.expected_updates(Some(
            pivot.len()
                + self.strongly_connected_components.number_of_components()
                + self.number_of_nodes,
        ));
        pl.start("Performing AllCCUpperBound step of ExactSumSweep algorithm");

        let dist_ecc_f = self
            .compute_dist_pivot(&pivot, true, pl.clone())
            .with_context(|| "Could not compute forward dist pivot")?;
        let dist_ecc_b = self
            .compute_dist_pivot(&pivot, false, pl.clone())
            .with_context(|| "Could not compute backward dist pivot")?;
        let dist_pivot_f = dist_ecc_f.0;
        let mut ecc_pivot_f = dist_ecc_f.1;
        let dist_pivot_b = dist_ecc_b.0;
        let mut ecc_pivot_b = dist_ecc_b.1;
        let components = self.strongly_connected_components.component();

        for (c, &p) in pivot.iter().enumerate() {
            for connection in self.strongly_connected_components_graph.children(c) {
                let next_c = connection.target;
                let start = connection.start;
                let end = connection.end;

                ecc_pivot_f[c] = std::cmp::max(
                    ecc_pivot_f[c],
                    dist_pivot_f[start] + 1 + dist_pivot_b[end] + ecc_pivot_f[next_c],
                );

                if ecc_pivot_f[c] >= self.upper_bound_forward_eccentricities[p] {
                    ecc_pivot_f[c] = self.upper_bound_forward_eccentricities[p];
                    break;
                }
            }
            pl.light_update();
        }

        for c in (0..self.strongly_connected_components.number_of_components()).rev() {
            for component in self.strongly_connected_components_graph.children(c) {
                let next_c = component.target;
                let start = component.start;
                let end = component.end;

                ecc_pivot_b[next_c] = std::cmp::max(
                    ecc_pivot_b[next_c],
                    dist_pivot_f[start] + 1 + dist_pivot_b[end] + ecc_pivot_b[c],
                );

                if ecc_pivot_b[next_c] >= self.upper_bound_backward_eccentricities[pivot[next_c]] {
                    ecc_pivot_b[next_c] = self.upper_bound_backward_eccentricities[pivot[next_c]];
                }
            }
            pl.light_update();
        }

        let radius = RwLock::new((self.radius_upper_bound, self.radius_vertex));

        (0..self.number_of_nodes).into_par_iter().for_each(|node| {
            // Safety for unsafe blocks: each node gets accessed exactly once, so no data races can happen
            unsafe {
                self.upper_bound_forward_eccentricities.write_once(
                    node,
                    std::cmp::min(
                        self.upper_bound_forward_eccentricities[node],
                        dist_pivot_b[node] + ecc_pivot_f[components[node]],
                    ),
                );
            }

            if !self.incomplete_forward_vertex(node) {
                let new_ecc = self.upper_bound_forward_eccentricities[node];

                if self.radial_vertices[node] {
                    let mut update_radius = false;
                    {
                        let radius_lock = radius.read().unwrap();
                        if new_ecc < radius_lock.0 {
                            update_radius = true;
                        }
                    }

                    if update_radius {
                        let mut radius_lock = radius.write().unwrap();
                        if new_ecc < radius_lock.0 {
                            radius_lock.0 = new_ecc;
                            radius_lock.1 = node;
                        }
                    }
                }
            }

            unsafe {
                self.upper_bound_backward_eccentricities.write_once(
                    node,
                    std::cmp::min(
                        self.upper_bound_backward_eccentricities[node],
                        dist_pivot_f[node] + ecc_pivot_b[components[node]],
                    ),
                );
            }
        });

        pl.update_with_count(self.number_of_nodes);

        (self.radius_upper_bound, self.radius_vertex) = radius.into_inner().unwrap();

        self.iterations += 3;

        pl.done();

        self.delta_debug(
            pl.clone(),
            format_args!("Completed AllCCUpperBound with pivots {:?}", pivot),
        );

        Ok(())
    }

    /// Computes how many nodes are still to be processed, before outputting the result.
    ///
    /// # Arguments
    /// - `pl`: A progress logger that implements [`dsi_progress_logger::ProgressLog`] may be passed to the
    /// method to log the progress. If `Option::<dsi_progress_logger::ProgressLogger>::None` is
    /// passed, logging code should be optimized away by the compiler.
    fn find_missing_nodes(&mut self, mut pl: impl ProgressLog) -> Result<usize> {
        pl.item_name("nodes");
        pl.display_memory(false);
        pl.expected_updates(Some(self.number_of_nodes));
        pl.start("Computing missing nodes");

        let missing_r = AtomicUsize::new(0);
        let missing_df = AtomicUsize::new(0);
        let missing_db = AtomicUsize::new(0);
        let missing_all_forward = AtomicUsize::new(0);
        let missing_all_backward = AtomicUsize::new(0);

        (0..self.number_of_nodes).into_par_iter().for_each(|node| {
            if self.incomplete_forward_vertex(node) {
                missing_all_forward.fetch_add(1, Ordering::Relaxed);
                if self.upper_bound_forward_eccentricities[node] > self.diameter_lower_bound {
                    missing_df.fetch_add(1, Ordering::Relaxed);
                }
                if self.radial_vertices[node]
                    && self.lower_bound_forward_eccentricities[node] < self.radius_upper_bound
                {
                    missing_r.fetch_add(1, Ordering::Relaxed);
                }
            }
            if self.incomplete_backward_vertex(node) {
                missing_all_backward.fetch_add(1, Ordering::Relaxed);
                if self.upper_bound_backward_eccentricities[node] > self.diameter_lower_bound {
                    missing_db.fetch_add(1, Ordering::Relaxed);
                }
            }
        });

        pl.update_with_count(self.number_of_nodes);

        let iterations =
            NonMaxUsize::new(self.iterations).expect("Iterations should never be usize::MAX");
        let missing_r = missing_r.load(Ordering::Relaxed);
        let missing_df = missing_df.load(Ordering::Relaxed);
        let missing_db = missing_db.load(Ordering::Relaxed);
        let missing_all_forward = missing_all_forward.load(Ordering::Relaxed);
        let missing_all_backward = missing_all_backward.load(Ordering::Relaxed);

        if missing_r == 0 && self.radius_iterations.is_none() {
            self.radius_iterations = Some(iterations);
        }
        if (missing_df == 0 || missing_db == 0) && self.diameter_iterations.is_none() {
            self.diameter_iterations = Some(iterations);
        }
        if missing_all_forward == 0 && self.forward_eccentricities_iterations.is_none() {
            self.forward_eccentricities_iterations = Some(iterations);
        }
        if missing_all_forward == 0 && missing_all_backward == 0 {
            self.all_eccentricities_iterations = Some(iterations);
        }

        pl.done();

        Ok(match self.output {
            SumSweepOutputLevel::Radius => missing_r,
            SumSweepOutputLevel::Diameter => std::cmp::min(missing_df, missing_db),
            SumSweepOutputLevel::RadiusDiameter => {
                missing_r + std::cmp::min(missing_df, missing_db)
            }
            SumSweepOutputLevel::AllForward => missing_all_forward,
            SumSweepOutputLevel::All => missing_all_backward + missing_all_forward,
        })
    }
}
