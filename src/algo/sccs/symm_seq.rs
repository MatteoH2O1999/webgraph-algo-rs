use std::mem::MaybeUninit;

use super::traits::{StronglyConnectedComponents, StronglyConnectedComponentsNoT};
use crate::{algo, prelude::depth_first, traits::Sequential};
use unwrap_infallible::UnwrapInfallible;
use webgraph::{labels::Left, prelude::VecGraph};

/// Connected components by sequential visits on symmetric graphs.
pub struct SymmSeq<A: algo::visits::Event, V: Sequential<A>> {
    num_components: usize,
    component: Box<[usize]>,
    _marker: std::marker::PhantomData<(A, V)>,
}

impl<A: algo::visits::Event, V: Sequential<A>> StronglyConnectedComponents for SymmSeq<A, V> {
    fn num_components(&self) -> usize {
        self.num_components
    }

    fn component(&self) -> &[usize] {
        &self.component
    }

    fn component_mut(&mut self) -> &mut [usize] {
        &mut self.component
    }

    fn compute_with_t(
        graph: impl webgraph::prelude::RandomAccessGraph,
        _transpose: impl webgraph::prelude::RandomAccessGraph,
        pl: &mut impl dsi_progress_logger::ProgressLog,
    ) -> Self {
        // debug_assert!(check_symmetric(&graph)); requires sync
        let mut visit = depth_first::Seq::new(&graph);
        let mut component = vec![MaybeUninit::uninit(); graph.num_nodes()].into_boxed_slice();
        let mut number_of_components = 0usize.wrapping_sub(1);

        visit
            .visit_all(
                |event| {
                    match event {
                        depth_first::Event::Init { .. } => {
                            number_of_components = number_of_components.wrapping_add(1);
                        }
                        depth_first::Event::Previsit { curr, .. } => {
                            component[curr].write(number_of_components);
                        }
                        _ => (),
                    }
                    Ok(())
                },
                pl,
            )
            .unwrap_infallible();

        let component =
            unsafe { std::mem::transmute::<Box<[MaybeUninit<usize>]>, Box<[usize]>>(component) };

        SymmSeq {
            component,
            num_components: number_of_components + 1,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<A: algo::visits::Event, V: Sequential<A>> StronglyConnectedComponentsNoT for SymmSeq<A, V> {
    fn compute(
        graph: impl webgraph::prelude::RandomAccessGraph,
        pl: &mut impl dsi_progress_logger::ProgressLog,
    ) -> Self {
        Self::compute_with_t(graph, Left(VecGraph::<()>::new()), pl)
    }
}