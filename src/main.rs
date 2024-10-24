use anyhow::Result;
use dsi_progress_logger::{ProgressLog, ProgressLogger};
use epserde::deser::{Deserialize, Flags};
use webgraph::prelude::{BvGraph, DCF};
use webgraph::traits::SequentialLabeling;
use webgraph_algo::algo::diameter::*;
use webgraph_algo::algo::hyperball::*;
use webgraph_algo::algo::scc::*;
use webgraph_algo::prelude::*;
use webgraph_algo::utils::HyperLogLogCounterArrayBuilder;

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()?;
    let basename = std::env::args().nth(2).expect("No graph basename provided");
    let graph = BvGraph::with_basename(&basename).load()?;
    let reversed_graph = BvGraph::with_basename(basename.clone() + "-t").load()?;
    let cumulative = DCF::load_mmap(basename + ".dcf", Flags::empty())?;
    let main_pl = ProgressLogger::default();
    main_pl.info(format_args!("Starting test..."));

    let mut flags = MmapFlags::empty();
    flags.set(MmapFlags::SHARED, true);
    flags.set(MmapFlags::RANDOM_ACCESS, true);

    let mem_options = TempMmapOptions::Default;

    match std::env::args()
        .nth(1)
        .expect("No operation provided")
        .as_str()
    {
        "tarjan" => {
            TarjanStronglyConnectedComponents::compute(&graph, false, mem_options, main_pl)?;
        }
        "diameter" => {
            let mut diameter = SumSweepDirectedDiameterRadiusBuilder::new(
                &graph,
                &reversed_graph,
                SumSweepOutputLevel::RadiusDiameter,
            )
            .mem_settings(mem_options)
            .build(main_pl.clone())?;
            diameter.compute(main_pl.clone())?;
        }
        "hyperball" => {
            let log2m = std::env::args()
                .nth(3)
                .expect("No log2m value provided")
                .parse()
                .expect("Expected integer");

            let mut hyperball = HyperBallBuilder::new(&graph, cumulative.as_ref())
                .hyperloglog_settings(
                    HyperLogLogCounterArrayBuilder::new()
                        .log_2_num_registers(log2m)
                        .mem_options(mem_options.clone())
                        .num_elements_upper_bound(graph.num_nodes()),
                )
                .mem_settings(mem_options)
                .sum_of_distances(true)
                .sum_of_inverse_distances(true)
                .transposed(Some(&reversed_graph))
                .build(main_pl.clone())?;
            hyperball.run_until_done(main_pl)?;
        }
        _ => panic!(),
    }

    Ok(())
}
