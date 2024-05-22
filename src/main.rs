use anyhow::Result;
use dsi_progress_logger::{ProgressLog, ProgressLogger};
use std::path::Path;
use webgraph::graphs::BVGraph;
use webgraph_algo::algo::diameter::*;
use webgraph_algo::prelude::*;

fn main() -> Result<()> {
    stderrlog::new()
        .verbosity(2)
        .timestamp(stderrlog::Timestamp::Second)
        .init()?;
    let basename = std::env::args().nth(1).expect("No graph basename provided");
    let graph = BVGraph::with_basename(&basename).load()?;
    let reversed_graph = BVGraph::with_basename(basename + "-t").load()?;
    let main_pl = ProgressLogger::default();
    main_pl.info(format_args!("Starting test..."));

    let mut flags = MmapFlags::empty();
    flags.set(MmapFlags::SHARED, true);
    flags.set(MmapFlags::RANDOM_ACCESS, true);

    let mut sum_sweep_pl = ProgressLogger::default();
    sum_sweep_pl.display_memory(true).local_speed(true);
    let mut sum_sweep = SumSweepDirectedDiameterRadius::new(
        &graph,
        &reversed_graph,
        SumSweepOutputLevel::RadiusDiameter,
        None,
        TempMmapOptions::CustomDir(Path::new("./graphs").to_owned(), flags),
        sum_sweep_pl.clone(),
    )?;
    sum_sweep.compute(sum_sweep_pl)?;
    main_pl.info(format_args!(
        "Diameter: {:?}\tRadius: {:?}",
        sum_sweep.diameter(),
        sum_sweep.radius()
    ));

    Ok(())
}
