import subprocess
import os
import tempfile

graphs = [
    "wordassociation-2011",
    "cnr-2000",
    "in-2004",
    "eu-2005",
    "uk-2007-05@100000",
    "uk-2007-05@1000000",
    "indochina-2004",
    "eu-2015-tpd",
    "uk-2014-tpd",
    "uk-2014-host",
]

base_datasets = "data.law.di.unimi.it/webdata/"

rust_path = "./target/release/webgraph-algo"
if os.name == "nt":
    rust_path = rust_path + ".exe"
classpath_sep = ";" if os.name == "nt" else ":"
java_base_command = f'java -cp "./java/webgraph.jar{classpath_sep}./java/lib/*"'

build_executable_command = "cargo build --release --bin webgraph-algo"
install_webgraph_command = "cargo install --git https://github.com/vigna/webgraph-rs.git webgraph"

encoding = "cp437" if os.name == "nt" else "utf-8"

def log_and_run(command: str):
    print(command)
    subprocess.run(command, shell=True)

def prepare_graph(graph_name: str, path: str) -> str:
    basename = f"{path}{os.sep}{graph_name}"
    print(f"Downloading base graph {graph_name}...")
    log_and_run(f'curl -o "{basename}.graph" {base_datasets}{graph_name}/{graph_name}.graph')
    log_and_run(f'curl -o "{basename}.properties" {base_datasets}{graph_name}/{graph_name}.properties')
    print(f"Downloading transposed graph {graph_name}...")
    log_and_run(f'curl -o "{basename}-t.graph" {base_datasets}{graph_name}/{graph_name}-t.graph')
    log_and_run(f'curl -o "{basename}-t.properties" {base_datasets}{graph_name}/{graph_name}-t.properties')
    print(f"Padding downloaded graphs...")
    log_and_run(f"webgraph run pad {basename} u32")
    log_and_run(f"webgraph run pad {basename}-t u32")
    print(f"Computing offsets...")
    log_and_run(f"webgraph build offsets {basename}")
    log_and_run(f"webgraph build offsets {basename}-t")
    print(f"Computing elias fano...")
    log_and_run(f"webgraph build ef {basename}")
    log_and_run(f"webgraph build ef {basename}-t")
    print(f"Computing cumulative outdegree...")
    log_and_run(f"webgraph build dcf {basename}")

    return basename

log_and_run(build_executable_command)
log_and_run(install_webgraph_command)

for graph in graphs:
    with tempfile.TemporaryDirectory() as tmpdir:
        graph_basename = prepare_graph(graph, tmpdir)

        # Diameter
        java_command = f"{java_base_command} it.unimi.dsi.webgraph.algo.SumSweepDirectedDiameterRadius {graph_basename} -m -l RADIUS_DIAMETER"
        rust_command = f"{rust_path} diameter {graph_basename}"

        rust_output = subprocess.run(rust_command, shell=os.name != "nt", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding=encoding)
        java_output = subprocess.run(java_command, shell=os.name != "nt", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding=encoding)

        rust_lines = []
        java_lines = []
        
        for line in rust_output.stdout.split("\n"):
            if "Delta debug: " in line:
                rust_lines.append(line.split("Delta debug: ")[1])
        for line in java_output.stdout.split("\n"):
            if "Delta debug: " in line:
                java_lines.append(line.split("Delta debug: ")[1])

        assert len(java_lines) > 0
        assert len(rust_lines) > 0
        assert len(java_lines) == len(rust_lines), f"{len(java_lines)} != {len(rust_lines)}"
        
        for java_line, rust_line in zip(java_lines, rust_lines):
            assert java_line == rust_line, f"{java_line} != {rust_line}"
