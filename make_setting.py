import networkx as nx
import numpy as np
import random
import pandas as pd
import os
import json
import argparse
import shutil

parser = argparse.ArgumentParser(description="Mininet")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--K",
    type=int,
    default=None,
    help="Deprecated and ignored; Imaginary nodes are generated for every layer.",
)
parser.add_argument("--N", type=int, default=10)
parser.add_argument("--bw", type=int, default=1000)
parser.add_argument("--candidates", type=int, default=1)
parser.add_argument("--delay", type=float, default=10)
parser.add_argument("--datadir", type=str, default="nsfnet")
parser.add_argument("--resdir", "-o", type=str, default="data")
parser.add_argument("--is_rng", action="store_true")
args = parser.parse_args()

N = args.N
bw = args.bw
delay = args.delay
seed = args.seed
datadir = args.datadir
resdir = args.resdir
is_rng = args.is_rng
number_of_candidates = args.candidates
np.random.seed(seed)


def random_connected_gnp_retry(
    n: int, p: float, seed: int | None = None, max_tries: int = 1000
) -> nx.Graph:
    rng = random.Random(seed)
    for _ in range(max_tries):
        G = nx.fast_gnp_random_graph(n, p, seed=rng.randint(0, 10**9))
        if nx.is_connected(G):
            return G
    raise RuntimeError("Failed to sample a connected graph within max_tries")


def main():
    df = pd.read_excel(f"{datadir}/resnet101.xlsx")
    df.columns = (
        "conv_name",
        "layer_index",
        "layer",
        "MACs",
        "FWD_FLOPs",
        "BWD_FLOPs",
        "dimension",
        "activation_bytes",
        "gradient_bytes",
        "parameters",
        "memory",
    )
    data = df.to_dict("records")
    L = len(data)
    activation_bytes = [data[i]["activation_bytes"] for i in range(L)]
    gradient_bytes = [data[i]["gradient_bytes"] for i in range(L)]
    fwd_flops = [data[i]["FWD_FLOPs"] for i in range(L)]
    bwd_flops = [data[i]["BWD_FLOPs"] for i in range(L)]

    li_bw = [0.5, 1, 2.5]
    filename = f"{datadir}/network.gml"
    if is_rng:
        G = random_connected_gnp_retry(N, p=0.1, seed=seed)
        for i, j in G.edges():
            # G[i][j]["bandwidth"] = 1e9  # bps
            G[i][j]["bandwidth"] = float(np.random.choice(li_bw)) * 1e9  # bps
        G = G.to_directed()
        for i, j in G.edges():
            if not "bandwidth" in G[i][j]:
                G[i][j]["bandwidth"] = G[j][i]["bandwidth"]
            # G[i][j]["delay"] = args.delay  # ms
            G[i][j]["delay"] = np.random.randint(1, 10)  # ms
    else:
        G = nx.read_gml(filename, label="id")
        for i, j in G.edges():
            # G[i][j]["bandwidth"] = 1e9  # bps
            G[i][j]["bandwidth"] = float(np.random.choice(li_bw)) * 1e9  # bps
        G = G.to_directed()
        for i, j in G.edges():
            G.add_edge(j, i, delay=G[i][j]["delay"], bandwidth=G[i][j]["bandwidth"])
    for i in G.nodes():
        # G.nodes()[i]["gpu_type"] = 1
        G.nodes()[i]["gpu_type"] = int(np.random.choice([0, 1]))
    print(len(G.nodes()), len(G.edges()))
    for i in G.nodes():
        G.nodes()[i]["label"] = str(i)
        G.nodes()[i]["frequency"] = 1455 * 1e6
        G.nodes()[i]["FLOPS"] = 1.612 * 1e10  # * 1e10
        G.nodes()[i]["kappa_gpu_leq128"] = (
            1 / G.nodes()[i]["FLOPS"]
        )  # / G.nodes()[i]['frequency']
        G.nodes()[i]["tau_gpu_leq128"] = 10 * 1e-3
        G.nodes()[i]["kappa_gpu_geq128"] = (
            1 / G.nodes()[i]["FLOPS"]
        )  # / G.nodes()[i]['frequency']
        G.nodes()[i]["tau_gpu_geq128"] = 10 * 1e-3
        G.nodes()[i]["kappa_cpu"] = (
            1 / G.nodes()[i]["FLOPS"]
        )  # / G.nodes()[i]['frequency']
        G.nodes()[i]["tau_cpu"] = 10 * 1e-3
        # G.nodes()[i]['kappa_gpu_leq128'] = 0.0145 * 1e-3 / sum(fwd_flops)
        # G.nodes()[i]['tau_gpu_leq128']   = 7.2206 * 1e-3 / sum(fwd_flops)
        # G.nodes()[i]['kappa_gpu_geq128'] = 0.0543 * 1e-3 / sum(fwd_flops)
        # G.nodes()[i]['tau_gpu_geq128']   = 2.4404 * 1e-3 / sum(fwd_flops)
        # G.nodes()[i]['kappa_cpu']        = 0.4140 * 1e-3 / sum(fwd_flops)
        # G.nodes()[i]['tau_cpu']          = 17.3625 * 1e-3 / sum(fwd_flops)
        # G.nodes()[i]['tau']              = 10 * 1e-3
        G.nodes()[i]["memory_capacity"] = 400 * 1e6
        G.nodes()[i]["storage_capacity"] = 400 * 1e6

    node_list = np.array(list(G.nodes()))
    physical_node_size = len(G.nodes())
    # Encode every possible Imaginary node (one per model layer) and connect
    # each of them to every Physical node.  The resulting L * N records are
    # the complete set of Imaginary links consumed by the placement model.
    service_chain = list(range(L))
    # placements = [
    #         {"function": int(layer), "physical_node": int(physical_node)}
    #         for layer in service_chain
    #         for physical_node in node_list
    #         ]
    origin = int(node_list[0])
    destination = int(node_list[-1])
    placements = []
    for layer in service_chain:
        if int(layer) == 0:
            ret = {"function": int(layer), "physical_node": int(origin)}
            placements.append(ret)
            continue
        if int(layer) == L - 1:
            ret = {"function": int(layer), "physical_node": int(destination)}
            placements.append(ret)
            continue
        for physical_node in node_list:
            ret = {"function": int(layer), "physical_node": int(physical_node)}
            placements.append(ret)

    # Keep the endpoint convention deterministic and independent of K.
    function_placement = {}
    function_placement["functions"] = service_chain
    function_placement["origin"] = origin
    function_placement["destination"] = destination
    function_placement["placement"] = placements

    os.makedirs(resdir, exist_ok=True)
    filename = f"{resdir}/network.gml"
    nx.write_gml(G, filename)
    filename = f"{resdir}/function_placement_candidates.json"
    with open(filename, "w") as f:
        json.dump(
            function_placement,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    new_path = shutil.copy(f"{datadir}/resnet18.xlsx", f"{resdir}/resnet18.xlsx")
    new_path = shutil.copy(f"{datadir}/resnet34.xlsx", f"{resdir}/resnet34.xlsx")
    new_path = shutil.copy(f"{datadir}/resnet50.xlsx", f"{resdir}/resnet50.xlsx")
    new_path = shutil.copy(f"{datadir}/resnet101.xlsx", f"{resdir}/resnet101.xlsx")
    new_path = shutil.copy(f"{datadir}/resnet152.xlsx", f"{resdir}/resnet152.xlsx")
    new_path = shutil.copy(f"{datadir}/vit_b_16.xlsx", f"{resdir}/vit_b_16.xlsx")


if __name__ == "__main__":
    main()
