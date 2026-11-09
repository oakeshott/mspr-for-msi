import argparse
import json
import os
import time

import networkx as nx
import pandas as pd
import pulp


parser = argparse.ArgumentParser(description="Model splitting with a layered graph")
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--datadir", type=str, default="data")
parser.add_argument("--resdir", "-o", type=str, default="results")
parser.add_argument("--model", type=str, default="resnet101")
parser.add_argument("--is_test", action="store_true")
parser.add_argument("--is_cpu", action="store_true")
parser.add_argument("--capacity", type=float, default=None)
args = parser.parse_args()

bs = args.bs
seed = args.seed
datadir = args.datadir
resdir = args.resdir
is_test = args.is_test
is_cpu = args.is_cpu
capacity = 1e10 if args.capacity is None else args.capacity


def load_model_data(filename):
    df = pd.read_excel(filename)
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
    return df.to_dict("records")


def main():
    physical_network = nx.read_gml(f"{datadir}/network.gml", label="id")
    if not physical_network.is_directed():
        physical_network = physical_network.to_directed()

    with open(f"{datadir}/function_placement_candidates.json") as f:
        function_placement = json.load(f)
    origin = function_placement["origin"]
    destination = function_placement["destination"]

    data = load_model_data(f"{datadir}/{args.model}.xlsx")
    L = len(data)
    if L == 0:
        raise ValueError("The model contains no layers.")

    activation_bytes = [row["activation_bytes"] for row in data]
    gradient_bytes = [row["gradient_bytes"] for row in data]
    fwd_flops = [row["FWD_FLOPs"] for row in data]
    bwd_flops = [row["BWD_FLOPs"] for row in data]
    layer_memory_sizes = [row["memory"] for row in data]
    layer_storage_sizes = [row["memory"] for row in data]

    physical_nodes = list(physical_network.nodes())
    physical_edges = list(physical_network.edges())
    if origin not in physical_network or destination not in physical_network:
        raise ValueError("origin and destination must be Physical nodes.")

    bandwidth = nx.get_edge_attributes(physical_network, "bandwidth")
    edge_delay = nx.get_edge_attributes(physical_network, "delay")
    rate = {(i, j): bandwidth[(i, j)] / 8 for i, j in physical_edges}
    delay = {(i, j): edge_delay[(i, j)] * 1e-3 for i, j in physical_edges}

    total_fwd = sum(fwd_flops)
    if total_fwd <= 0:
        raise ValueError("The total forward FLOPs must be positive.")

    tau_cpu = 0
    tau = {}
    kappa = {}
    for node in physical_network.nodes():
        if node == origin or node == destination:
            if bs <= 8:
                kappa_cpu = (15.9363e-3 * bs + 21.1870e-3) / total_fwd
            else:
                kappa_cpu = (36.1756e-3 * bs + -218.7523e-3) / total_fwd
            kappa[node] = kappa_cpu
        else:
            if physical_network.nodes()[node]["gpu_type"] == 0:
                kappa_gpu = (0.9346e-3 * bs + 4.0646e-3) / total_fwd
            else:
                if bs <= 8:
                    kappa_gpu = (2.0377e-3 * bs + 1.7261e-3) / total_fwd
                else:
                    kappa_gpu = (2.9693e-3 * bs + -5.9813e-3) / total_fwd
            kappa[node] = kappa_gpu
    tau = {node: tau_cpu for node in physical_nodes}

    # print("The number of layers", L)
    # print("The number of nodes of physical network", len(physical_network.nodes()))
    # print("The number of links of physical network", len(physical_network.edges()))

    memory_capacity = {}
    storage_capacity = {}
    for node in physical_nodes:
        # if node == origin or node == destination:
        #     memory_capacity[node] = 8e9
        #     storage_capacity[node] = 8e14
        # else:
        #     memory_capacity[node] = capacity
        #     storage_capacity[node] = capacity
        memory_capacity[node] = capacity
        storage_capacity[node] = capacity

    # Layered graph construction:
    #   (physical_node, stage)
    #
    # A physical link stays within one stage.  An Imaginary link for model
    # layer k is the zero-cost edge (node,k) -> (node,k+1).  Selecting this
    # edge means that layer k is placed on that Physical node.
    graph_creation_start = time.perf_counter()
    layered_graph = nx.DiGraph()
    for stage in range(L + 1):
        for node in physical_nodes:
            layered_graph.add_node((node, stage), node_type="physical", stage=stage)

    for stage in range(L + 1):
        for i, j in physical_edges:
            layered_graph.add_edge(
                (i, stage),
                (j, stage),
                edge_type="physical",
                stage=stage,
                physical_edge=(i, j),
            )

    for layer in range(L):
        for node in physical_nodes:
            layered_graph.add_edge(
                (node, layer),
                (node, layer + 1),
                edge_type="imaginary",
                layer=layer,
                physical_node=node,
            )

    # print("The number of nodes of layered graph", len(layered_graph.nodes()))
    # print("The number of links of layered graph", len(layered_graph.edges()))

    source = (origin, 0)
    sink = (destination, L)
    if source not in layered_graph or sink not in layered_graph:
        raise ValueError("The layered graph has no valid source or sink.")

    graph_creation_time = (time.perf_counter() - graph_creation_start) * 1000
    print(f"Graph creation time: {graph_creation_time} ms")

    problem = pulp.LpProblem(
        "Model_splitting_placement_chaining_layered", pulp.LpMinimize
    )
    y = {}
    for u, v in layered_graph.edges():
        y[u, v] = pulp.LpVariable(f"y_{u[0]}_{u[1]}_{v[0]}_{v[1]}", 0, 1, pulp.LpBinary)
    # print(f"Number of path variables: {len(y)}")

    # A single source-to-sink path in the layered graph.
    for node in layered_graph.nodes():
        incoming = pulp.lpSum(y[u, node] for u in layered_graph.predecessors(node))
        outgoing = pulp.lpSum(y[node, v] for v in layered_graph.successors(node))
        if node == source:
            problem += outgoing - incoming == 1, f"flow_source_{node}"
        elif node == sink:
            problem += incoming - outgoing == 1, f"flow_sink_{node}"
        else:
            problem += incoming == outgoing, f"flow_conservation_{node}"

    # Each Physical node receives the layers assigned to it by its selected
    # Imaginary links.  The activation memory is the maximum activation held
    # on that Physical node.
    # mem_variable = {
    #     node: pulp.LpVariable(f"mem_variable_{node}", lowBound=0, cat=pulp.LpContinuous)
    #     for node in physical_nodes
    # }
    # for node in physical_nodes:
    #     assigned_layers = [y[(node, layer), (node, layer + 1)] for layer in range(L)]
    #     # problem += (
    #     #     pulp.lpSum(
    #     #         assigned_layers[layer] * layer_storage_sizes[layer]
    #     #         for layer in range(L)
    #     #     )
    #     #     <= storage_capacity[node],
    #     #     f"storage_capacity_{node}",
    #     # )
    #     problem += (
    #         pulp.lpSum(
    #             assigned_layers[layer] * layer_memory_sizes[layer] for layer in range(L)
    #         )
    #         + mem_variable[node]
    #         <= memory_capacity[node],
    #         f"memory_capacity_{node}",
    #     )
    #     for layer in range(L):
    #         problem += (
    #             mem_variable[node]
    #             >= bs * activation_bytes[layer] * assigned_layers[layer],
    #             f"activation_memory_{node},{layer}",
    #         )

    # Match the boundary restrictions in the original formulation:
    # layer 0 is executed at origin and layer L-1 at destination.  Therefore
    # stage 0 and stage L contain only their Imaginary-link transition and no
    # Physical-network path.
    problem += y[(origin, 0), (origin, 1)] == 1, "first_layer_at_origin"
    problem += (
        y[(destination, L - 1), (destination, L)] == 1,
        ("last_layer_at_destination"),
    )
    for i, j in physical_edges:
        problem += y[(i, 0), (j, 0)] == 0, f"no_physical_path_before_layer0_{i},{j}"
        problem += y[(i, L), (j, L)] == 0, f"no_physical_path_after_last_layer_{i},{j}"

    def physical_link_cost(i, j, stage):
        # Match model_splitting_placement_chaining.py: the source-to-first-
        # layer segment is not included in link_propagation_delay.
        value = 0
        if stage >= 1:
            value = delay[(i, j)]
            if not is_test:
                value += delay.get((j, i), delay[(i, j)])
        if 1 <= stage <= L - 1:
            layer = stage - 1
            value += bs * activation_bytes[layer] / rate[(i, j)]
            if not is_test:
                value += bs * gradient_bytes[layer] / rate[(i, j)]
        return value

    objective_terms = []
    for u, v, attributes in layered_graph.edges(data=True):
        edge_variable = y[u, v]
        if attributes["edge_type"] == "physical":
            i, j = attributes["physical_edge"]
            objective_terms.append(
                edge_variable * physical_link_cost(i, j, attributes["stage"])
            )
        else:
            layer = attributes["layer"]
            node = attributes["physical_node"]
            compute_cost = tau[node] + kappa[node] * fwd_flops[layer]
            if not is_test:
                compute_cost += tau[node] + kappa[node] * bwd_flops[layer]
            objective_terms.append(edge_variable * compute_cost)
    problem += pulp.lpSum(objective_terms), "Objective function"
    # print(f"Number of constraints: {len(problem.constraints)}")

    # problem.writeLP("results/ilp_layered.lp")
    solver = pulp.GUROBI_CMD(
        msg=True,
        timeLimit=1000,
        options=[
            ("MIPGap", 0),
            ("MIPGapAbs", 0),
            ("IntFeasTol", 1e-9),
            ("FeasibilityTol", 1e-9),
            ("OptimalityTol", 1e-9),
        ],
    )
    start_time = time.perf_counter()
    status = problem.solve(solver)
    end_time = time.perf_counter()
    elapsed_time = (end_time - start_time) * 1000
    elapsed_time_from_graph_creation = (end_time - graph_creation_start) * 1000

    results = {"status": pulp.LpStatus[status]}
    if status != pulp.LpStatusOptimal:
        print("The problem does not have an optimal solution.")
        return

    physical_edges_by_stage = {stage: [] for stage in range(L + 1)}
    x_result = []
    imaginary_node_offset = len(physical_nodes)
    for u, v in layered_graph.edges():
        value = pulp.value(y[u, v])
        if value is None or value <= 0:
            continue
        attributes = layered_graph[u][v]
        if attributes["edge_type"] == "physical":
            i, j = attributes["physical_edge"]
            stage = attributes["stage"]
            physical_edges_by_stage[stage].append((i, j, value))
            x_result.append({"i": int(i), "j": int(j), "k": stage, "value": value})
        else:
            layer = attributes["layer"]
            node = attributes["physical_node"]
            imaginary_node = imaginary_node_offset + layer
            # Expand one layered-graph transition into the two directed
            # Imaginary-link records used by the original output format.
            x_result.append(
                {
                    "i": int(node),
                    "j": int(imaginary_node),
                    "k": layer,
                    "value": value,
                }
            )
            x_result.append(
                {
                    "i": int(imaginary_node),
                    "j": int(node),
                    "k": layer + 1,
                    "value": value,
                }
            )

    layer_placement = []
    selected_hosts = []
    for layer in range(L):
        for node in physical_nodes:
            value = pulp.value(y[(node, layer), (node, layer + 1)])
            if value is not None and value > 0:
                selected_hosts.append(node)
                layer_placement.append(
                    {
                        "layer": layer,
                        "imaginary_node": len(physical_nodes) + layer,
                        "physical_node": node,
                        "value": value,
                    }
                )
                break
    results["x"] = x_result
    results["imaginary_link_placement"] = layer_placement
    results["splitting_points"] = [
        {
            "layer": layer,
            "from_physical_node": selected_hosts[layer - 1],
            "to_physical_node": selected_hosts[layer],
        }
        for layer in range(1, len(selected_hosts))
        if selected_hosts[layer - 1] != selected_hosts[layer]
    ]

    def selected_link_delay(layer):
        return sum(
            value
            * (delay[(i, j)] + (0 if is_test else delay.get((j, i), delay[(i, j)])))
            for i, j, value in physical_edges_by_stage[layer + 1]
        )

    def selected_transmission_delay(layer, data_bytes):
        if layer >= L - 1:
            return 0
        return sum(
            value * bs * data_bytes[layer] / rate[(i, j)]
            for i, j, value in physical_edges_by_stage[layer + 1]
        )

    forward_delays = {
        layer: tau[selected_hosts[layer]]
        + kappa[selected_hosts[layer]] * fwd_flops[layer]
        for layer in range(L)
    }
    results["overall_forward_propagation_delay"] = sum(forward_delays.values())
    results["forward_propagation_delay"] = forward_delays
    results["overall_link_propagation_delay"] = sum(
        selected_link_delay(layer) for layer in range(L)
    )
    results["link_propagation_delay"] = {
        layer: selected_link_delay(layer) for layer in range(L)
    }
    results["activation_transmission_delay"] = {
        layer: selected_transmission_delay(layer, activation_bytes)
        for layer in range(L - 1)
    }
    results["overall_activation_transmission_delay"] = sum(
        results["activation_transmission_delay"].values()
    )
    results["activation_bytes"] = {
        layer: activation_bytes[layer] for layer in range(L - 1)
    }

    if not is_test:
        backward_delays = {
            layer: tau[selected_hosts[layer]]
            + kappa[selected_hosts[layer]] * bwd_flops[layer]
            for layer in range(L)
        }
        results["overall_backward_propagation_delay"] = sum(backward_delays.values())
        results["backward_propagation_delay"] = backward_delays
        results["gradient_transmission_delay"] = {
            layer: selected_transmission_delay(layer, gradient_bytes)
            for layer in range(L - 1)
        }
        results["overall_gradient_transmission_delay"] = sum(
            results["gradient_transmission_delay"].values()
        )
        results["gradient_bytes"] = {
            layer: gradient_bytes[layer] for layer in range(L - 1)
        }

    results["objective_value"] = pulp.value(problem.objective)
    results["elapsed_time"] = elapsed_time
    results["elapsed_time_from_graph_creation"] = elapsed_time_from_graph_creation
    results["graph_creation_time"] = graph_creation_time

    # print(results)
    os.makedirs(resdir, exist_ok=True)
    output_filename = f"{resdir}/lp_layered_bs{bs}_seed{seed}_is_test{is_test}.json"
    with open(output_filename, "w") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    print(
        f"Status: {results['status']}, Objective value: {results['objective_value']}, "
        f"Elapsed time: {elapsed_time} ms"
    )
    print(f"Results: {output_filename}")
    print("is_cpu =", is_cpu)
    print("origin =", origin, "destination =", destination)
    print("last layer flops =", fwd_flops[-1])
    print("destination tau =", tau[destination])
    print("destination kappa =", kappa[destination])
    print(
        "last layer compute delay =",
        tau[destination] + kappa[destination] * fwd_flops[-1],
    )


if __name__ == "__main__":
    main()
