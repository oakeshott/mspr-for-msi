import argparse
import json
import os
import time

import networkx as nx
import pandas as pd
import pulp


parser = argparse.ArgumentParser(
    description="Model splitting with independent placement variables"
)
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


def value_of(variable):
    value = pulp.value(variable)
    return 0.0 if value is None else float(value)


def main():
    physical_network = nx.read_gml(f"{datadir}/network.gml", label="id")
    if not physical_network.is_directed():
        physical_network = physical_network.to_directed()

    with open(f"{datadir}/function_placement_candidates.json") as f:
        function_placement = json.load(f)
    origin = function_placement["origin"]
    destination = function_placement["destination"]

    data = load_model_data(f"{datadir}/{args.model}.xlsx")
    layer_count = len(data)
    if layer_count == 0:
        raise ValueError("The model contains no layers.")

    physical_nodes = list(physical_network.nodes())
    physical_edges = list(physical_network.edges())
    node_count = len(physical_nodes)
    edge_count = len(physical_edges)
    if origin not in physical_network or destination not in physical_network:
        raise ValueError("origin and destination must be Physical nodes.")

    activation_bytes = [row["activation_bytes"] for row in data]
    gradient_bytes = [row["gradient_bytes"] for row in data]
    fwd_flops = [row["FWD_FLOPs"] for row in data]
    bwd_flops = [row["BWD_FLOPs"] for row in data]
    layer_memory_sizes = [row["memory"] for row in data]
    layer_storage_sizes = [row["memory"] for row in data]

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

    # z[i, k] = 1 iff model layer k is placed on Physical node i.
    # p[i, j, s] = 1 iff Physical link (i, j) is used by segment s.
    # Segment 0 is origin -> layer 0, segment L is layer L-1 -> destination.
    problem = pulp.LpProblem("Model_splitting_placement_chaining_z", pulp.LpMinimize)

    z = {
        (node, layer): pulp.LpVariable(
            f"z_{node}_{layer}", lowBound=0, upBound=1, cat=pulp.LpBinary
        )
        for node in physical_nodes
        for layer in range(layer_count)
    }
    p = {
        (i, j, segment): pulp.LpVariable(
            f"p_{i}_{j}_{segment}", lowBound=0, upBound=1, cat=pulp.LpBinary
        )
        for segment in range(layer_count + 1)
        for i, j in physical_edges
    }

    print("Number of Physical nodes:", node_count)
    print("Number of directed Physical links:", edge_count)
    print("Number of placement variables z:", len(z))
    print("Number of physical-path variables p:", len(p))

    # Exactly one Physical node hosts each model layer.
    for layer in range(layer_count):
        problem += (
            pulp.lpSum(z[node, layer] for node in physical_nodes) == 1,
            f"one_host_for_layer_{layer}",
        )

    # Match the boundary restrictions of the original formulation.
    problem += z[origin, 0] == 1, "first_layer_at_origin"
    problem += (
        z[destination, layer_count - 1] == 1,
        "last_layer_at_destination",
    )

    # Physical flow for every segment.  The right-hand side is
    # source_indicator - destination_indicator, so z variables select the
    # endpoints while p variables select the physical path between them.
    for segment in range(layer_count + 1):
        for node in physical_nodes:
            incoming = pulp.lpSum(
                p[i, node, segment] for i in physical_network.predecessors(node)
            )
            outgoing = pulp.lpSum(
                p[node, j, segment] for j in physical_network.successors(node)
            )

            if segment == 0:
                rhs = (1 if node == origin else 0) - z[node, 0]
            elif segment == layer_count:
                rhs = z[node, layer_count - 1] - (1 if node == destination else 0)
            else:
                rhs = z[node, segment - 1] - z[node, segment]

            problem += (
                outgoing - incoming == rhs,
                f"physical_flow_{node}_{segment}",
            )

    # Capacity constraints.  mem_variable represents the maximum activation
    # size among the layers placed on a Physical node.
    # mem_variable = {
    #     node: pulp.LpVariable(
    #         f"mem_variable_{node}",
    #         lowBound=0,
    #         upBound=memory_capacity[node],
    #         cat=pulp.LpContinuous,
    #     )
    #     for node in physical_nodes
    # }
    # for node in physical_nodes:
    #     # problem += (
    #     #     pulp.lpSum(
    #     #         z[node, layer] * layer_storage_sizes[layer]
    #     #         for layer in range(layer_count)
    #     #     )
    #     #     <= storage_capacity[node],
    #     #     f"storage_capacity_{node}",
    #     # )
    #     problem += (
    #         pulp.lpSum(
    #             z[node, layer] * layer_memory_sizes[layer]
    #             for layer in range(layer_count)
    #         )
    #         + mem_variable[node]
    #         <= memory_capacity[node],
    #         f"memory_capacity_{node}",
    #     )
    #     for layer in range(layer_count):
    #         problem += (
    #             mem_variable[node] >= bs * activation_bytes[layer] * z[node, layer],
    #             f"activation_memory_{node}_{layer}",
    #         )

    def physical_link_cost(i, j, segment):
        # This follows the objective convention of the existing programs:
        # source -> first layer has no communication term; communication for
        # layer k is charged on segment k+1, except after the final layer.
        if segment == 0:
            return 0.0

        value = delay[i, j]
        if not is_test:
            value += delay.get((j, i), delay[i, j])

        if 1 <= segment <= layer_count - 1:
            layer = segment - 1
            value += bs * activation_bytes[layer] / rate[i, j]
            if not is_test:
                value += bs * gradient_bytes[layer] / rate[i, j]
        return value

    placement_cost = {}
    for node in physical_nodes:
        for layer in range(layer_count):
            value = tau[node] + kappa[node] * fwd_flops[layer]
            if not is_test:
                value += tau[node] + kappa[node] * bwd_flops[layer]
            placement_cost[node, layer] = value

    problem += (
        pulp.lpSum(
            z[node, layer] * placement_cost[node, layer]
            for node in physical_nodes
            for layer in range(layer_count)
        )
        + pulp.lpSum(
            p[i, j, segment] * physical_link_cost(i, j, segment)
            for segment in range(layer_count + 1)
            for i, j in physical_edges
        ),
        "Objective function",
    )

    os.makedirs(resdir, exist_ok=True)
    # problem.writeLP(f"{resdir}/ilp_z.lp")
    print("Number of variables:", len(problem.variables()))
    print("Number of constraints:", len(problem.constraints))

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

    results = {
        "status": pulp.LpStatus[status],
        "number_of_physical_nodes": node_count,
        "number_of_physical_links": edge_count,
        "number_of_placement_variables": len(z),
        "number_of_physical_path_variables": len(p),
        "number_of_variables": len(problem.variables()),
        "number_of_constraints": len(problem.constraints),
    }

    if status != pulp.LpStatusOptimal:
        print("The problem does not have an optimal solution.")
        results["objective_value"] = None
        results["elapsed_time"] = (end_time - start_time) * 1000
        with open(f"{resdir}/lp_z_bs{bs}_seed{seed}_is_test{is_test}.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        return

    objective_value = pulp.value(problem.objective)
    elapsed_time = (end_time - start_time) * 1000
    print(
        f"Status: {results['status']}, Objective value: {objective_value}, "
        f"Elapsed time: {elapsed_time} ms"
    )
    results["objective_value"] = objective_value
    results["elapsed_time"] = elapsed_time

    selected_hosts = []
    placement = []
    z_output = []
    for layer in range(layer_count):
        selected_node = None
        for node in physical_nodes:
            value = value_of(z[node, layer])
            z_output.append(
                {
                    "layer": layer,
                    "physical_node": int(node),
                    "value": value,
                }
            )
            if value > 0.5:
                selected_node = node
        if selected_node is None:
            raise RuntimeError(f"No Physical node selected for layer {layer}.")
        selected_hosts.append(int(selected_node))
        placement.append(
            {
                "layer": layer,
                "imaginary_node": node_count + layer,
                "physical_node": int(selected_node),
                "value": 1.0,
            }
        )

    results["z"] = z_output
    results["imaginary_link_placement"] = placement

    splitting_points = []
    for layer in range(1, layer_count):
        if selected_hosts[layer] != selected_hosts[layer - 1]:
            splitting_points.append(
                {
                    "layer": layer,
                    "from_physical_node": selected_hosts[layer - 1],
                    "to_physical_node": selected_hosts[layer],
                }
            )
    results["splitting_points"] = splitting_points

    # Export physical paths and reconstruct the original augmented-network
    # representation for compatibility with existing result consumers.
    physical_path = []
    service_path = []
    for segment in range(layer_count + 1):
        for i, j in physical_edges:
            if value_of(p[i, j, segment]) > 0.5:
                record = {"i": int(i), "j": int(j), "k": segment, "value": 1.0}
                physical_path.append(record)
                service_path.append(record)

        if segment < layer_count:
            host = selected_hosts[segment]
            imaginary_node = node_count + segment
            service_path.append(
                {
                    "i": host,
                    "j": imaginary_node,
                    "k": segment,
                    "value": 1.0,
                }
            )
        if segment > 0:
            host = selected_hosts[segment - 1]
            imaginary_node = node_count + segment - 1
            service_path.append(
                {
                    "i": imaginary_node,
                    "j": host,
                    "k": segment,
                    "value": 1.0,
                }
            )

    results["x"] = service_path
    results["physical_path"] = physical_path

    def path_cost(segment, cost_function):
        return sum(
            value_of(p[i, j, segment]) * cost_function(i, j, segment)
            for i, j in physical_edges
        )

    forward_delay = sum(
        value_of(z[node, layer]) * (tau[node] + kappa[node] * fwd_flops[layer])
        for node in physical_nodes
        for layer in range(layer_count)
    )
    backward_delay = sum(
        value_of(z[node, layer]) * (tau[node] + kappa[node] * bwd_flops[layer])
        for node in physical_nodes
        for layer in range(layer_count)
    )
    link_delay = sum(
        path_cost(
            segment,
            lambda i, j, _segment: delay[i, j]
            + (0 if is_test else delay.get((j, i), delay[i, j])),
        )
        for segment in range(1, layer_count + 1)
    )
    activation_delay = sum(
        path_cost(
            segment,
            lambda i, j, current_segment: (
                bs * activation_bytes[current_segment - 1] / rate[i, j]
                if 1 <= current_segment <= layer_count - 1
                else 0
            ),
        )
        for segment in range(layer_count + 1)
    )
    gradient_delay = sum(
        path_cost(
            segment,
            lambda i, j, current_segment: (
                bs * gradient_bytes[current_segment - 1] / rate[i, j]
                if 1 <= current_segment <= layer_count - 1
                else 0
            ),
        )
        for segment in range(layer_count + 1)
    )

    results["overall_forward_propagation_delay"] = forward_delay
    results["overall_link_propagation_delay"] = link_delay
    results["overall_activation_transmission_delay"] = activation_delay
    results["forward_propagation_delay"] = {
        layer: tau[selected_hosts[layer]]
        + kappa[selected_hosts[layer]] * fwd_flops[layer]
        for layer in range(layer_count)
    }
    results["activation_transmission_delay"] = {
        layer: path_cost(
            layer + 1,
            lambda i, j, _segment: bs * activation_bytes[layer] / rate[i, j],
        )
        for layer in range(layer_count - 1)
    }
    results["link_propagation_delay"] = {
        layer: path_cost(
            layer + 1,
            lambda i, j, _segment: delay[i, j]
            + (0 if is_test else delay.get((j, i), delay[i, j])),
        )
        for layer in range(layer_count)
    }
    results["activation_bytes"] = {
        layer: activation_bytes[layer] for layer in range(layer_count - 1)
    }

    if not is_test:
        results["overall_backward_propagation_delay"] = backward_delay
        results["overall_gradient_transmission_delay"] = gradient_delay
        results["backward_propagation_delay"] = {
            layer: tau[selected_hosts[layer]]
            + kappa[selected_hosts[layer]] * bwd_flops[layer]
            for layer in range(layer_count)
        }
        results["gradient_transmission_delay"] = {
            layer: path_cost(
                layer + 1,
                lambda i, j, _segment: bs * gradient_bytes[layer] / rate[i, j],
            )
            for layer in range(layer_count - 1)
        }
        results["gradient_bytes"] = {
            layer: gradient_bytes[layer] for layer in range(layer_count - 1)
        }

    output_filename = f"{resdir}/lp_z_bs{bs}_seed{seed}_is_test{is_test}.json"
    with open(output_filename, "w") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    # print(results)
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
