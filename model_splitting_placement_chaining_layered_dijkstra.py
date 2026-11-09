import argparse
import heapq
import json
import os
import time

import networkx as nx
import pandas as pd


parser = argparse.ArgumentParser(
    description="Model splitting placement and chaining with Layered graph+Dijkstra"
)
parser.add_argument("--bs", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--datadir", type=str, default="data")
parser.add_argument("--resdir", "-o", type=str, default="results")
parser.add_argument("--model", type=str, default="resnet101")
parser.add_argument("--is_test", action="store_true")
parser.add_argument("--is_cpu", action="store_true")
parser.add_argument(
    "--capacity",
    type=float,
    default=None,
    help="Accepted for command-line compatibility; ignored by Dijkstra.",
)
args = parser.parse_args()

bs = args.bs
seed = args.seed
datadir = args.datadir
resdir = args.resdir
is_test = args.is_test
is_cpu = args.is_cpu

TIME_LIMIT_SECONDS = 1000


class SearchTimeout(Exception):
    def __init__(self, elapsed_seconds):
        self.elapsed_seconds = elapsed_seconds


class HeapStats:
    """Collect the same priority-queue metrics for every solver."""

    def __init__(self, number_of_heaps):
        self.number_of_heaps = number_of_heaps
        self.push_count = 0
        self.pop_count = 0
        self.stale_pop_count = 0
        self.current_size = 0
        self.peak_size = 0
        self.peak_sizes = [0] * number_of_heaps

    def push(self, heap, item, heap_id=0):
        heapq.heappush(heap, item)
        self.push_count += 1
        self.current_size += 1
        self.peak_size = max(self.peak_size, self.current_size)
        if self.number_of_heaps:
            self.peak_sizes[heap_id] = max(self.peak_sizes[heap_id], len(heap))

    def pop(self, heap, heap_id=0):
        item = heapq.heappop(heap)
        self.pop_count += 1
        self.current_size -= 1
        return item

    def mark_stale(self):
        self.stale_pop_count += 1

    def as_dict(self, heaps):
        return {
            "number_of_heaps": self.number_of_heaps,
            "push_count": self.push_count,
            "pop_count": self.pop_count,
            "stale_pop_count": self.stale_pop_count,
            "peak_size": self.peak_size,
            "peak_sizes": self.peak_sizes,
            "final_size": sum(len(heap) for heap in heaps),
        }


def check_timeout(start_time):
    elapsed_seconds = time.perf_counter() - start_time
    if elapsed_seconds > TIME_LIMIT_SECONDS:
        raise SearchTimeout(elapsed_seconds)


def save_timeout_result(filename, elapsed_seconds):
    os.makedirs(resdir, exist_ok=True)
    with open(filename, "w") as f:
        json.dump(
            {
                "status": "Infeasible",
                "elapsed_time": elapsed_seconds * 1000,
                "reason": "Time limit exceeded (1000 s)",
            },
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )
    print(f"Status: Infeasible, Elapsed time: {elapsed_seconds * 1000} ms")
    print(f"Results: {filename}")


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
    layer_count = len(data)
    if layer_count == 0:
        raise ValueError("The model contains no layers.")
    if layer_count == 1 and origin != destination:
        raise ValueError(
            "With one layer, the first-layer-at-origin and "
            "last-layer-at-destination restrictions conflict."
        )

    physical_nodes = list(physical_network.nodes())
    physical_edges = list(physical_network.edges())
    if origin not in physical_network or destination not in physical_network:
        raise ValueError("origin and destination must be Physical nodes.")

    activation_bytes = [row["activation_bytes"] for row in data]
    gradient_bytes = [row["gradient_bytes"] for row in data]
    fwd_flops = [row["FWD_FLOPs"] for row in data]
    bwd_flops = [row["BWD_FLOPs"] for row in data]

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

    def compute_cost(node, layer):
        value = tau[node] + kappa[node] * fwd_flops[layer]
        if not is_test:
            value += tau[node] + kappa[node] * bwd_flops[layer]
        return value

    def communication_cost(i, j, layer):
        # This is the cost of the physical segment from model layer `layer`
        # to model layer `layer + 1`.  There is no segment after the final
        # layer because the final layer is fixed at destination.
        value = delay[i, j]
        if not is_test:
            value += delay.get((j, i), delay[i, j])
        value += bs * activation_bytes[layer] / rate[i, j]
        if not is_test:
            value += bs * gradient_bytes[layer] / rate[i, j]
        return value

    # Layered graph nodes are (Physical node, stage), where stage k means
    # that k model layers have been completed.  The source is (origin, 0)
    # and the sink is (destination, L).
    graph_creation_start = time.perf_counter()
    layered_graph = nx.DiGraph()
    for stage in range(layer_count + 1):
        for node in physical_nodes:
            layered_graph.add_node((node, stage), stage=stage)

    # Physical edges are copied into stages 1,...,L-1.  Stage 0 is omitted
    # because layer 0 is fixed at origin; stage L is omitted because the
    # final layer is fixed at destination.
    for stage in range(1, layer_count):
        for i, j in physical_edges:
            layer = stage - 1
            layered_graph.add_edge(
                (i, stage),
                (j, stage),
                edge_type="physical",
                stage=stage,
                physical_edge=(i, j),
                weight=communication_cost(i, j, layer),
            )

    # Imaginary edges represent layer placement.  Their weights are the
    # computation costs of the corresponding model layer.
    for layer in range(layer_count):
        if layer == 0:
            candidates = [origin]
        elif layer == layer_count - 1:
            candidates = [destination]
        else:
            candidates = physical_nodes

        for node in candidates:
            layered_graph.add_edge(
                (node, layer),
                (node, layer + 1),
                edge_type="imaginary",
                layer=layer,
                physical_node=node,
                weight=compute_cost(node, layer),
            )

    source = (origin, 0)
    sink = (destination, layer_count)
    if source not in layered_graph or sink not in layered_graph:
        raise ValueError("The layered graph has no valid source or sink.")

    graph_creation_time = (time.perf_counter() - graph_creation_start) * 1000
    print(f"Graph creation time: {graph_creation_time} ms")

    print("Number of layered graph nodes:", layered_graph.number_of_nodes())
    print("Number of layered graph links:", layered_graph.number_of_edges())
    print("Number of Physical nodes:", len(physical_nodes))
    print("Number of directed Physical links:", len(physical_edges))

    start_search = time.perf_counter()

    # Use an explicit heapq-based Dijkstra so that the search behavior and
    # heap operations are measurable independently of NetworkX algorithms.
    heap_stats = HeapStats(1)
    heap = []
    distance = {state: float("inf") for state in layered_graph.nodes}
    predecessor = {}
    settled = set()
    push_serial = 0
    distance[source] = 0.0
    heap_stats.push(heap, (0.0, push_serial, source))

    while heap:
        check_timeout(start_search)
        cost, _, state = heap_stats.pop(heap)
        if state in settled or cost != distance[state]:
            heap_stats.mark_stale()
            continue

        settled.add(state)
        if state == sink:
            break

        for next_state in layered_graph.successors(state):
            check_timeout(start_search)
            edge_data = layered_graph[state][next_state]
            next_cost = cost + edge_data["weight"]
            if next_cost < distance[next_state]:
                distance[next_state] = next_cost
                predecessor[next_state] = state
                push_serial += 1
                heap_stats.push(heap, (next_cost, push_serial, next_state))

    if sink not in settled:
        raise RuntimeError("Dijkstra could not find a feasible placement path.")

    objective_value = distance[sink]
    path = [sink]
    while path[-1] != source:
        previous = predecessor.get(path[-1])
        if previous is None:
            raise RuntimeError("Dijkstra path reconstruction failed.")
        path.append(previous)
    path.reverse()
    check_timeout(start_search)
    end_search = time.perf_counter()
    elapsed_time = (end_search - start_search) * 1000
    elapsed_time_from_graph_creation = (end_search - graph_creation_start) * 1000

    selected_hosts = [None] * layer_count
    physical_path_by_stage = {stage: [] for stage in range(layer_count + 1)}
    service_path = []
    placement = []

    for u, v in zip(path[:-1], path[1:]):
        attributes = layered_graph[u][v]
        if attributes["edge_type"] == "physical":
            i, j = attributes["physical_edge"]
            stage = attributes["stage"]
            physical_path_by_stage[stage].append((i, j))
            service_path.append({"i": int(i), "j": int(j), "k": stage, "value": 1.0})
        else:
            layer = attributes["layer"]
            node = attributes["physical_node"]
            selected_hosts[layer] = node
            imaginary_node = len(physical_nodes) + layer
            placement.append(
                {
                    "layer": layer,
                    "imaginary_node": imaginary_node,
                    "physical_node": int(node),
                    "value": 1.0,
                }
            )
            service_path.append(
                {
                    "i": int(node),
                    "j": imaginary_node,
                    "k": layer,
                    "value": 1.0,
                }
            )
            service_path.append(
                {
                    "i": imaginary_node,
                    "j": int(node),
                    "k": layer + 1,
                    "value": 1.0,
                }
            )

    if any(node is None for node in selected_hosts):
        raise RuntimeError("Dijkstra path does not contain every model layer.")

    splitting_points = [
        {
            "layer": layer,
            "from_physical_node": int(selected_hosts[layer - 1]),
            "to_physical_node": int(selected_hosts[layer]),
        }
        for layer in range(1, layer_count)
        if selected_hosts[layer - 1] != selected_hosts[layer]
    ]

    def sum_stage_cost(stage, cost_function):
        return sum(
            cost_function(i, j) for i, j in physical_path_by_stage.get(stage, [])
        )

    forward_delay = sum(
        tau[selected_hosts[layer]] + kappa[selected_hosts[layer]] * fwd_flops[layer]
        for layer in range(layer_count)
    )
    backward_delay = sum(
        tau[selected_hosts[layer]] + kappa[selected_hosts[layer]] * bwd_flops[layer]
        for layer in range(layer_count)
    )

    link_delay_by_layer = {
        layer: sum_stage_cost(
            layer + 1,
            lambda i, j: delay[i, j]
            + (0 if is_test else delay.get((j, i), delay[i, j])),
        )
        for layer in range(layer_count - 1)
    }
    activation_delay_by_layer = {
        layer: sum_stage_cost(
            layer + 1,
            lambda i, j: bs * activation_bytes[layer] / rate[i, j],
        )
        for layer in range(layer_count - 1)
    }
    gradient_delay_by_layer = {
        layer: sum_stage_cost(
            layer + 1,
            lambda i, j: bs * gradient_bytes[layer] / rate[i, j],
        )
        for layer in range(layer_count - 1)
    }

    results = {
        "status": "Optimal",
        "objective_value": objective_value,
        "elapsed_time": elapsed_time,
        "elapsed_time_from_graph_creation": elapsed_time_from_graph_creation,
        "graph_creation_time": graph_creation_time,
        "number_of_layered_graph_nodes": layered_graph.number_of_nodes(),
        "number_of_layered_graph_links": layered_graph.number_of_edges(),
        "number_of_physical_nodes": len(physical_nodes),
        "number_of_physical_links": len(physical_edges),
        "heap_statistics": heap_stats.as_dict([heap]),
        "capacity_constraints": "ignored",
        "is_cpu": is_cpu,
        "origin": origin,
        "destination": destination,
        "x": service_path,
        "imaginary_link_placement": placement,
        "splitting_points": splitting_points,
        "overall_forward_propagation_delay": forward_delay,
        "overall_link_propagation_delay": sum(link_delay_by_layer.values()),
        "overall_activation_transmission_delay": sum(
            activation_delay_by_layer.values()
        ),
        "forward_propagation_delay": {
            layer: tau[selected_hosts[layer]]
            + kappa[selected_hosts[layer]] * fwd_flops[layer]
            for layer in range(layer_count)
        },
        "link_propagation_delay": link_delay_by_layer,
        "activation_transmission_delay": activation_delay_by_layer,
        "activation_bytes": {
            layer: activation_bytes[layer] for layer in range(layer_count - 1)
        },
    }

    if not is_test:
        results["overall_backward_propagation_delay"] = backward_delay
        results["overall_gradient_transmission_delay"] = sum(
            gradient_delay_by_layer.values()
        )
        results["backward_propagation_delay"] = {
            layer: tau[selected_hosts[layer]]
            + kappa[selected_hosts[layer]] * bwd_flops[layer]
            for layer in range(layer_count)
        }
        results["gradient_transmission_delay"] = gradient_delay_by_layer
        results["gradient_bytes"] = {
            layer: gradient_bytes[layer] for layer in range(layer_count - 1)
        }

    os.makedirs(resdir, exist_ok=True)
    output_filename = (
        f"{resdir}/layered_dijkstra_bs{bs}_seed{seed}_is_test{is_test}.json"
    )
    check_timeout(start_search)
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
        f"Status: Optimal, Objective value: {objective_value}, "
        f"Elapsed time: {elapsed_time} ms"
    )
    print(f"Results: {output_filename}")


if __name__ == "__main__":
    try:
        main()
    except SearchTimeout as error:
        save_timeout_result(
            f"{resdir}/layered_dijkstra_bs{bs}_seed{seed}_is_test{is_test}.json",
            error.elapsed_seconds,
        )
