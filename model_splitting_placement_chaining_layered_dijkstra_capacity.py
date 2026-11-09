import argparse
import heapq
import json
import os
import time

import networkx as nx
import pandas as pd


parser = argparse.ArgumentParser(
    description=(
        "Capacity-constrained model splitting with a layered-graph "
        "resource-constrained shortest-path label-setting algorithm"
    )
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
    help=(
        "Capacity for every node except origin and destination. "
        "If omitted, use 1e10, as in the layered ILP."
    ),
)
args = parser.parse_args()

bs = args.bs
seed = args.seed
datadir = args.datadir
resdir = args.resdir
is_test = args.is_test
is_cpu = args.is_cpu
capacity = 1e10 if args.capacity is None else float(args.capacity)

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


def make_capacities(nodes, origin, destination, global_capacity):
    """Use the same capacities as model_splitting_placement_chaining_layered.py."""
    if global_capacity < 0:
        raise ValueError("Capacity must be non-negative.")

    memory_capacity = {}
    storage_capacity = {}
    for node in nodes:
        # if node == origin or node == destination:
        #     memory_capacity[node] = 8e9
        #     storage_capacity[node] = 8e14
        # else:
        #     memory_capacity[node] = global_capacity
        #     storage_capacity[node] = global_capacity
        memory_capacity[node] = capacity
        storage_capacity[node] = capacity
    return memory_capacity, storage_capacity


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
    layer_storage_sizes = [float(row["memory"]) for row in data]
    layer_memory_sizes = layer_storage_sizes[:]

    memory_capacity, storage_capacity = make_capacities(
        physical_nodes, origin, destination, capacity
    )

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
        value = delay[i, j]
        if not is_test:
            value += delay.get((j, i), delay[i, j])
        value += bs * activation_bytes[layer] / rate[i, j]
        if not is_test:
            value += bs * gradient_bytes[layer] / rate[i, j]
        return value

    # Build the same layered graph as the unconstrained Dijkstra version.
    # Stage k means that layers 0,...,k-1 have been completed.
    graph_creation_start = time.perf_counter()
    layered_graph = nx.DiGraph()
    for stage in range(layer_count + 1):
        for node in physical_nodes:
            layered_graph.add_node((node, stage), stage=stage)

    # No communication is allowed before layer 0 or after the final layer.
    for stage in range(1, layer_count):
        layer = stage - 1
        for i, j in physical_edges:
            layered_graph.add_edge(
                (i, stage),
                (j, stage),
                edge_type="physical",
                stage=stage,
                physical_edge=(i, j),
                weight=communication_cost(i, j, layer),
            )

    for layer in range(layer_count):
        if layer == 0:
            candidates = (origin,)
        elif layer == layer_count - 1:
            candidates = (destination,)
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

    start_search = time.perf_counter()

    # A single distance per (node, stage) cannot represent the capacity
    # history.  Each label therefore stores cumulative storage and maximum
    # activation memory at every physical node.  Labels are organized by
    # layered node, and a label is retained only if no label at that same
    # layered node has both no greater cost and no greater resource usage.
    # This is safe because all later resource increments are non-negative and
    # depend only on the layered node and the continuation, not on the path
    # used to reach it.
    node_index = {node: index for index, node in enumerate(physical_nodes)}
    zero_storage = (0.0,) * len(physical_nodes)
    zero_activation = (0.0,) * len(physical_nodes)

    labels = []
    labels_by_state = {}
    heap = []
    heap_stats = HeapStats(1)
    number_of_labels_pruned_by_dominance = 0

    def resource_vector_dominates(
        left_cost,
        left_storage,
        left_activation,
        right_cost,
        right_storage,
        right_activation,
    ):
        """Return whether the left resource/cost tuple covers the right one.

        Equality is included deliberately: an identical label is redundant.
        All values are non-negative, so componentwise domination remains valid
        for every continuation of the two paths.
        """
        return (
            left_cost <= right_cost
            and all(
                left_value <= right_value
                for left_value, right_value in zip(left_storage, right_storage)
            )
            and all(
                left_value <= right_value
                for left_value, right_value in zip(left_activation, right_activation)
            )
        )

    def add_label(state, cost, storage, activation, parent_id, edge):
        """Insert a label if it is not dominated at its layered node."""
        nonlocal number_of_labels_pruned_by_dominance

        check_timeout(start_search)
        state_labels = labels_by_state.setdefault(state, [])
        for old_id in state_labels:
            check_timeout(start_search)
            old_label = labels[old_id]
            if old_label["dominated"]:
                continue
            if resource_vector_dominates(
                old_label["cost"],
                old_label["storage"],
                old_label["activation"],
                cost,
                storage,
                activation,
            ):
                number_of_labels_pruned_by_dominance += 1
                return None

        label_id = len(labels)
        labels.append(
            {
                "state": state,
                "cost": cost,
                "storage": storage,
                "activation": activation,
                "parent": parent_id,
                "edge": edge,
                "settled": False,
                "dominated": False,
            }
        )

        # Remove labels dominated by the new label from this layered node's
        # active frontier.  Settled labels are also marked so stale heap
        # entries and already-generated descendants cannot affect the result.
        for old_id in state_labels:
            check_timeout(start_search)
            old_label = labels[old_id]
            if old_label["dominated"]:
                continue
            if resource_vector_dominates(
                cost,
                storage,
                activation,
                old_label["cost"],
                old_label["storage"],
                old_label["activation"],
            ) and (
                cost < old_label["cost"]
                or storage != old_label["storage"]
                or activation != old_label["activation"]
            ):
                old_label["dominated"] = True
                number_of_labels_pruned_by_dominance += 1

        state_labels.append(label_id)
        heap_stats.push(heap, (cost, label_id))
        return label_id

    def pop_minimum_label():
        """Pop the cheapest current label, ignoring stale heap entries."""
        while heap:
            check_timeout(start_search)
            cost, label_id = heap_stats.pop(heap)
            label = labels[label_id]
            if label["dominated"] or label["settled"]:
                heap_stats.mark_stale()
                continue
            if cost != label["cost"]:
                heap_stats.mark_stale()
                continue
            label["settled"] = True
            return label_id
        return None

    add_label(source, 0.0, zero_storage, zero_activation, None, None)

    def resource_after_imaginary_edge(label, node, layer):
        index = node_index[node]
        storage = list(label["storage"])
        activation = list(label["activation"])
        storage[index] += layer_storage_sizes[layer]
        activation[index] = max(activation[index], bs * activation_bytes[layer])

        if storage[index] > storage_capacity[node]:
            return None
        if storage[index] + activation[index] > memory_capacity[node]:
            return None
        return tuple(storage), tuple(activation)

    target_label_id = None
    while (label_id := pop_minimum_label()) is not None:
        check_timeout(start_search)
        label = labels[label_id]

        state = label["state"]
        if state == sink:
            target_label_id = label_id
            break

        for next_state in layered_graph.successors(state):
            check_timeout(start_search)
            edge_data = layered_graph[state][next_state]
            next_storage = label["storage"]
            next_activation = label["activation"]

            if edge_data["edge_type"] == "imaginary":
                layer = edge_data["layer"]
                resources = resource_after_imaginary_edge(
                    label, edge_data["physical_node"], layer
                )
                if resources is None:
                    continue
                next_storage, next_activation = resources

            add_label(
                next_state,
                label["cost"] + edge_data["weight"],
                next_storage,
                next_activation,
                label_id,
                (state, next_state),
            )

    check_timeout(start_search)
    end_search = time.perf_counter()
    elapsed_time = (end_search - start_search) * 1000
    elapsed_time_from_graph_creation = (end_search - graph_creation_start) * 1000
    if target_label_id is None:
        raise RuntimeError(
            "The resource-constrained label-setting search could not find "
            "a feasible placement path."
        )

    objective_value = labels[target_label_id]["cost"]

    # Reconstruct the selected layered-graph path from the winning label.
    path_edges = []
    current_label_id = target_label_id
    while current_label_id is not None:
        label = labels[current_label_id]
        if label["edge"] is not None:
            path_edges.append(label["edge"])
        current_label_id = label["parent"]
    path_edges.reverse()

    selected_hosts = [None] * layer_count
    physical_path_by_stage = {stage: [] for stage in range(layer_count + 1)}
    service_path = []
    placement = []
    physical_path = []

    for state, next_state in path_edges:
        attributes = layered_graph[state][next_state]
        if attributes["edge_type"] == "physical":
            i, j = attributes["physical_edge"]
            stage = attributes["stage"]
            physical_path_by_stage[stage].append((i, j))
            record = {"i": int(i), "j": int(j), "k": stage, "value": 1.0}
            physical_path.append(record)
            service_path.append(record)
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

    # Independent reconstruction-time capacity check, matching the label
    # transition test and the layered ILP constraints.
    storage_usage = {node: 0.0 for node in physical_nodes}
    activation_usage = {node: 0.0 for node in physical_nodes}
    for layer, node in enumerate(selected_hosts):
        storage_usage[node] += layer_storage_sizes[layer]
        activation_usage[node] = max(
            activation_usage[node], bs * activation_bytes[layer]
        )
    for node in physical_nodes:
        if storage_usage[node] > storage_capacity[node]:
            raise RuntimeError("Reconstructed path violates storage capacity.")
        if storage_usage[node] + activation_usage[node] > memory_capacity[node]:
            raise RuntimeError("Reconstructed path violates memory capacity.")

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

    target_label = labels[target_label_id]
    current_solution_state_label_count = sum(
        1
        for label_id in labels_by_state[target_label["state"]]
        if not labels[label_id]["dominated"]
    )
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
        "number_of_labels_generated": len(labels),
        "number_of_labels_pruned_by_dominance": number_of_labels_pruned_by_dominance,
        "number_of_labels_at_solution_state": current_solution_state_label_count,
        "heap_statistics": heap_stats.as_dict([heap]),
        "capacity_constraints": (
            "hard aggregate storage and memory capacity "
            "(weights plus maximum activation)"
        ),
        "is_cpu": is_cpu,
        "origin": origin,
        "destination": destination,
        "node_memory_capacity": {
            str(node): value for node, value in memory_capacity.items()
        },
        "node_storage_capacity": {
            str(node): value for node, value in storage_capacity.items()
        },
        "node_storage_usage": {
            str(node): value for node, value in storage_usage.items()
        },
        "node_activation_memory_usage": {
            str(node): value for node, value in activation_usage.items()
        },
        "node_memory_usage": {
            str(node): storage_usage[node] + activation_usage[node]
            for node in physical_nodes
        },
        "x": service_path,
        "physical_path": physical_path,
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
        f"{resdir}/layered_dijkstra_capacity_bs{bs}_seed{seed}_is_test{is_test}.json"
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
            f"{resdir}/layered_dijkstra_capacity_bs{bs}_seed{seed}_is_test{is_test}.json",
            error.elapsed_seconds,
        )
