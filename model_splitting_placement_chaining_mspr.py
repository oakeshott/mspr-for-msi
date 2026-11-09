import argparse
import heapq
import json
import os
import time

import networkx as nx
import pandas as pd


parser = argparse.ArgumentParser(
    description="Model splitting placement and chaining with MSPR"
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
    help="Accepted for command-line compatibility; ignored by MSPR.",
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
    edge_count = len(physical_edges)
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

    def layer_compute_cost(node, layer):
        value = tau[node] + kappa[node] * fwd_flops[layer]
        if not is_test:
            value += tau[node] + kappa[node] * bwd_flops[layer]
        return value

    def communication_cost(i, j, stage):
        """Cost of moving from layer stage-1 to layer stage.

        stage=1 corresponds to layer 0 -> layer 1.  Communication after the
        final layer is unnecessary because the final layer is fixed at the
        destination, so MSPR only uses stages 1,...,L-1.
        """
        value = delay[i, j]
        if not is_test:
            value += delay.get((j, i), delay[i, j])
        layer = stage - 1
        value += bs * activation_bytes[layer] / rate[i, j]
        if not is_test:
            value += bs * gradient_bytes[layer] / rate[i, j]
        return value

    # MSPR state:
    #   stage = number of completed model layers.
    #   (stage, node) means that the packet is at node after completing
    #   layers 0,...,stage-1 and is searching for layer stage.
    #
    # Layer 0 is fixed at origin, and layer L-1 is fixed at destination.
    # The algorithm uses one heap per tour stage, as in MSPR, but does not
    # construct the layered graph explicitly.
    start_time = time.perf_counter()
    heap_stats = HeapStats(0)
    if layer_count == 1:
        selected_hosts = [origin]
        objective_value = layer_compute_cost(origin, 0)
        physical_path_by_stage = {}
        mspr_elapsed = 0.0
    else:
        stage_count = layer_count + 1
        infinity = float("inf")
        distance = {
            stage: {node: infinity for node in physical_nodes}
            for stage in range(1, stage_count)
        }
        settled = {
            stage: {node: False for node in physical_nodes}
            for stage in range(1, stage_count)
        }
        parent = {
            stage: {node: None for node in physical_nodes}
            for stage in range(1, stage_count)
        }
        heaps = [[] for _ in range(stage_count)]
        heap_stats = HeapStats(layer_count)

        # Initial state: layer 0 is executed at origin.
        initial_cost = layer_compute_cost(origin, 0)
        distance[1][origin] = initial_cost
        heap_stats.push(heaps[1], (initial_cost, origin), 0)

        def is_candidate(stage, node):
            # stage is the zero-based index of the next layer to place.
            # All Physical nodes are candidates for intermediate layers;
            # the final layer is fixed at destination.
            return stage < layer_count - 1 or node == destination

        def advance_to_next_stage(stage, node, base_cost):
            """Place the next layer at node and update the next-stage heap."""
            check_timeout(start_time)
            if stage >= layer_count or not is_candidate(stage, node):
                return
            next_stage = stage + 1
            candidate_cost = base_cost + layer_compute_cost(node, stage)
            if candidate_cost < distance[next_stage][node]:
                distance[next_stage][node] = candidate_cost
                parent[next_stage][node] = (stage, node, "placement", stage)
                heap_stats.push(
                    heaps[next_stage],
                    (candidate_cost, node),
                    next_stage - 1,
                )

        def pop_global_minimum():
            """Pop the minimum entry among all stage-specific heaps."""
            best_stage = None
            best_item = None
            for stage in range(1, stage_count):
                while heaps[stage]:
                    check_timeout(start_time)
                    cost, node = heaps[stage][0]
                    if cost != distance[stage][node] or settled[stage][node]:
                        heap_stats.pop(heaps[stage], stage - 1)
                        heap_stats.mark_stale()
                    else:
                        break
                if heaps[stage]:
                    item = heaps[stage][0]
                    if best_item is None or item < best_item:
                        best_item = item
                        best_stage = stage
            if best_stage is None:
                return None
            cost, node = heap_stats.pop(heaps[best_stage], best_stage - 1)
            return best_stage, node, cost

        target = None
        while True:
            check_timeout(start_time)
            item = pop_global_minimum()
            if item is None:
                break
            stage, node, current_cost = item
            if settled[stage][node]:
                continue
            settled[stage][node] = True

            # The final layer is already completed in stage L.
            if stage == layer_count and node == destination:
                target = (stage, node)
                break

            # A layer may be placed at the current node without traversing a
            # physical link.  This is the implicit zero-cost imaginary link.
            advance_to_next_stage(stage, node, current_cost)

            if stage >= layer_count:
                continue

            # Continue searching the current physical graph stage.  Whenever
            # a better label for a candidate node is found, MSPR also inserts
            # the corresponding next-stage label immediately.
            for next_node in physical_network.successors(node):
                check_timeout(start_time)
                next_cost = current_cost + communication_cost(node, next_node, stage)
                if next_cost < distance[stage][next_node]:
                    distance[stage][next_node] = next_cost
                    parent[stage][next_node] = (
                        stage,
                        node,
                        "physical",
                        (node, next_node, stage),
                    )
                    heap_stats.push(
                        heaps[stage],
                        (next_cost, next_node),
                        stage - 1,
                    )
                    advance_to_next_stage(stage, next_node, next_cost)

        check_timeout(start_time)
        if target is None:
            raise RuntimeError("MSPR could not find a feasible placement path.")
        end_time = time.perf_counter()
        mspr_elapsed = (end_time - start_time) * 1000
        objective_value = distance[target[0]][target[1]]

        # Reconstruct placements and physical links from the MSPR parent map.
        selected_host_by_layer = {0: origin}
        physical_path_by_stage = {}
        current = target
        while current != (1, origin):
            stage, node = current
            previous = parent[stage][node]
            if previous is None:
                raise RuntimeError("Incomplete MSPR parent information.")
            previous_stage, previous_node, action, payload = previous
            if action == "placement":
                layer = payload
                selected_host_by_layer[layer] = node
            elif action == "physical":
                i, j, physical_stage = payload
                physical_path_by_stage.setdefault(physical_stage, []).append((i, j))
            current = (previous_stage, previous_node)

        selected_hosts = [selected_host_by_layer[layer] for layer in range(layer_count)]
        for stage in physical_path_by_stage:
            physical_path_by_stage[stage].reverse()

    # Output reconstruction compatible with the existing result files.
    physical_node_count = len(physical_nodes)
    imaginary_nodes = {
        layer: physical_node_count + layer for layer in range(layer_count)
    }

    placement = [
        {
            "layer": layer,
            "imaginary_node": imaginary_nodes[layer],
            "physical_node": int(selected_hosts[layer]),
            "value": 1.0,
        }
        for layer in range(layer_count)
    ]

    splitting_points = []
    for layer in range(1, layer_count):
        if selected_hosts[layer] != selected_hosts[layer - 1]:
            splitting_points.append(
                {
                    "layer": layer,
                    "from_physical_node": int(selected_hosts[layer - 1]),
                    "to_physical_node": int(selected_hosts[layer]),
                }
            )

    physical_path = []
    for stage, edges in sorted(physical_path_by_stage.items()):
        for i, j in edges:
            physical_path.append({"i": int(i), "j": int(j), "k": stage, "value": 1.0})

    # Reconstruct augmented-network-style x records.  The MSPR solver itself
    # does not use these imaginary nodes; they are generated only for output.
    service_path = list(physical_path)
    for layer, host in enumerate(selected_hosts):
        service_path.append(
            {
                "i": int(host),
                "j": imaginary_nodes[layer],
                "k": layer,
                "value": 1.0,
            }
        )
        service_path.append(
            {
                "i": imaginary_nodes[layer],
                "j": int(host),
                "k": layer + 1,
                "value": 1.0,
            }
        )

    def stage_path_cost(stage, cost_function):
        return sum(
            cost_function(i, j, stage) for i, j in physical_path_by_stage.get(stage, [])
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
        layer: stage_path_cost(
            layer + 1,
            lambda i, j, _stage: delay[i, j]
            + (0 if is_test else delay.get((j, i), delay[i, j])),
        )
        for layer in range(layer_count - 1)
    }
    activation_delay_by_layer = {
        layer: stage_path_cost(
            layer + 1,
            lambda i, j, _stage: bs * activation_bytes[layer] / rate[i, j],
        )
        for layer in range(layer_count - 1)
    }
    gradient_delay_by_layer = {
        layer: stage_path_cost(
            layer + 1,
            lambda i, j, _stage: bs * gradient_bytes[layer] / rate[i, j],
        )
        for layer in range(layer_count - 1)
    }

    results = {
        "status": "Optimal",
        "objective_value": objective_value,
        "elapsed_time": mspr_elapsed,
        "number_of_physical_nodes": len(physical_nodes),
        "number_of_physical_links": edge_count,
        "number_of_mspr_states": layer_count * len(physical_nodes),
        "number_of_mspr_heaps": layer_count,
        "heap_statistics": heap_stats.as_dict(heaps if layer_count > 1 else []),
        "capacity_constraints": "ignored",
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
    output_filename = f"{resdir}/mspr_bs{bs}_seed{seed}_is_test{is_test}.json"
    check_timeout(start_time)
    with open(output_filename, "w") as f:
        json.dump(
            results,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
            separators=(",", ": "),
        )

    print("Number of Physical nodes:", len(physical_nodes))
    print("Number of directed Physical links:", edge_count)
    print("Number of MSPR states:", layer_count * len(physical_nodes))
    print("Number of MSPR heaps:", layer_count)
    print(
        f"Status: Optimal, Objective value: {objective_value}, "
        f"Elapsed time: {mspr_elapsed} ms"
    )
    # print(results)


if __name__ == "__main__":
    try:
        main()
    except SearchTimeout as error:
        save_timeout_result(
            f"{resdir}/mspr_bs{bs}_seed{seed}_is_test{is_test}.json",
            error.elapsed_seconds,
        )
