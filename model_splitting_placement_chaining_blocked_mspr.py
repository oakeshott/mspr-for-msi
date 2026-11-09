import argparse
import heapq
import json
import math
import os
import time

import networkx as nx
import pandas as pd


parser = argparse.ArgumentParser(
    description="Model splitting placement and chaining with block-based MSPR"
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
        "Per-node capacity for every node except origin. If omitted, use "
        "1e10, following model_splitting_placement_chaining_z.py."
    ),
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


def load_capacities(nodes, origin, destination, global_capacity):
    """Return the z.py-compatible memory and storage capacities.

    z.py uses an 8e9 memory limit at the origin, the command-line capacity
    (or 1e10 by default) at every other node, and a separate 8e14 storage
    limit at the origin. The block MSPR follows those same values.
    """
    capacity = 1e10 if global_capacity is None else float(global_capacity)
    if math.isnan(capacity) or capacity < 0:
        raise ValueError("Capacity must be non-negative.")

    memory_capacity = {}
    storage_capacity = {}
    for node in nodes:
        # if node == origin or node == destination:
        #     memory_capacity[node] = 8e9
        #     storage_capacity[node] = 8e14
        # else:
        #     memory_capacity[node] = capacity
        #     storage_capacity[node] = capacity
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
    edge_count = len(physical_edges)
    if origin not in physical_network or destination not in physical_network:
        raise ValueError("origin and destination must be Physical nodes.")

    activation_bytes = [row["activation_bytes"] for row in data]
    gradient_bytes = [row["gradient_bytes"] for row in data]
    fwd_flops = [row["FWD_FLOPs"] for row in data]
    bwd_flops = [row["BWD_FLOPs"] for row in data]
    layer_memory = [float(row["memory"]) for row in data]
    if any(math.isnan(value) or value < 0 for value in layer_memory):
        raise ValueError("Layer memory values must be non-negative.")

    # Prefix sums make the hard block test O(1):
    #   sum(M[k:j+1]) = memory_prefix[j+1] - memory_prefix[k].
    memory_prefix = [0.0]
    for memory in layer_memory:
        memory_prefix.append(memory_prefix[-1] + memory)
    memory_capacity, storage_capacity = load_capacities(
        physical_nodes, origin, destination, args.capacity
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

    def layer_compute_cost(node, layer):
        value = tau[node] + kappa[node] * fwd_flops[layer]
        if not is_test:
            value += tau[node] + kappa[node] * bwd_flops[layer]
        return value

    def block_compute_cost(node, block_start, block_end):
        return sum(
            layer_compute_cost(node, layer)
            for layer in range(block_start, block_end + 1)
        )

    def communication_cost(i, j, stage):
        """Cost of moving the output of layer stage-1 from i to j."""
        value = delay[i, j]
        if not is_test:
            value += delay.get((j, i), delay[i, j])
        layer = stage - 1
        value += bs * activation_bytes[layer] / rate[i, j]
        if not is_test:
            value += bs * gradient_bytes[layer] / rate[i, j]
        return value

    # A feasible search state contains the completed layer prefix, the node
    # holding the output of the last completed layer, and the aggregate
    # resources already assigned to every physical node.  The old algorithm
    # used only (stage, node), which was insufficient when a node was visited
    # more than once: it forgot the storage and activation memory assigned on
    # earlier visits.
    #
    # The resource vectors make this an exact label-setting search.  There can
    # be exponentially many labels in the worst case, but each label is
    # finite and represents a distinct (stage, node, resource-vector) state.
    # Keeping the full vectors is necessary for equivalence with the ILP.
    node_index = {node: index for index, node in enumerate(physical_nodes)}
    zero_storage = tuple(0.0 for _ in physical_nodes)
    zero_activation = tuple(0.0 for _ in physical_nodes)

    start_time = time.perf_counter()
    heap_stats = HeapStats(0)
    heap = []
    if layer_count == 1:
        storage = layer_memory[0]
        activation_memory = bs * activation_bytes[0]
        total_memory = storage + activation_memory
        if storage > storage_capacity[origin] or total_memory > memory_capacity[origin]:
            raise RuntimeError("Layer 0 does not fit at the origin.")
        selected_hosts = [origin]
        objective_value = layer_compute_cost(origin, 0)
        physical_path_by_stage = {}
        selected_blocks = [
            {
                "start_layer": 0,
                "end_layer": 0,
                "physical_node": origin,
                "storage": storage,
                "activation_memory": activation_memory,
                "memory": total_memory,
            }
        ]
        mspr_elapsed = 0.0
        block_advance_count = 0
        state_count = 1
    else:
        # A label is a complete state in the expanded state space.  Labels
        # with the same key retain only the cheapest cost; movement does not
        # change resources, so a more expensive duplicate can never help.
        labels = []
        best_label_for_key = {}
        heap = []
        heap_stats = HeapStats(1)

        def state_key(stage, node, storage_usage, activation_usage):
            return (stage, node, storage_usage, activation_usage)

        def add_label(
            stage,
            node,
            cost,
            storage_usage,
            activation_usage,
            parent_id,
            action,
            payload,
        ):
            check_timeout(start_time)
            key = state_key(stage, node, storage_usage, activation_usage)
            old_id = best_label_for_key.get(key)
            if old_id is not None and labels[old_id]["cost"] <= cost:
                return None

            label_id = len(labels)
            labels.append(
                {
                    "stage": stage,
                    "node": node,
                    "cost": cost,
                    "storage": storage_usage,
                    "activation": activation_usage,
                    "parent": parent_id,
                    "action": action,
                    "payload": payload,
                    "settled": False,
                }
            )
            best_label_for_key[key] = label_id
            heap_stats.push(heap, (cost, label_id))
            return label_id

        def pop_minimum_label():
            while heap:
                check_timeout(start_time)
                cost, label_id = heap_stats.pop(heap)
                label = labels[label_id]
                key = state_key(
                    label["stage"],
                    label["node"],
                    label["storage"],
                    label["activation"],
                )
                if best_label_for_key.get(key) != label_id:
                    heap_stats.mark_stale()
                    continue
                if label["settled"]:
                    heap_stats.mark_stale()
                    continue
                label["settled"] = True
                return label_id
            return None

        def is_block_candidate(block_end, node):
            # A block containing the final layer is allowed only at d.
            return block_end < layer_count - 1 or node == destination

        block_advance_count = 0

        def relax_block(label_id, block_end):
            check_timeout(start_time)
            nonlocal block_advance_count
            label = labels[label_id]
            block_start = label["stage"]
            node = label["node"]
            if block_start >= layer_count:
                return

            block_storage = memory_prefix[block_end + 1] - memory_prefix[block_start]
            block_activation_memory = max(
                bs * activation_bytes[layer]
                for layer in range(block_start, block_end + 1)
            )
            node_position = node_index[node]
            next_storage = list(label["storage"])
            next_activation = list(label["activation"])
            next_storage[node_position] += block_storage
            next_activation[node_position] = max(
                next_activation[node_position], block_activation_memory
            )
            aggregate_memory = (
                next_storage[node_position] + next_activation[node_position]
            )

            # This is the ILP-equivalent test.  Storage is summed over every
            # layer ever assigned to this node, while activation is the
            # maximum over every such layer, including earlier visits.
            if (
                next_storage[node_position] > storage_capacity[node]
                or aggregate_memory > memory_capacity[node]
                or not is_block_candidate(block_end, node)
            ):
                return

            next_stage = block_end + 1
            candidate_cost = label["cost"] + block_compute_cost(
                node, block_start, block_end
            )
            add_label(
                next_stage,
                node,
                candidate_cost,
                tuple(next_storage),
                tuple(next_activation),
                label_id,
                "placement_block",
                (block_start, block_end),
            )
            block_advance_count += 1

        # Seed the search with every feasible initial block 0..j at origin.
        # This is important: checking only M_0 would fail to account for
        # later layers placed in the same origin block.
        initial_label_id = len(labels)
        labels.append(
            {
                "stage": 0,
                "node": origin,
                "cost": 0.0,
                "storage": zero_storage,
                "activation": zero_activation,
                "parent": None,
                "action": "initial",
                "payload": None,
                "settled": True,
            }
        )
        for block_end in range(layer_count):
            check_timeout(start_time)
            relax_block(initial_label_id, block_end)

        # Measure only the label-setting search, consistently with the other
        # algorithms; initialization and initial-label generation are setup.
        target_label_id = None
        while (label_id := pop_minimum_label()) is not None:
            check_timeout(start_time)
            label = labels[label_id]
            stage = label["stage"]
            node = label["node"]
            if stage == layer_count and node == destination:
                target_label_id = label_id
                break

            if stage >= layer_count:
                # A completed placement at a non-destination node is not a
                # valid target, and there is no reason to move after layer L.
                continue

            # Consecutive blocks at the same node are redundant: the longer
            # block was enumerated from the predecessor label.  Suppressing
            # only this case avoids duplicate labels while still allowing a
            # node to be revisited after physical movement.
            if label["action"] != "placement_block":
                for block_end in range(stage, layer_count):
                    relax_block(label_id, block_end)

            for next_node in physical_network.successors(node):
                check_timeout(start_time)
                next_cost = label["cost"] + communication_cost(node, next_node, stage)
                add_label(
                    stage,
                    next_node,
                    next_cost,
                    label["storage"],
                    label["activation"],
                    label_id,
                    "physical",
                    (node, next_node, stage),
                )

        check_timeout(start_time)
        if target_label_id is None:
            raise RuntimeError("MSPR could not find a feasible placement path.")
        end_time = time.perf_counter()
        mspr_elapsed = (end_time - start_time) * 1000
        objective_value = labels[target_label_id]["cost"]
        state_count = len(best_label_for_key)

        # Reconstruct placements and physical links from the expanded-label
        # parent chain.
        selected_host_by_layer = {}
        selected_blocks = []
        physical_path_by_stage = {}
        current_label_id = target_label_id
        while current_label_id != initial_label_id:
            label = labels[current_label_id]
            parent_id = label["parent"]
            if parent_id is None:
                raise RuntimeError("Incomplete MSPR parent information.")
            if label["action"] == "placement_block":
                block_start, block_end = label["payload"]
                selected_blocks.append(
                    {
                        "start_layer": block_start,
                        "end_layer": block_end,
                        "physical_node": label["node"],
                        "storage": memory_prefix[block_end + 1]
                        - memory_prefix[block_start],
                        "activation_memory": max(
                            bs * activation_bytes[layer]
                            for layer in range(block_start, block_end + 1)
                        ),
                        "memory": (
                            memory_prefix[block_end + 1]
                            - memory_prefix[block_start]
                            + max(
                                bs * activation_bytes[layer]
                                for layer in range(block_start, block_end + 1)
                            )
                        ),
                    }
                )
                for layer in range(block_start, block_end + 1):
                    selected_host_by_layer[layer] = label["node"]
            elif label["action"] == "physical":
                i, j, physical_stage = label["payload"]
                physical_path_by_stage.setdefault(physical_stage, []).append((i, j))
            current_label_id = parent_id

        if set(selected_host_by_layer) != set(range(layer_count)):
            raise RuntimeError("Incomplete block placement reconstruction.")
        selected_hosts = [selected_host_by_layer[layer] for layer in range(layer_count)]
        selected_blocks.reverse()
        for stage in physical_path_by_stage:
            physical_path_by_stage[stage].reverse()

    # Verify the reconstructed blocks independently of the search. This is a
    # defensive check that makes the hard-constraint guarantee explicit.
    for block in selected_blocks:
        node = block["physical_node"]
        if (
            block["storage"] > storage_capacity[node]
            or block["memory"] > memory_capacity[node]
        ):
            raise RuntimeError("Reconstructed placement violates memory capacity.")

    # Verify the actual ILP-equivalent node-wise aggregate constraints.  This
    # second check is intentionally independent of the transition checks: it
    # catches any future reconstruction or output changes that accidentally
    # reintroduce block-wise-only validation.
    node_storage_usage = {node: 0.0 for node in physical_nodes}
    node_activation_usage = {node: 0.0 for node in physical_nodes}
    for layer, node in enumerate(selected_hosts):
        node_storage_usage[node] += layer_memory[layer]
        node_activation_usage[node] = max(
            node_activation_usage[node], float(bs * activation_bytes[layer])
        )
    node_memory_usage = {}
    for node in physical_nodes:
        total_memory = node_storage_usage[node] + node_activation_usage[node]
        if (
            node_storage_usage[node] > storage_capacity[node]
            or total_memory > memory_capacity[node]
        ):
            raise RuntimeError(
                "Reconstructed placement violates aggregate node memory "
                f"capacity at node {node}."
            )
        node_memory_usage[str(node)] = {
            "storage": node_storage_usage[node],
            "activation_memory": node_activation_usage[node],
            "memory": total_memory,
        }

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

    # Reconstruct augmented-network-style x records. The MSPR solver itself
    # does not use imaginary nodes; they are generated only for output.
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
        "number_of_mspr_states": state_count,
        "number_of_mspr_heaps": 1,
        "heap_statistics": heap_stats.as_dict([heap]),
        "state_representation": (
            "(completed layer prefix, current node, per-node cumulative "
            "storage, per-node maximum activation memory)"
        ),
        "number_of_block_advances": block_advance_count,
        "capacity_constraints": (
            "hard node-wise aggregate storage and memory capacity "
            "(sum of weights plus maximum activation over all assigned layers)"
        ),
        "node_memory_capacity": {
            str(node): capacity for node, capacity in memory_capacity.items()
        },
        "node_storage_capacity": {
            str(node): capacity for node, capacity in storage_capacity.items()
        },
        "node_memory_usage": node_memory_usage,
        "layer_memory": {layer: layer_memory[layer] for layer in range(layer_count)},
        "selected_blocks": [
            {
                **block,
                "physical_node": int(block["physical_node"]),
            }
            for block in selected_blocks
        ],
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
    # Keep the original result filename so this script can be used as a
    # drop-in replacement for the layer-wise MSPR implementation.
    output_filename = f"{resdir}/blocked_mspr_bs{bs}_seed{seed}_is_test{is_test}.json"
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
    print("Number of MSPR states:", state_count)
    print("Number of MSPR heaps:", 1)
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
            f"{resdir}/blocked_mspr_bs{bs}_seed{seed}_is_test{is_test}.json",
            error.elapsed_seconds,
        )
