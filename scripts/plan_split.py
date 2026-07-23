#!/usr/bin/env python3
"""Amalfi split planner: nodes.json -> capability-proportional layer plan."""
import argparse, json, sys

GPU_BONUS = {"cuda": 3.0, "metal": 2.0, "vulkan": 0.5, "none": 0.0}


def score_node(node: dict) -> float:
    bw = node.get("mem_bandwidth_gbps") or 1.0
    bonus = GPU_BONUS.get(node.get("gpu", {}).get("type", "none"), 0.0)
    return round(bw * (1.0 + bonus), 2)


def node_capacity_gb(node: dict) -> float:
    return round(node.get("free_ram_gb", 0.0) + node.get("gpu", {}).get("vram_gb", 0.0), 2)


def estimate_required_gb(model: dict, headroom_gb: float = 2.0) -> float:
    return round(model["size_gb"] + model.get("kv_per_seq_gb", 0.5) + headroom_gb, 2)


def select_nodes(nodes: list, required_gb: float) -> list:
    ranked = sorted(nodes, key=lambda n: (score_node(n), node_capacity_gb(n)), reverse=True)
    chosen, cap = [], 0.0
    for n in ranked:
        chosen.append(n)
        cap += node_capacity_gb(n)
        if cap >= required_gb:
            return chosen
    raise ValueError(
        f"Fleet capacity {cap:.1f} GB < required {required_gb:.1f} GB. "
        f"Add nodes or choose a smaller model.")


def compute_tensor_split(selected: list) -> list:
    scores = [score_node(n) for n in selected]
    total = sum(scores) or 1.0
    split = [round(s / total, 3) for s in scores]
    split[-1] = round(split[-1] + (1.0 - sum(split)), 3)
    return split


def capped_budget_gb(node: dict, cap_gb: float, overhead_gb: float) -> float:
    """GB of model weights a node may hold under an ABSOLUTE per-node RAM cap.

    Limited by both the cap and the node's actual free memory, minus overhead
    (KV cache + compute buffers + the rpc-server process)."""
    return round(max(0.0, min(cap_gb, node_capacity_gb(node)) - overhead_gb), 3)


def dynamic_budget_gb(node: dict, fraction: float, overhead_gb: float) -> float:
    """GB of weights a node may hold under a DYNAMIC per-node cap.

    Footprint (weights + overhead) is capped at `fraction` of the node's total
    RAM+VRAM (be a good citizen — leave the rest for the user), and additionally
    never exceeds what is currently free (avoid OOM)."""
    total = node.get("total_ram_gb", node.get("free_ram_gb", 0.0)) + node.get("gpu", {}).get("vram_gb", 0.0)
    free = node_capacity_gb(node)  # free_ram + vram
    footprint_ceiling = min(fraction * total, free)
    return round(max(0.0, footprint_ceiling - overhead_gb), 3)


def _select_by_budget(nodes: list, model_size_gb: float, budget_fn, why: str) -> list:
    """Pick the fewest strongest nodes whose per-node weight budgets sum to the model."""
    ranked = sorted(nodes, key=lambda n: (budget_fn(n), score_node(n)), reverse=True)
    chosen, total = [], 0.0
    for n in ranked:
        b = budget_fn(n)
        if b <= 0:
            continue
        chosen.append(n)
        total += b
        if total >= model_size_gb:
            return chosen
    raise ValueError(
        f"{why}: fleet holds only {total:.1f} GB of weights < {model_size_gb} GB needed. "
        f"Add laptops, loosen the cap, or choose a smaller model.")


def _split_by_budget(selected: list, budget_fn) -> list:
    """Split proportional to each node's budget (guarantees no node exceeds it)."""
    budgets = [budget_fn(n) for n in selected]
    total = sum(budgets) or 1.0
    split = [round(b / total, 3) for b in budgets]
    split[-1] = round(split[-1] + (1.0 - sum(split)), 3)
    return split


def select_nodes_capped(nodes: list, model_size_gb: float,
                        cap_gb: float, overhead_gb: float = 1.0) -> list:
    return _select_by_budget(nodes, model_size_gb,
                             lambda n: capped_budget_gb(n, cap_gb, overhead_gb),
                             f"With a {cap_gb} GB/node cap")


def compute_tensor_split_capped(selected: list, cap_gb: float,
                                overhead_gb: float = 1.0) -> list:
    return _split_by_budget(selected, lambda n: capped_budget_gb(n, cap_gb, overhead_gb))


def build_launch_commands(selected: list, model: dict, rpc_port: int = 50052,
                          api_port: int = 8080, split: list = None) -> dict:
    rpc = ",".join(f'{n["rpc_host"]}:{rpc_port}' for n in selected)
    if split is None:
        split = compute_tensor_split(selected)
    split_str = ",".join(str(x) for x in split)
    coordinator = (
        f'llama-server --model models/{model["gguf"]} '
        f'--rpc {rpc} --tensor-split {split_str} --n-gpu-layers 999 '
        f'--ctx-size {model.get("ctx_size", 4096)} --host 0.0.0.0 --port {api_port}')
    return {
        "rpc": rpc,
        "tensor_split": split,
        "worker_cmd": f"ggml-rpc-server --host 0.0.0.0 --port {rpc_port}",
        "coordinator_cmd": coordinator,
        "coordinator_host": selected[0]["rpc_host"],
    }


def plan(nodes: list, model_key: str, catalog: dict, rpc_port: int = 50052,
         max_ram_gb: float = None, ram_fraction: float = 0.5,
         ram_overhead_gb: float = 1.0) -> dict:
    """Plan the cell. Cap policy (precedence): an absolute --max-ram-gb wins; else a
    dynamic per-node cap at `ram_fraction` of each laptop's RAM/VRAM (default 0.5);
    pass ram_fraction=None to disable capping and split purely by compute score."""
    if model_key not in catalog:
        raise ValueError(f"Unknown model '{model_key}'. Known: {list(catalog)}")
    model = catalog[model_key]
    size = model["size_gb"]
    if max_ram_gb:
        policy = f"max_ram_gb={max_ram_gb}"
        budget_fn = lambda n: capped_budget_gb(n, max_ram_gb, ram_overhead_gb)
        selected = _select_by_budget(nodes, size, budget_fn, f"With a {max_ram_gb} GB/node cap")
        split = _split_by_budget(selected, budget_fn)
        required = size
    elif ram_fraction:
        policy = f"ram_fraction={ram_fraction}"
        budget_fn = lambda n: dynamic_budget_gb(n, ram_fraction, ram_overhead_gb)
        selected = _select_by_budget(nodes, size, budget_fn,
                                     f"Using {int(ram_fraction*100)}% of each laptop's RAM")
        split = _split_by_budget(selected, budget_fn)
        required = size
    else:
        policy = "uncapped (compute-score split)"
        required = estimate_required_gb(model)
        selected = select_nodes(nodes, required)
        split = compute_tensor_split(selected)
    cmds = build_launch_commands(selected, model, rpc_port=rpc_port, split=split)
    # per-node RAM estimate the user can eyeball
    est = [{"host": n["rpc_host"],
            "weights_gb": round(split[i] * size, 2),
            "total_gb": round(split[i] * size + ram_overhead_gb, 2)}
           for i, n in enumerate(selected)]
    return {
        "model": model_key, "model_size_gb": size,
        "required_gb": required, "rpc_port": rpc_port,
        "cap_policy": policy, "ram_overhead_gb": ram_overhead_gb,
        "selected_hosts": [n["rpc_host"] for n in selected],
        "est_ram_per_node": est,
        **cmds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="path to merged nodes.json (list)")
    ap.add_argument("--model", required=True, help="model key from config/models.json")
    ap.add_argument("--catalog", default="config/models.json")
    ap.add_argument("--rpc-port", type=int, default=50052)
    ap.add_argument("--max-ram-gb", type=float, default=None,
                    help="ABSOLUTE cap on RAM per laptop (weights+overhead). Overrides "
                         "--ram-fraction. Spreads the model across enough nodes to stay under it.")
    ap.add_argument("--ram-fraction", type=float, default=0.5,
                    help="DYNAMIC cap: use at most this fraction of each laptop's RAM/VRAM "
                         "(default 0.5 = half), bounded by free memory. Set 0 to disable capping.")
    ap.add_argument("--ram-overhead-gb", type=float, default=1.0,
                    help="reserved per-node non-weight RAM (KV cache + buffers + process)")
    ap.add_argument("--out", default="fleet.json")
    args = ap.parse_args()
    with open(args.nodes) as f:
        nodes = json.load(f)
    with open(args.catalog) as f:
        catalog = json.load(f)
    fleet = plan(nodes, args.model, catalog, rpc_port=args.rpc_port,
                 max_ram_gb=args.max_ram_gb,
                 ram_fraction=(args.ram_fraction or None),
                 ram_overhead_gb=args.ram_overhead_gb)
    with open(args.out, "w") as f:
        json.dump(fleet, f, indent=2)
    print(json.dumps(fleet, indent=2))
    print(f"\n# Cap policy: {fleet['cap_policy']}")
    print("# Estimated RAM per laptop:")
    for e in fleet["est_ram_per_node"]:
        print(f"#   {e['host']}: ~{e['total_gb']} GB  ({e['weights_gb']} GB weights + "
              f"{fleet['ram_overhead_gb']} GB overhead)")
    print(f"\n# On each worker node:\n{fleet['worker_cmd']}")
    print(f"\n# On the coordinator ({fleet['coordinator_host']}):\n{fleet['coordinator_cmd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
