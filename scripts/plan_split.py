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


def build_launch_commands(selected: list, model: dict,
                          rpc_port: int = 50052, api_port: int = 8080) -> dict:
    rpc = ",".join(f'{n["rpc_host"]}:{rpc_port}' for n in selected)
    split = compute_tensor_split(selected)
    split_str = ",".join(str(x) for x in split)
    coordinator = (
        f'llama-server --model models/{model["gguf"]} '
        f'--rpc {rpc} --tensor-split {split_str} --n-gpu-layers 999 '
        f'--ctx-size {model.get("ctx_size", 4096)} --host 0.0.0.0 --port {api_port}')
    return {
        "rpc": rpc,
        "tensor_split": split,
        "worker_cmd": f"rpc-server --host 0.0.0.0 --port {rpc_port}",
        "coordinator_cmd": coordinator,
        "coordinator_host": selected[0]["rpc_host"],
    }


def plan(nodes: list, model_key: str, catalog: dict, rpc_port: int = 50052) -> dict:
    if model_key not in catalog:
        raise ValueError(f"Unknown model '{model_key}'. Known: {list(catalog)}")
    model = catalog[model_key]
    required = estimate_required_gb(model)
    selected = select_nodes(nodes, required)
    cmds = build_launch_commands(selected, model, rpc_port=rpc_port)
    return {
        "model": model_key, "model_size_gb": model["size_gb"],
        "required_gb": required, "rpc_port": rpc_port,
        "selected_hosts": [n["rpc_host"] for n in selected],
        **cmds,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", required=True, help="path to merged nodes.json (list)")
    ap.add_argument("--model", required=True, help="model key from config/models.json")
    ap.add_argument("--catalog", default="config/models.json")
    ap.add_argument("--rpc-port", type=int, default=50052)
    ap.add_argument("--out", default="fleet.json")
    args = ap.parse_args()
    with open(args.nodes) as f:
        nodes = json.load(f)
    with open(args.catalog) as f:
        catalog = json.load(f)
    fleet = plan(nodes, args.model, catalog, rpc_port=args.rpc_port)
    with open(args.out, "w") as f:
        json.dump(fleet, f, indent=2)
    print(json.dumps(fleet, indent=2))
    print(f"\n# On each worker node:\n{fleet['worker_cmd']}")
    print(f"\n# On the coordinator ({fleet['coordinator_host']}):\n{fleet['coordinator_cmd']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
