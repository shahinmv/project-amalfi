#!/usr/bin/env python3
"""Auto-form the cell from whichever registered workers are ONLINE, then launch the coordinator.

Flexible by design — re-run it anytime and it adapts to node availability:
  1. reads config/fleet_registry.json (candidate workers + their capabilities)
  2. pings each; keeps the reachable ones
  3. enumerates their real compute devices (llama-server --list-devices)
  4. computes a capability-weighted --tensor-split (faster nodes get more layers)
  5. writes dashboard/cell.json (so the dashboard reflects the live split)
  6. launches llama-server on this Mac (coordinator) pinned to those devices

Usage: ./.venv/bin/python scripts/launch_cell.py [--model qwen2.5-7b-q4] [--port 8080]
"""
import argparse, importlib.util, json, pathlib, re, socket, subprocess, sys

ROOT = pathlib.Path(__file__).parent.parent
BIN = ROOT / "vendor" / "llama.cpp" / "build" / "bin"


def _load(mod):
    spec = importlib.util.spec_from_file_location(mod, ROOT / "scripts" / (mod + ".py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


ps = _load("plan_split")


def _reachable(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def _score(n):
    return ps.score_node({"mem_bandwidth_gbps": n.get("bw", 1),
                          "gpu": {"type": n.get("backend", "cpu")}})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--registry", default=str(ROOT / "config" / "fleet_registry.json"))
    ap.add_argument("--catalog", default=str(ROOT / "config" / "models.json"))
    ap.add_argument("--model", default="qwen2.5-7b-q4")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--min-mib", type=int, default=100, help="ignore devices smaller than this")
    args = ap.parse_args()

    registry = json.load(open(args.registry))
    catalog = json.load(open(args.catalog))
    if args.model not in catalog:
        print(f"unknown model '{args.model}'. known: {list(catalog)}"); return 2
    model = catalog[args.model]

    up = [n for n in registry if _reachable(n["host"], n.get("port", 50052))]
    if not up:
        print("No workers reachable. Start workers (scripts/start_worker.*) first."); return 1
    print(f">> online workers: {', '.join(n['name'] for n in up)}")

    rpc_all = ",".join(f'{n["host"]}:{n.get("port", 50052)}' for n in up)
    try:
        r = subprocess.run([str(BIN / "llama-server"), "--rpc", rpc_all, "--list-devices"],
                           capture_output=True, text=True, timeout=45)
    except subprocess.TimeoutExpired:
        print("!! Device enumeration hung — a worker answers TCP but is unresponsive at the RPC\n"
              "   layer (stale/stuck worker). Restart the worker on these node(s) and retry:")
        for n in up:
            print(f"     - {n['name']} ({n['host']})")
        return 1
    best = {}  # "host:port" -> (device_name, mib)
    for line in (r.stdout + r.stderr).splitlines():
        m = re.search(r"(RPC\d+):\s+(\d+\.\d+\.\d+\.\d+:\d+)\s+\((\d+)\s*MiB", line)
        if not m:
            continue
        dev, ep, mib = m.group(1), m.group(2), int(m.group(3))
        if mib < args.min_mib:
            continue
        if ep not in best or mib > best[ep][1]:
            best[ep] = (dev, mib)

    sel = [n for n in up if f'{n["host"]}:{n.get("port", 50052)}' in best]
    if not sel:
        print("No usable compute devices enumerated from the online workers."); return 1

    scores = [_score(n) for n in sel]
    total = sum(scores) or 1.0
    split = [round(s / total, 3) for s in scores]
    split[-1] = round(split[-1] + (1.0 - sum(split)), 3)
    devices = [best[f'{n["host"]}:{n.get("port", 50052)}'][0] for n in sel]
    rpc_sel = ",".join(f'{n["host"]}:{n.get("port", 50052)}' for n in sel)

    pooled = sum(n.get("ram_gb", 0) for n in sel)
    print(f">> forming cell: {args.model} ({model['size_gb']} GB) across {len(sel)} node(s); "
          f"pooled RAM {pooled:.1f} GB")
    for n, s in zip(sel, split):
        print(f"   {n['name']:24} {int(round(s*100)):3d}%  (~{s*model['size_gb']:.1f} GB)  [{devices[sel.index(n)]}]")
    if pooled and model["size_gb"] > pooled:
        print(f"!! WARNING: model {model['size_gb']} GB > pooled RAM {pooled:.1f} GB — may not fit.")

    # dashboard/cell.json so the UI reflects the live split
    nodes_meta = {n["host"]: {"name": n["name"], "backend": n.get("backend", "cpu"),
                              "pct": int(round(s * 100)), "cores": n.get("cores"),
                              "ram_gb": n.get("ram_gb"), "bw": n.get("bw")}
                  for n, s in zip(sel, split)}
    disp = model.get("gguf", args.model).rsplit(".gguf", 1)[0]
    json.dump({"model": disp, "quant": "Q4_K_M", "model_size_gb": model["size_gb"],
               "ctx_size": model.get("ctx_size", 4096), "nodes": nodes_meta},
              open(ROOT / "dashboard" / "cell.json", "w"), indent=2)

    subprocess.run(["pkill", "-f", "llama-server --model"], capture_output=True)
    cmd = [str(BIN / "llama-server"), "--model", f"models/{model['gguf']}",
           "--rpc", rpc_sel, "--device", ",".join(devices),
           "--tensor-split", ",".join(str(x) for x in split), "-ngl", "999",
           "--ctx-size", str(model.get("ctx_size", 4096)), "--host", "0.0.0.0",
           "--port", str(args.port)]
    logf = open(ROOT / "coordinator.log", "w")
    p = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, stderr=subprocess.STDOUT)
    print(f">> coordinator pid {p.pid} launching (log: coordinator.log). "
          f"Watch http://<this-mac>:{args.port} — loads in ~1-4 min as weights stream.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
