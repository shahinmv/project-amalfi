#!/usr/bin/env python3
"""Self-healing cell supervisor — auto-forms and re-forms the cell as nodes join/drop.

Watches the fleet registry for reachable workers and the coordinator's health. When the
online-worker set changes (a node dropped or a new one joined) or the coordinator dies, it
re-runs launch_cell.py to re-form the cell over whoever is up.

This is the pragmatic 'reroute-on-drop' achievable on the llama.cpp RPC stack: recover by
RE-FORMING the cell (the engine has no per-block hot-failover). A Petals-grade network would
hot-swap individual layer-blocks instead — see docs/productization.md.

Run:  ./.venv/bin/python scripts/supervisor.py [--model qwen2.5-7b-q4] [--interval 10]
"""
import argparse, json, pathlib, socket, subprocess, sys, time, urllib.request

ROOT = pathlib.Path(__file__).parent.parent


def reachable_set(registry, timeout=1.5):
    up = set()
    for n in registry:
        try:
            socket.create_connection((n["host"], n.get("port", 50052)), timeout=timeout).close()
            up.add(n["host"])
        except OSError:
            pass
    return up


def coord_healthy(api, timeout=2.0):
    try:
        urllib.request.urlopen(api + "/health", timeout=timeout)
        return True
    except Exception:
        return False


def needs_reform(active, reachable, healthy, loading, min_nodes=1):
    """Pure decision — should we (re-)form the cell now?

    active:   set of hosts in the currently-formed cell
    reachable: set of hosts online right now
    healthy:  is the coordinator answering /health
    loading:  are we within the post-launch grace period (model still streaming)
    """
    if len(reachable) < min_nodes:
        return False                      # nothing to run on
    if not active:
        return True                       # nothing formed yet -> form it
    if reachable != active:
        return True                       # a node joined or dropped -> re-form
    if not healthy and not loading:
        return True                       # coordinator died (and not just loading)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5-7b-q4")
    ap.add_argument("--registry", default=str(ROOT / "config" / "fleet_registry.json"))
    ap.add_argument("--api", default="http://127.0.0.1:8080")
    ap.add_argument("--interval", type=float, default=10.0)
    ap.add_argument("--debounce", type=int, default=2,
                    help="consecutive stable scans required before acting on a membership change")
    ap.add_argument("--load-grace", type=float, default=240.0,
                    help="seconds after a (re)form to ignore an unhealthy coordinator (it's loading)")
    args = ap.parse_args()

    registry = json.load(open(args.registry))
    active, last_seen, stable, last_reform = set(), None, 0, 0.0
    print(f">> supervisor watching {len(registry)} candidate node(s); model={args.model}; "
          f"every {args.interval}s. Auto-forms when workers appear, re-forms on drop/join.")
    while True:
        reach = reachable_set(registry)
        healthy = coord_healthy(args.api)
        loading = (time.time() - last_reform) < args.load_grace
        stable = stable + 1 if reach == last_seen else 0
        last_seen = reach

        change = needs_reform(active, reach, healthy, loading)
        membership_changed = bool(active) and reach != active
        # act immediately for first-form / coordinator-death; debounce membership flaps
        ready = (not active) or (not healthy and not loading) or (stable >= args.debounce)
        if change and ready:
            names = [n["name"] for n in registry if n["host"] in reach]
            reason = ("initial form" if not active else
                      "coordinator down" if not healthy else "membership changed")
            print(f">> [{time.strftime('%H:%M:%S')}] {reason}: forming over {len(reach)} node(s): {names}")
            r = subprocess.run([sys.executable, str(ROOT / "scripts" / "launch_cell.py"),
                                "--model", args.model])
            if r.returncode == 0:
                active, last_reform = set(reach), time.time()
                print(f">> cell (re)formed over {sorted(active)}")
            else:
                print(">> launch_cell failed; retrying next cycle")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
