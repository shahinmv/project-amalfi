#!/usr/bin/env python3
"""Ping every rpc-server in a fleet.json and report cell health."""
import argparse, json, socket, sys


def parse_rpc(rpc: str) -> list:
    out = []
    for part in rpc.split(","):
        host, port = part.rsplit(":", 1)
        out.append((host, int(port)))
    return out


def check_endpoint(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def run_healthcheck(fleet: dict, timeout: float = 3.0) -> dict:
    nodes = []
    for host, port in parse_rpc(fleet["rpc"]):
        nodes.append({"host": host, "port": port,
                      "up": check_endpoint(host, port, timeout)})
    return {"nodes": nodes, "all_up": all(n["up"] for n in nodes)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleet", default="fleet.json")
    ap.add_argument("--timeout", type=float, default=3.0)
    args = ap.parse_args()
    with open(args.fleet) as f:
        fleet = json.load(f)
    res = run_healthcheck(fleet, args.timeout)
    for n in res["nodes"]:
        print(f'{"UP  " if n["up"] else "DOWN"}  {n["host"]}:{n["port"]}')
    if not res["all_up"]:
        print("\nCell degraded. Re-run the planner over surviving nodes:\n"
              "  python scripts/plan_split.py --nodes nodes.json --model <smaller-if-needed> --out fleet.json\n"
              "then restart workers + coordinator.")
        return 1
    print("\nAll nodes up.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
