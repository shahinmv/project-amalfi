#!/usr/bin/env python3
"""Merge per-laptop node_*.json probe records into one nodes.json array.

Each input file may contain a single node record (object) or an array of them.
Records are de-duplicated by (rpc_host, rpc_port), keeping the last seen.
"""
import argparse, glob, json, sys


def merge_node_objs(objs: list) -> list:
    """Flatten a list of (dict | list-of-dict) into a deduped list of node records."""
    flat = []
    for o in objs:
        if isinstance(o, list):
            flat.extend(o)
        elif isinstance(o, dict):
            flat.append(o)
        else:
            raise ValueError(f"expected object or array, got {type(o).__name__}")
    deduped = {}
    for rec in flat:
        key = (rec.get("rpc_host"), rec.get("rpc_port"))
        deduped[key] = rec
    return list(deduped.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", default=[],
                    help="node json files (default: node_*.json in --dir)")
    ap.add_argument("--dir", default=".", help="dir to glob node_*.json from if no inputs given")
    ap.add_argument("--out", default="nodes.json")
    args = ap.parse_args()
    paths = args.inputs or sorted(glob.glob(f"{args.dir}/node_*.json"))
    if not paths:
        print("no node_*.json files found; run scripts/probe.py on each laptop first",
              file=sys.stderr)
        return 1
    objs = []
    for p in paths:
        with open(p) as f:
            objs.append(json.load(f))
    merged = merge_node_objs(objs)
    with open(args.out, "w") as f:
        json.dump(merged, f, indent=2)
    print(f">> merged {len(paths)} file(s) -> {len(merged)} node(s) in {args.out}")
    for rec in merged:
        print(f'   {rec.get("hostname","?")}  {rec.get("rpc_host")}:{rec.get("rpc_port")}  '
              f'{rec.get("gpu",{}).get("type","?")}  {rec.get("free_ram_gb","?")}GB free')
    return 0


if __name__ == "__main__":
    sys.exit(main())
