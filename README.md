# Project Amalfi — Distributed Inference Cell (v1)

Serve one 25–30B model across several commodity laptops (16 GB RAM each) that no
single laptop can hold, using **llama.cpp RPC pipeline-parallelism** over the LAN.
Proves the "cell" concept and the **throughput-over-latency** insight: per-request
latency stays high, but aggregate tokens/sec rises sharply under concurrency.

This is v1 — the smallest slice that proves the idea on real hardware. The broader
decentralized vision (verification, credit ledger, internet-scale, auto cell-formation)
is captured as future work in the design spec, not built here.

## How it works

Every selected laptop runs a `llama.cpp rpc-server`. One laptop additionally runs
`llama-server`, which connects to all rpc-servers via `--rpc`, distributes the model's
layers with a capability-proportional `--tensor-split`, and exposes an OpenAI-compatible
API + web chat UI. Running rpc-server on *every* node makes the device order equal the
`--rpc` list order, so the tensor-split mapping is deterministic.

## Pipeline

1. `scripts/probe.py` on each node → node record (RAM, cores, GPU, memory bandwidth).
2. Merge records into `nodes.json`; `scripts/plan_split.py` → `fleet.json` + launch commands.
3. `scripts/build_llamacpp.sh <backend>` on each node (pinned source; backend per hardware).
4. `scripts/start_worker.*` on every node; `scripts/start_coordinator.*` on one.
5. Chat via the coordinator web UI (`http://<coordinator>:8080`); benchmark with `bench/run_bench.py`.

## Quickstart (localhost loopback — no laptops needed)

```bash
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m pytest -v            # unit suite (loopback test auto-skips)
./scripts/build_llamacpp.sh cpu            # build llama.cpp once
# drop a small GGUF (e.g. a 1–3B model) into models/, then:
./scripts/loopback_demo.sh <model.gguf>    # 2 workers + coordinator on localhost
```

## Auto-form the cell (flexible / availability-aware)

Instead of hand-listing node IPs, let the coordinator discover whoever's online and form the
best cell automatically. Start workers on any machines you want (`start_worker.*`), then:

```bash
./.venv/bin/python scripts/launch_cell.py --model qwen2.5-7b-q4
```

It reads `config/fleet_registry.json` (candidate workers + capabilities), pings each, keeps
the reachable ones, enumerates their compute devices, computes a **capability-weighted split**
(faster nodes get more layers), writes `dashboard/cell.json`, and launches the coordinator.
Re-run it anytime to re-form the cell for the machines that happen to be up — 2 nodes, 4, or 1.

## Live dashboard (macOS coordinator)

Visualize real per-node data transfer while the cell runs:

```bash
./.venv/bin/python dashboard/server.py --port 8090      # on the coordinator Mac
# open http://<coordinator-ip>:8090
```

A background thread samples the coordinator's per-connection byte counters (`nettop`) and
the page animates particles flowing coordinator↔each worker in proportion to the live
bytes/sec — activations out, computed results back. Nodes light up only while generating.

## Security

RPC mode is **LAN-only and unauthenticated**. Never expose rpc-server ports to the
internet. Open the RPC port (default `50052`) only between laptops on the office LAN.

## Docs

- Office-laptop procedure: [`docs/runbook.md`](docs/runbook.md)
- Design spec: [`docs/superpowers/specs/2026-07-23-distributed-inference-cell-design.md`](docs/superpowers/specs/2026-07-23-distributed-inference-cell-design.md)
- Implementation plan: [`docs/superpowers/plans/2026-07-23-distributed-inference-cell.md`](docs/superpowers/plans/2026-07-23-distributed-inference-cell.md)
- Results: [`docs/results.md`](docs/results.md) (filled during acceptance on the laptops)

## Repo layout

```
scripts/   probe.py · plan_split.py · build_flags.py · build_llamacpp.{sh,ps1}
           start_worker.{sh,ps1} · start_coordinator.{sh,ps1} · healthcheck.py · loopback_demo.sh
bench/     run_bench.py · report.py · prompts.txt
config/    models.json · fleet.example.json · llamacpp.pin
tests/     unit tests + loopback integration test
docs/      runbook.md · results.md · superpowers/{specs,plans}/
```
