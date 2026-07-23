# Project Amalfi — Distributed Inference Cell (v1) — Design Spec

**Date:** 2026-07-23
**Status:** Approved (design), pending spec review
**Author:** shmammadov@jltech.az (with Claude Code)

---

## 1. Background & Motivation

This project originates from an idea explored in a prior conversation: *why can't LLM
inference work like a blockchain — a peer-to-peer network of consumer machines, each
contributing a slice of compute, instead of centralized data centers?*

The honest conclusion from that exploration:

- The naive version (split one **interactive** chat request across strangers' laptops over
  the internet) fails on physics — home internet is a terrible interconnect for sequential
  work, so per-token latency is unacceptable.
- The version that **works** routes around that constraint:
  - **Sell throughput, not latency.** Target embarrassingly-parallel batch workloads
    (synthetic data generation, evals, RL rollouts, batch processing) where no human waits
    on a spinner.
  - **Pipeline-parallelism across LAN-close "cells."** A group of network-close machines
    collectively hosts one model replica by sharding its layers. Activations passed between
    stages are tiny (~KB/token). In batch mode, pipelining hides per-stage latency:
    throughput is set by the slowest stage, not the sum of stages.
  - **Split layers proportional to each node's memory bandwidth**, keep chains short
    (prefer fewest/fastest nodes that fit the model), and treat churn as the real enemy.

**This spec covers only the first, concretely testable slice of that vision:**

> Take 4–6 commodity office laptops (16 GB RAM each, mixed hardware, some with GPUs) and
> make them collectively serve a single **25–30B-parameter model that no single laptop can
> hold**, via pipeline/layer-parallelism over the office LAN — and demonstrate the
> throughput-over-latency insight with a batch benchmark.

Everything beyond that (verification, credit ledger, internet-scale, auto cell-formation)
is **explicitly out of scope for v1** and recorded in §11 as future work.

## 2. Goals & Success Criteria

**Primary goal:** Prove the "cell" concept on real hardware.

**v1 is successful when:**

1. A 25–30B model (Q4-class GGUF) loads with its layers **split across ≥2 laptops**, where
   the model does **not** fit on any single laptop.
2. The built-in chat UI produces **coherent** single-stream output through the distributed
   cell.
3. The batch benchmark demonstrates **aggregate throughput (tok/s) meaningfully greater than
   single-stream throughput**, empirically confirming the throughput-over-latency insight.
4. The whole pipeline (probe → plan → launch → chat → benchmark) is reproducible from
   documentation by someone who is not the author.

**Non-goals for v1:** beating a data center on latency or cost/token; supporting untrusted
nodes; operating over the public internet; automatic recovery from node churn.

## 3. Foundation Decision

**Build on `llama.cpp` RPC backend.** (Approaches B/exo and C/from-scratch were considered
and rejected for v1 — see §10.)

Rationale:
- Cross-platform and heterogeneous: one codebase handles NVIDIA (CUDA), Apple (Metal),
  integrated GPUs (Vulkan), and pure CPU — matching the mixed office fleet.
- Native pipeline/layer distribution over TCP via `rpc-server` + `--rpc` on the coordinator.
- Direct control of the layer split via `--tensor-split` — the hook for capability-aware
  splitting (§7), which is our core engineering contribution.
- Built-in OpenAI-compatible server + web chat UI — the chat demo is nearly free.
- Caveat: RPC mode is **LAN-only / not secure by design** — acceptable for an office test,
  and consistent with the v1 scope (the internet story is future work).

**Build method (decided):** compile a **pinned llama.cpp release from source** on each laptop
with the appropriate backend enabled. This guarantees matching RPC protocol versions across
nodes (a hard requirement) and correct hardware acceleration on mixed hardware. A specific
release tag/commit will be pinned in `config/` and used everywhere.

## 4. Architecture

Six components. Three are stock llama.cpp; three are ours (where the value lives).

| Component | Role | Origin |
|---|---|---|
| **Worker** (per laptop) | `llama.cpp rpc-server` exposing that machine's compute + RAM over TCP. Backend auto-selected: CUDA / Metal / Vulkan / CPU. | stock |
| **Coordinator** (one laptop) | `llama-server`: loads the GGUF model, connects to all workers via `--rpc`, distributes layers via `--tensor-split`, exposes OpenAI-compatible API + web chat UI. | stock |
| **Capability probe** | Runs once per node; measures RAM, CPU cores, GPU type/VRAM, and a quick memory-bandwidth microbenchmark; emits a per-node record merged into `nodes.json`. | ours |
| **Split planner** | Reads `nodes.json`; computes a capability-proportional layer split; selects the fewest strong nodes that fit the model; emits exact launch commands for coordinator + workers. | ours |
| **Orchestration + healthcheck** | Cross-platform launchers (start workers, start coordinator, ping the cell, report degraded state, one-command relaunch, teardown). | ours |
| **Benchmark harness** | Fires N independent prompts concurrently at the OpenAI API; measures aggregate throughput and per-request latency distribution; writes a report. | ours |

**Topology:** star for control (coordinator ↔ each worker); pipeline for compute (layer
stages chained). All traffic on the office LAN.

**Node roles are just processes:** the coordinator laptop also runs a share of layers (it is
both coordinator and a worker); dedicating it purely to coordination is a config option if it
proves to be a bottleneck.

## 5. Data Flow

**Setup phase (once):**
`probe.py` on each node → merged `nodes.json` → `plan_split.py` → launch commands +
`fleet.json` (resolved plan).

**Runtime:**
1. Workers start `rpc-server` on a known port.
2. Coordinator starts `llama-server` with `--rpc host1:p1,host2:p2,…` and
   `--tensor-split w1,w2,…` for the chosen model.
3. Client request → coordinator OpenAI API → coordinator computes its layer range → streams
   activations (~KB/token) to the next stage → … → logits return to coordinator → token
   sampled → repeat until stop.

**Batch mode:** the benchmark fires many concurrent requests; `llama-server` continuous
batching keeps the pipeline full, so every stage works simultaneously on different sequences.
Aggregate tok/s ≫ single-stream tok/s — this is the demonstration of the core insight.

## 6. Model Selection

Test **two** models to show the tradeoff (final pick confirmed from probe results):

- **Dense 27–32B** — e.g. **Gemma 3 27B** or **Qwen2.5-32B**, Q4_K_M (~16–19 GB). The "hard"
  case: memory-bandwidth-bound, forces genuine sharding.
- **Qwen3-30B-A3B (MoE)** — Q4 (~18 GB weights, but only ~3B params active per token). Same
  memory-sharding requirement, but far faster tokens on CPU-heavy fleets — makes the demo
  feel responsive.

**Memory math:** 30B Q4 ≈ 18 GB weights + KV cache. 4–6 laptops × ~10–12 GB usable each =
40–70 GB fleet RAM → comfortably fits weights + KV + headroom. Confirms the model cannot fit
on one 16 GB node (≈10–12 GB usable) while the fleet holds it easily.

## 7. Capability-Aware Layer Splitting (core contribution)

llama.cpp assigns whole layers per device; `--tensor-split` sets the proportions across
`[coordinator-local, worker1, worker2, …]`.

**Planner algorithm:**
1. Score each node: `score = f(has_GPU, VRAM_or_free_RAM, mem_bandwidth, cpu_cores)`. GPU
   presence and memory bandwidth dominate, since token generation is bandwidth-bound.
2. **Node selection:** choose the fewest, highest-scoring nodes whose combined usable memory
   holds `model_weights + estimated_KV_cache + headroom`. (Per the "prefer fewest/fastest
   nodes" principle — do not blindly spread across all 6.)
3. **Split proportions:** set each selected node's share of layers ∝ its score, so every
   stage takes roughly equal wall-time and the slowest stage doesn't stall the pipeline.
4. Emit `--tensor-split` values and per-node launch commands; write the resolved plan to
   `fleet.json`.

The planner is pure/deterministic given a `nodes.json`, making it directly unit-testable.

## 8. Node-Drop Handling (lightweight, honest)

Full automatic re-formation is **out of scope** for v1. Instead:
- `healthcheck.py` pings all RPC endpoints and the coordinator.
- On a dropped node it reports the degraded cell clearly and offers a **one-command
  relaunch** that re-runs the planner over surviving nodes (which may require choosing a
  smaller model if capacity dropped below the model size).
- This is demoable and honest about the churn problem without over-building recovery.

## 9. Error Handling & Preflight

Actionable checks before launch:
- **Version match:** all nodes run the pinned llama.cpp build (RPC requires it) — fail fast
  with the mismatch listed.
- **Model integrity:** verify GGUF file hash on the coordinator.
- **Reachability:** each worker's RPC port reachable from the coordinator (surface likely
  firewall issues explicitly).
- **Capacity:** total selected-node usable RAM ≥ model size + KV estimate; else refuse and
  suggest a smaller model or more nodes.
- All failures produce a specific, human-readable message and a suggested fix.

## 10. Testing Strategy

- **Unit tests:**
  - Split-planner math: given fixture `nodes.json` inputs, assert correct node selection and
    `--tensor-split` values (including edge cases: one strong + several weak; homogeneous;
    insufficient total memory).
  - Probe output parsing.
- **Integration — "loopback cell":** run 2–3 `rpc-server` instances on a single dev machine
  (localhost, distinct ports) and drive the full pipeline (plan → launch → chat request →
  benchmark). Validates ~all orchestration/benchmark code **before** touching office laptops.
  This lets the bulk of the system be built and verified on the author's Mac.
- **Acceptance (on the laptops):** the three success criteria in §2 — split load across ≥2
  nodes, coherent chat, batch throughput > single-stream.

## 11. Repository Layout & Stack

**Stack:** Python (orchestration, probe, planner, benchmark — readable & cross-platform);
thin `sh` + `ps1` launchers; a pinned llama.cpp release built from source per node.

```
project_amalfi/
  README.md
  docs/
    superpowers/specs/2026-07-23-distributed-inference-cell-design.md   (this file)
    runbook.md            # step-by-step office-laptop setup & run
    results.md            # benchmark results, filled during acceptance
  scripts/
    probe.py              # capability probe -> node record
    plan_split.py         # nodes.json -> split + launch commands + fleet.json
    start_worker.sh / .ps1
    start_coordinator.sh / .ps1
    healthcheck.py
    build_llamacpp.sh / .ps1   # pinned-source build per backend
  bench/
    run_bench.py          # concurrent batch throughput benchmark
    report.py             # summarize into results.md
  config/
    fleet.example.json    # fleet manifest template
    llamacpp.pin          # pinned release tag/commit + build flags per backend
  tests/
    test_plan_split.py
    test_probe.py
```

## 12. Out of Scope for v1 (YAGNI)

Recorded so the larger vision stays intact, explicitly **not built now**:

- TOPLOC-style activation-hash verification of worker output.
- Compute-credit ledger / blockchain / tokenomics.
- Internet / WAN operation and NAT traversal.
- Automatic cell-formation, peer discovery, and churn recovery (KV checkpointing, stage
  redundancy).
- Security hardening / untrusted-node threat model.
- Global tiered scheduler and multi-cell routing.

## 13. Open Questions (to resolve during implementation)

- Exact pinned llama.cpp release tag (choose a recent stable release with solid RPC support).
- Final model files/quantization confirmed once probe data reveals real per-node capacity.
- Whether the coordinator laptop should also serve layers or be coordination-only (decided
  empirically from acceptance runs).
