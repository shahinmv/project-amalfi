# Amalfi Results

## A. Dev-machine loopback validation (2026-07-23) — PASSED

End-to-end proof of the whole pipeline on a single machine before office-laptop testing.

- Machine: Apple M3 Pro, 11 cores, ~19 GB RAM, Metal; llama.cpp pinned tag `b10103`
  (built from source with `-DGGML_RPC=ON -DGGML_METAL=ON`).
- Cell: 2 × `ggml-rpc-server` on `127.0.0.1:50060` and `:50061`; coordinator
  `llama-server` pinned to `--device RPC0,RPC2 --tensor-split 0.5,0.5 -ngl 999`.
- Model: `qwen2.5-0.5b-instruct-q4_k_m.gguf` (small model to exercise the mechanism).

**Evidence the model was split across both workers:** both rpc-server logs show Metal
compute kernels compiling and running (q4_K/q6_K matmuls) — both nodes actively compute.

**Coherent output** (chat via the distributed cell):
> "Distributed computing involves breaking a problem into smaller, manageable parts and
> distributing these parts across multiple computers or processors to achieve faster and
> more efficient processing."

**Benchmark (`bench/run_bench.py`, server `n_slots=4`):**

| mode | concurrency | aggregate tok/s | mean latency s | p95 latency s | ok/total |
|------|-------------|-----------------|----------------|---------------|----------|
| single | 1 | 155.73 | 0.587 | 0.627 | 4/4 |
| batch  | 4 | 247.05 | 1.501 | 1.785 | 32/32 |

**Throughput speedup (batch vs single): 1.59×.** Per-request latency rose (0.59s → 1.5s)
while aggregate tok/s climbed — the throughput-over-latency insight, confirmed. On the real
fleet with a 25–30B model and more slots, the effect is larger.

- Unit suite: 19 passed. Loopback integration test: passed (with `AMALFI_TEST_MODEL` set).

---

## B. Office-laptop acceptance (to fill in during the real test)

- Date:
- Fleet: <n> laptops (list OS / GPU / RAM / measured mem-bandwidth per node)
- Model: <key> (<size> GB), quantization Q4_K_M
- Split: <tensor-split values> across <hosts>

### Success criteria (spec §2)
- [ ] Model loaded split across >= 2 nodes (does not fit on one) — evidence:
- [ ] Coherent chat output — sample:
- [ ] Batch aggregate tok/s > single-stream tok/s — numbers below

### Benchmark
| mode | concurrency | aggregate tok/s | p50 latency s | p95 latency s | ok/total |
|------|-------------|-----------------|---------------|---------------|----------|
| single | 1 | | | | |
| batch  | 8 | | | | |

### Notes / observations
