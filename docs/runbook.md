# Amalfi Office-Laptop Runbook (v1)

Goal: run one 25–30B model split across 4–6 office laptops on the LAN, chat with it,
and benchmark single-stream vs batch throughput.

> **Security:** RPC mode is LAN-only and unauthenticated. Only open the RPC port
> (`50052`) between laptops on the office network. Never expose it to the internet.

---

## 0. Roles

- Pick **one** laptop as the **coordinator** (ideally the strongest / most stable, on a
  wired connection if possible). It runs `llama-server` *and* an rpc-server.
- Every laptop (including the coordinator) is a **worker** and runs an rpc-server.

## 1. Prerequisites (every laptop)

Install: `git`, `cmake`, a C/C++ build toolchain, and `python3` (3.9+).

- **Windows:** Visual Studio Build Tools (C++ workload), Git, Python. Use PowerShell.
  (CMake is installed automatically by `bootstrap.ps1` via pip — no separate install.)
  For NVIDIA laptops also install the CUDA Toolkit; for integrated-GPU accel install the
  Vulkan SDK (otherwise the `cpu` backend is used).
- **macOS:** `xcode-select --install`, plus `brew install cmake git python`.
- **Linux:** `sudo apt install -y build-essential cmake git python3 python3-venv`
  (add the Vulkan SDK / `libvulkan-dev` for integrated-GPU acceleration, or CUDA for NVIDIA).

## Fast path (recommended): one-command bootstrap

On each laptop, after installing the prereqs above and cloning the repo, run:

```bash
./scripts/bootstrap.sh --backend auto --rpc-host <this-laptops-LAN-IP> --start-worker
# Windows:  ./scripts/bootstrap.ps1 -Backend auto -RpcHost <IP> -StartWorker
```

This creates the venv, installs deps, auto-detects the backend, builds the pinned
llama.cpp, probes the machine (writing `node_<host>.json`), and — with `--start-worker` —
launches this node's rpc-server. Omit `--start-worker` to just set up + probe.

Then on the **coordinator only**, merge every `node_*.json` into a single `nodes.json`
array and continue from step 4. Steps 2–3 below are the manual equivalent of what
bootstrap does.

## 2. Build llama.cpp (every laptop, same pinned tag)

```bash
git clone <this-repo> amalfi && cd amalfi
python3 -m venv .venv && ./.venv/bin/python -m pip install -r requirements.txt
# Confirm the pinned tag in config/llamacpp.pin is one your team validated.
./scripts/build_llamacpp.sh <cuda|metal|vulkan|cpu>     # choose backend for THIS laptop
```

- NVIDIA laptop → `cuda`. Apple Silicon → `metal`. Intel/AMD integrated → `vulkan`
  (fallback `cpu` if Vulkan SDK is missing). No usable GPU → `cpu`.
- Windows: `./scripts/build_llamacpp.ps1 -Backend <cuda|vulkan|cpu>`.
- Verify: `vendor/llama.cpp/build/bin/ggml-rpc-server` (or `ggml-rpc-server.exe`) exists.

**All nodes must use the identical pinned tag** — RPC refuses mismatched builds.

## 3. Probe every laptop

On each laptop, find its LAN IP (e.g. `ipconfig` / `ifconfig` / `ip addr`), then:

```bash
./.venv/bin/python scripts/probe.py --rpc-host <this-laptops-LAN-IP> --out node.json
```

Copy every `node.json` to the coordinator and combine them into a single JSON **array**
named `nodes.json` (see `config/fleet.example.json` for the exact shape).

## 4. Plan the split (coordinator)

```bash
./.venv/bin/python scripts/plan_split.py --nodes nodes.json --model qwen3-30b-a3b-q4 --out fleet.json
```

- **Default model: `qwen3-30b-a3b-q4`** (Qwen3-30B-A3B, MoE — ~3B active/token, so it's
  fast even on CPU/integrated-GPU laptops while still proving 30B-scale sharding). Other
  keys in `config/models.json`: `gemma-3-27b-q4`, `qwen2.5-32b-q4`.
- Review `fleet.json`: `selected_hosts`, `tensor_split`, and the printed commands.
- If it reports insufficient capacity, add nodes or pick a smaller model.

## 5. Get the model (coordinator)

Download the chosen GGUF into `models/` on the coordinator (workers do **not** need it —
only the coordinator loads it and streams layers to the rpc-servers). For the default:

```bash
mkdir -p models
curl -L -o models/Qwen3-30B-A3B-Q4_K_M.gguf \
  https://huggingface.co/Qwen/Qwen3-30B-A3B-GGUF/resolve/main/Qwen3-30B-A3B-Q4_K_M.gguf
```

(~18.6 GB single file.) Each model's `hf_repo`/`hf_file` in `config/models.json` gives the
download source; the local filename must match the entry's `gguf` field.

## 6. Launch the cell

1. On **every** laptop (workers): `./scripts/start_worker.sh 50052`
   (Windows: `./scripts/start_worker.ps1 -Port 50052`).
2. On the **coordinator**: `./scripts/start_coordinator.sh fleet.json`
   (Windows: `./scripts/start_coordinator.ps1 -Fleet fleet.json`).

**Verify the device mapping (important — validated on a real build):**
llama.cpp enumerates the coordinator's **own local GPU plus every RPC device**, and a
device may appear more than once. List them first:

```bash
vendor/llama.cpp/build/bin/llama-server --rpc <rpc-list-from-fleet.json> --list-devices
```

You'll see entries like `MTL0` (local), `RPC0: <host>:<port>`, `RPC2: <host2>:<port2>`, …
To distribute the model across **only the worker devices** with a deterministic split,
pass the real RPC device names via `--device` and give `--tensor-split` the same number of
values in the same order. Example validated on the loopback cell:

```bash
llama-server --model models/<gguf> \
  --rpc <host1>:50052,<host2>:50052 \
  --device RPC0,RPC2 --tensor-split 0.6,0.4 -ngl 999 \
  --ctx-size 4096 --host 0.0.0.0 --port 8080
```

The `tensor_split` values in `fleet.json` are the intended proportions — map them onto the
RPC devices you selected with `--list-devices`. If you omit `--device`, the coordinator's
local GPU also receives a share (fine, but then add a leading split value for it).

## 7. Health check

```bash
./.venv/bin/python scripts/healthcheck.py --fleet fleet.json
```

Expect every node `UP`. If a node is `DOWN`, fix networking/firewall (open TCP 50052
between laptops), or re-plan over the surviving nodes and restart.

## 8. Chat + benchmark

- Chat UI: open `http://<coordinator-IP>:8080` in a browser.
- Benchmark:

```bash
./.venv/bin/python bench/run_bench.py --url http://<coordinator-IP>:8080 --concurrency 8 --max-tokens 128
```

It measures single-stream (concurrency 1) then batch (concurrency 8) and writes
`docs/results.md`. Expect **batch aggregate tok/s > single-stream tok/s**.

## 9. Record results

Fill in `docs/results.md` with fleet composition, model, split, and the benchmark table.

## Troubleshooting

- **RPC version mismatch:** rebuild all nodes from the same `config/llamacpp.pin` tag.
- **Node unreachable:** open TCP `50052` on the LAN; confirm the probe recorded the real
  LAN IP (not `127.0.0.1`).
- **OOM on load:** the model is too big for the selected nodes — pick a smaller model key
  or add nodes; the planner's capacity check should catch this before launch.
- **Very slow tokens single-stream:** expected. The win is batch throughput — run the
  benchmark with higher `--concurrency`.
