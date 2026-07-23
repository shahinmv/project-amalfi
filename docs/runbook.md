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

- **Windows:** Git, Python 3 (tick "Add to PATH"), and **Visual Studio Build Tools 2022
  with the "Desktop development with C++" workload** (provides the MSVC compiler — required).
  Install the C++ tools in one command (admin PowerShell):

  ```powershell
  winget install --id Microsoft.VisualStudio.2022.BuildTools -e `
    --override "--quiet --wait --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
  ```

  CMake itself is installed automatically by `bootstrap.ps1` via pip — no separate install.
  Start with `-Backend cpu` (works with just the C++ tools). Only add `-Backend cuda`
  (needs the CUDA Toolkit) or `-Backend vulkan` (needs the Vulkan SDK) once CPU works.
- **macOS:** `xcode-select --install`, plus `brew install cmake git python`.
- **Linux:** `sudo apt install -y build-essential cmake git python3 python3-venv`
  (add the Vulkan SDK / `libvulkan-dev` for integrated-GPU acceleration, or CUDA for NVIDIA).

## Fast path (recommended)

### Windows — one command does everything (installs too)

Open **PowerShell as Administrator** and run **one** of these — no IP needed, it auto-detects:

```powershell
# fresh laptop (installs Git, clones, sets up):
& ([scriptblock]::Create((irm https://raw.githubusercontent.com/shahinmv/project-amalfi/main/scripts/setup_windows.ps1)))

# already cloned the repo:
powershell -ExecutionPolicy Bypass -File scripts\setup_windows.ps1
```

`setup_windows.ps1` installs Git + Python + the Visual Studio C++ Build Tools, clones the
repo if needed, **auto-detects this laptop's LAN IP** (and prints it), builds llama.cpp, and
probes the laptop (writes `node_<host>.json`). It's idempotent — safe to re-run. Add
`-StartWorker` to also launch the worker, `-RpcHost <ip>` to override the detected IP, or
`-Backend cuda`/`-Backend vulkan` once CPU works.

> If a build error appears right after a *fresh* Build Tools install, reboot once and re-run.

### macOS / Linux — one command (prereqs installed separately, see step 1)

```bash
./scripts/bootstrap.sh --backend auto --rpc-host <this-laptops-LAN-IP> --start-worker
```

Both create the venv, install deps + cmake, build the pinned llama.cpp, and probe the
machine. Then on the **coordinator only**, merge every `node_*.json` into `nodes.json`
(`scripts/merge_nodes.py`) and continue from step 4. Steps 2–3 below are the manual
equivalent of what these scripts do.

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

Copy every laptop's `node_*.json` to the coordinator (into the repo dir), then merge them
into a single `nodes.json` with the helper:

```bash
./.venv/bin/python scripts/merge_nodes.py            # globs node_*.json in the current dir
# or list them explicitly:  scripts/merge_nodes.py node_a.json node_b.json --out nodes.json
```

It de-dups by `rpc_host:port` and prints a summary of the fleet it assembled.

## 4. Plan the split (coordinator)

```bash
./.venv/bin/python scripts/plan_split.py --nodes nodes.json --model qwen3-30b-a3b-q4 --out fleet.json
```

- **Default model: `qwen3-30b-a3b-q4`** (Qwen3-30B-A3B, MoE — ~3B active/token, so it's
  fast even on CPU/integrated-GPU laptops while still proving 30B-scale sharding). Other
  keys in `config/models.json`: `gemma-3-27b-q4`, `qwen2.5-32b-q4`.
- Review `fleet.json`: `cap_policy`, `selected_hosts`, `tensor_split`, `est_ram_per_node`.
- **RAM per laptop is capped automatically.** By default the planner uses at most **half of
  each laptop's RAM/VRAM** (`--ram-fraction 0.5`), bounded by whatever is actually free — so
  it adapts per machine and never hogs a work laptop. It prints the estimated RAM per laptop.
- **Tune the policy:**
  - `--ram-fraction 0.6` — use up to 60% per laptop (fits the model on fewer machines).
  - `--max-ram-gb 4` — hard absolute cap instead of a fraction (overrides `--ram-fraction`).
    A 4 GB cap on the 30B needs ~6–7 laptops (~3 GB weights each).
  - `--ram-fraction 0` — disable capping; split purely by compute speed (uses the fewest
    laptops that physically fit).
  - `--ram-overhead-gb` — tune the reserved non-weight RAM (default 1 GB).
- If it reports insufficient capacity, add laptops, raise the fraction/cap, or pick a smaller model.

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
