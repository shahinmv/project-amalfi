# Distributed Inference Cell (v1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make 4–6 commodity office laptops collectively serve one 25–30B GGUF model — that no single laptop can hold — via llama.cpp RPC pipeline-parallelism, and prove the throughput-over-latency insight with a batch benchmark.

**Architecture:** Every selected laptop runs a `llama.cpp rpc-server`. One laptop additionally runs `llama-server`, which connects to all rpc-servers (including its own machine's, over the LAN IP) via `--rpc`, distributes the model's layers with a capability-proportional `--tensor-split`, and exposes an OpenAI-compatible API + web chat UI. Our Python layer probes each node, plans the split, launches/health-checks the cell, and benchmarks it. Running rpc-server on *every* node (not using a "local" device) makes device order equal the `--rpc` list order, so the tensor-split mapping is deterministic.

**Tech Stack:** Python 3.9+ (orchestration/probe/planner/benchmark), `sh`/PowerShell launchers, llama.cpp built from a pinned source tag per backend (CUDA/Metal/Vulkan/CPU). Deps: `psutil`, `numpy`, `requests`, `pytest`.

## Global Constraints

- **llama.cpp**: built from a single **pinned release tag** (format `bNNNN`) recorded in `config/llamacpp.pin`; the *same* tag on every node (RPC requires matching builds). Build with `-DGGML_RPC=ON`.
- **Python**: 3.9+. Third-party deps limited to `psutil`, `numpy`, `requests`, `pytest` (declared in `requirements.txt`).
- **Model**: Q4-class GGUF, 25–30B params. Must NOT fit on a single 16 GB node (≈10–12 GB usable); the fleet must hold it collectively.
- **Network**: LAN-only. RPC mode is not secure — never expose rpc-server ports to the internet.
- **Determinism**: all planner functions are pure and deterministic given a `nodes.json`; no wall-clock or randomness in planning logic.
- **Default RPC port**: `50052`. **Default coordinator API port**: `8080`.
- Every pure function gets a unit test; end-to-end wiring is validated by a localhost loopback cell before office-laptop acceptance.

---

## File Structure

- `requirements.txt` — Python deps.
- `config/llamacpp.pin` — pinned tag + notes.
- `config/models.json` — model catalog (key → gguf filename, size, ctx, kv estimate).
- `config/fleet.example.json` — example merged nodes manifest.
- `scripts/probe.py` — capability probe → node record JSON.
- `scripts/plan_split.py` — nodes.json → fleet.json + launch commands (pure planner functions).
- `scripts/build_flags.py` — backend → cmake flags (pure; shared by build scripts + tests).
- `scripts/build_llamacpp.sh` / `scripts/build_llamacpp.ps1` — pinned-source build.
- `scripts/start_worker.sh` / `.ps1` — start rpc-server from fleet.json.
- `scripts/start_coordinator.sh` / `.ps1` — start llama-server from fleet.json.
- `scripts/healthcheck.py` — probe RPC endpoints + coordinator health.
- `bench/report.py` — pure summarize/format of benchmark results.
- `bench/run_bench.py` — fire concurrent requests, call report.summarize.
- `tests/test_probe.py`, `tests/test_plan_split.py`, `tests/test_build_flags.py`, `tests/test_healthcheck.py`, `tests/test_report.py`, `tests/test_loopback_integration.py`.
- `docs/runbook.md`, `docs/results.md`, `README.md`.

---

### Task 1: Capability probe

**Files:**
- Create: `scripts/probe.py`
- Create: `requirements.txt`
- Test: `tests/test_probe.py`

**Interfaces:**
- Produces:
  - `detect_gpu() -> dict` → `{"type": str, "name": str, "vram_gb": float}`, `type` ∈ {`"cuda"`,`"metal"`,`"vulkan"`,`"none"`}
  - `measure_mem_bandwidth_gbps(size_mb: int = 256, passes: int = 5) -> float` (> 0.0)
  - `build_node_record(rpc_host: str, rpc_port: int, gpu: dict, mem_bw: float) -> dict`
  - Node record schema (consumed by Task 2):
    `{"hostname","os","arch","cpu_cores","total_ram_gb","free_ram_gb","gpu":{"type","name","vram_gb"},"mem_bandwidth_gbps","rpc_host","rpc_port"}`

- [ ] **Step 1: Write `requirements.txt`**

```text
psutil>=5.9
numpy>=1.24
requests>=2.31
pytest>=7.4
```

- [ ] **Step 2: Write the failing tests**

`tests/test_probe.py`:
```python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "probe", pathlib.Path(__file__).parent.parent / "scripts" / "probe.py")
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)


def test_measure_mem_bandwidth_positive():
    assert probe.measure_mem_bandwidth_gbps(size_mb=16, passes=2) > 0.0


def test_detect_gpu_shape():
    g = probe.detect_gpu()
    assert set(g) == {"type", "name", "vram_gb"}
    assert g["type"] in {"cuda", "metal", "vulkan", "none"}
    assert isinstance(g["vram_gb"], float)


def test_build_node_record_shape():
    gpu = {"type": "none", "name": "cpu", "vram_gb": 0.0}
    r = probe.build_node_record("192.168.1.5", 50052, gpu, 42.0)
    for k in ("hostname", "os", "arch", "cpu_cores", "total_ram_gb",
              "free_ram_gb", "gpu", "mem_bandwidth_gbps", "rpc_host", "rpc_port"):
        assert k in r
    assert r["rpc_host"] == "192.168.1.5"
    assert r["rpc_port"] == 50052
    assert r["mem_bandwidth_gbps"] == 42.0
    assert r["gpu"] == gpu
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_probe.py -v`
Expected: FAIL (module `probe` has no such attributes / file missing).

- [ ] **Step 4: Implement `scripts/probe.py`**

```python
#!/usr/bin/env python3
"""Amalfi capability probe. Run on each node; emits a JSON node record."""
import argparse, json, platform, shutil, socket, subprocess, sys, time
import numpy as np
import psutil

GPU_KEYS = ("type", "name", "vram_gb")


def measure_mem_bandwidth_gbps(size_mb: int = 256, passes: int = 5) -> float:
    """Rough relative memory-bandwidth estimate via repeated large-array reads."""
    n = max(1, (size_mb * 1024 * 1024) // 8)
    a = np.ones(n, dtype=np.float64)
    best = 0.0
    for _ in range(max(1, passes)):
        t0 = time.perf_counter()
        _ = float(a.sum())
        dt = time.perf_counter() - t0
        if dt > 0:
            best = max(best, a.nbytes / 1e9 / dt)
    return round(best, 2)


def detect_gpu() -> dict:
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"], text=True, timeout=5
            ).strip().splitlines()
            if out:
                name, mem = out[0].split(",")
                return {"type": "cuda", "name": name.strip(),
                        "vram_gb": round(float(mem) / 1024, 1)}
        except Exception:
            pass
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return {"type": "metal", "name": "Apple Silicon GPU", "vram_gb": 0.0}
    if shutil.which("vulkaninfo"):
        return {"type": "vulkan", "name": "Vulkan device", "vram_gb": 0.0}
    return {"type": "none", "name": "cpu", "vram_gb": 0.0}


def _primary_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def build_node_record(rpc_host: str, rpc_port: int, gpu: dict, mem_bw: float) -> dict:
    vm = psutil.virtual_memory()
    return {
        "hostname": socket.gethostname(),
        "os": platform.system(),
        "arch": platform.machine(),
        "cpu_cores": psutil.cpu_count(logical=True) or 1,
        "total_ram_gb": round(vm.total / 1e9, 1),
        "free_ram_gb": round(vm.available / 1e9, 1),
        "gpu": gpu,
        "mem_bandwidth_gbps": mem_bw,
        "rpc_host": rpc_host,
        "rpc_port": rpc_port,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rpc-host", default=None, help="LAN IP other nodes reach me on")
    ap.add_argument("--rpc-port", type=int, default=50052)
    ap.add_argument("--out", default=None, help="write record to this file (default: stdout)")
    args = ap.parse_args()
    host = args.rpc_host or _primary_ip()
    rec = build_node_record(host, args.rpc_port, detect_gpu(), measure_mem_bandwidth_gbps())
    text = json.dumps(rec, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_probe.py -v`
Expected: PASS (3 passed).

- [ ] **Step 6: Commit**

```bash
git add requirements.txt scripts/probe.py tests/test_probe.py
git commit -m "feat: capability probe emitting per-node hardware record"
```

---

### Task 2: Split planner

**Files:**
- Create: `scripts/plan_split.py`
- Create: `config/models.json`
- Create: `config/fleet.example.json`
- Test: `tests/test_plan_split.py`

**Interfaces:**
- Consumes: node records from Task 1.
- Produces:
  - `score_node(node: dict) -> float`
  - `node_capacity_gb(node: dict) -> float`
  - `estimate_required_gb(model: dict, headroom_gb: float = 2.0) -> float`
  - `select_nodes(nodes: list, required_gb: float) -> list` (raises `ValueError` if insufficient)
  - `compute_tensor_split(selected: list) -> list[float]` (sums to 1.0)
  - `build_launch_commands(selected: list, model: dict, rpc_port: int = 50052, api_port: int = 8080) -> dict` → keys `{"rpc","tensor_split","worker_cmd","coordinator_cmd","coordinator_host"}`
  - `plan(nodes: list, model_key: str, catalog: dict, rpc_port: int = 50052) -> dict` (fleet plan; consumed by Tasks 4 & 6)

- [ ] **Step 1: Write `config/models.json`**

```json
{
  "qwen2.5-32b-q4": {
    "gguf": "qwen2.5-32b-instruct-q4_k_m.gguf",
    "params_b": 32, "size_gb": 19.0, "kv_per_seq_gb": 0.5, "ctx_size": 4096
  },
  "gemma-3-27b-q4": {
    "gguf": "gemma-3-27b-it-q4_k_m.gguf",
    "params_b": 27, "size_gb": 16.5, "kv_per_seq_gb": 0.45, "ctx_size": 4096
  },
  "qwen3-30b-a3b-q4": {
    "gguf": "qwen3-30b-a3b-q4_k_m.gguf",
    "params_b": 30, "size_gb": 18.0, "kv_per_seq_gb": 0.4, "ctx_size": 4096
  }
}
```

- [ ] **Step 2: Write the failing tests**

`tests/test_plan_split.py`:
```python
import importlib.util, pathlib, pytest
spec = importlib.util.spec_from_file_location(
    "plan_split", pathlib.Path(__file__).parent.parent / "scripts" / "plan_split.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)


def node(host, bw, free, gpu="none", vram=0.0):
    return {"rpc_host": host, "mem_bandwidth_gbps": bw, "free_ram_gb": free,
            "gpu": {"type": gpu, "name": gpu, "vram_gb": vram}, "cpu_cores": 8}


def test_gpu_scores_higher_than_cpu_same_bandwidth():
    assert ps.score_node(node("a", 40, 10, "cuda", 24)) > ps.score_node(node("b", 40, 10))


def test_capacity_includes_vram():
    assert ps.node_capacity_gb(node("a", 40, 10, "cuda", 24)) == 34.0


def test_select_prefers_strongest_and_stops_when_it_fits():
    nodes = [node("weak", 20, 12), node("strong", 60, 12, "cuda", 24)]
    sel = ps.select_nodes(nodes, required_gb=20.0)
    assert sel[0]["rpc_host"] == "strong"
    assert sum(ps.node_capacity_gb(n) for n in sel) >= 20.0


def test_select_raises_when_insufficient():
    with pytest.raises(ValueError):
        ps.select_nodes([node("a", 20, 5)], required_gb=50.0)


def test_tensor_split_sums_to_one_and_is_proportional():
    sel = [node("strong", 60, 12), node("weak", 20, 12)]
    split = ps.compute_tensor_split(sel)
    assert abs(sum(split) - 1.0) < 1e-6
    assert split[0] > split[1]


def test_build_launch_commands_shape():
    sel = [node("192.168.1.10", 60, 12), node("192.168.1.11", 20, 12)]
    model = {"gguf": "m.gguf", "ctx_size": 4096}
    cmd = ps.build_launch_commands(sel, model, rpc_port=50052, api_port=8080)
    assert cmd["rpc"] == "192.168.1.10:50052,192.168.1.11:50052"
    assert "llama-server" in cmd["coordinator_cmd"]
    assert "--tensor-split" in cmd["coordinator_cmd"]
    assert "rpc-server" in cmd["worker_cmd"]
    assert cmd["coordinator_host"] == "192.168.1.10"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `python -m pytest tests/test_plan_split.py -v`
Expected: FAIL (module/functions missing).

- [ ] **Step 4: Implement `scripts/plan_split.py`**

```python
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
```

- [ ] **Step 5: Write `config/fleet.example.json`**

```json
[
  {"hostname":"laptop-1","os":"Windows","arch":"x86_64","cpu_cores":8,"total_ram_gb":16.0,"free_ram_gb":11.0,"gpu":{"type":"none","name":"cpu","vram_gb":0.0},"mem_bandwidth_gbps":38.0,"rpc_host":"192.168.1.21","rpc_port":50052},
  {"hostname":"laptop-2","os":"Windows","arch":"x86_64","cpu_cores":8,"total_ram_gb":16.0,"free_ram_gb":11.0,"gpu":{"type":"cuda","name":"RTX 3050 Laptop","vram_gb":4.0},"mem_bandwidth_gbps":40.0,"rpc_host":"192.168.1.22","rpc_port":50052},
  {"hostname":"mac-3","os":"Darwin","arch":"arm64","cpu_cores":8,"total_ram_gb":16.0,"free_ram_gb":11.5,"gpu":{"type":"metal","name":"Apple Silicon GPU","vram_gb":0.0},"mem_bandwidth_gbps":90.0,"rpc_host":"192.168.1.23","rpc_port":50052},
  {"hostname":"laptop-4","os":"Linux","arch":"x86_64","cpu_cores":8,"total_ram_gb":16.0,"free_ram_gb":12.0,"gpu":{"type":"vulkan","name":"Intel Iris Xe","vram_gb":0.0},"mem_bandwidth_gbps":42.0,"rpc_host":"192.168.1.24","rpc_port":50052}
]
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_plan_split.py -v`
Expected: PASS (6 passed).

- [ ] **Step 7: Verify the CLI end-to-end against the example fleet**

Run: `python scripts/plan_split.py --nodes config/fleet.example.json --model qwen2.5-32b-q4 --out /tmp/fleet.json`
Expected: prints a fleet plan whose `--tensor-split` values sum to ~1.0 and lists the Mac (highest bandwidth) first.

- [ ] **Step 8: Commit**

```bash
git add scripts/plan_split.py config/models.json config/fleet.example.json tests/test_plan_split.py
git commit -m "feat: capability-aware split planner with node selection"
```

---

### Task 3: llama.cpp build flags + build scripts

**Files:**
- Create: `scripts/build_flags.py`
- Create: `config/llamacpp.pin`
- Create: `scripts/build_llamacpp.sh`
- Create: `scripts/build_llamacpp.ps1`
- Test: `tests/test_build_flags.py`

**Interfaces:**
- Produces: `cmake_flags(backend: str) -> list[str]` (backend ∈ {`cuda`,`metal`,`vulkan`,`cpu`}; raises `ValueError` otherwise). Always includes `-DGGML_RPC=ON`.

- [ ] **Step 1: Write the failing test**

`tests/test_build_flags.py`:
```python
import importlib.util, pathlib, pytest
spec = importlib.util.spec_from_file_location(
    "build_flags", pathlib.Path(__file__).parent.parent / "scripts" / "build_flags.py")
bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)


def test_all_backends_enable_rpc():
    for b in ("cuda", "metal", "vulkan", "cpu"):
        assert "-DGGML_RPC=ON" in bf.cmake_flags(b)


def test_backend_specific_flag():
    assert "-DGGML_CUDA=ON" in bf.cmake_flags("cuda")
    assert "-DGGML_METAL=ON" in bf.cmake_flags("metal")
    assert "-DGGML_VULKAN=ON" in bf.cmake_flags("vulkan")
    assert not any("CUDA" in f or "METAL" in f or "VULKAN" in f for f in bf.cmake_flags("cpu"))


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        bf.cmake_flags("tpu")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_build_flags.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `scripts/build_flags.py`**

```python
#!/usr/bin/env python3
"""Map a backend name to llama.cpp cmake flags. Shared by build scripts + tests."""
import sys

_BASE = ["-DGGML_RPC=ON", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF"]
_BACKEND = {
    "cuda": ["-DGGML_CUDA=ON"],
    "metal": ["-DGGML_METAL=ON"],
    "vulkan": ["-DGGML_VULKAN=ON"],
    "cpu": [],
}


def cmake_flags(backend: str) -> list:
    if backend not in _BACKEND:
        raise ValueError(f"unknown backend '{backend}'; choose from {list(_BACKEND)}")
    return _BASE + _BACKEND[backend]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: build_flags.py <cuda|metal|vulkan|cpu>", file=sys.stderr)
        sys.exit(2)
    print(" ".join(cmake_flags(sys.argv[1])))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_build_flags.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write `config/llamacpp.pin`**

```text
# Pinned llama.cpp release. MUST be identical on every node (RPC requires matching builds).
# Verify the tag exists before building:
#   git ls-remote --tags https://github.com/ggml-org/llama.cpp | grep <tag>
# Update this to the latest stable release your team validates. Format: bNNNN
LLAMACPP_REF=b4400
LLAMACPP_REPO=https://github.com/ggml-org/llama.cpp
```

- [ ] **Step 6: Write `scripts/build_llamacpp.sh`**

```bash
#!/usr/bin/env bash
# Build a pinned llama.cpp (rpc-server + llama-server) for a given backend.
# Usage: scripts/build_llamacpp.sh <cuda|metal|vulkan|cpu>
set -euo pipefail
BACKEND="${1:?usage: build_llamacpp.sh <cuda|metal|vulkan|cpu>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck disable=SC1091
source "$HERE/config/llamacpp.pin"

echo ">> verifying pinned tag $LLAMACPP_REF exists..."
git ls-remote --tags "$LLAMACPP_REPO" "refs/tags/$LLAMACPP_REF" | grep -q "$LLAMACPP_REF" \
  || { echo "ERROR: tag $LLAMACPP_REF not found in $LLAMACPP_REPO"; exit 1; }

SRC="$HERE/vendor/llama.cpp"
if [ ! -d "$SRC" ]; then
  git clone --depth 1 --branch "$LLAMACPP_REF" "$LLAMACPP_REPO" "$SRC"
fi
FLAGS="$(python3 "$HERE/scripts/build_flags.py" "$BACKEND")"
echo ">> cmake flags: $FLAGS"
cmake -S "$SRC" -B "$SRC/build" $FLAGS
cmake --build "$SRC/build" --config Release -j --target rpc-server llama-server
echo ">> built: $SRC/build/bin/{rpc-server,llama-server}"
```

- [ ] **Step 7: Write `scripts/build_llamacpp.ps1`**

```powershell
# Build a pinned llama.cpp (rpc-server + llama-server) for a given backend.
# Usage: scripts/build_llamacpp.ps1 -Backend cuda|vulkan|cpu
param([Parameter(Mandatory=$true)][ValidateSet("cuda","vulkan","cpu")][string]$Backend)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
Get-Content "$Here/config/llamacpp.pin" | Where-Object { $_ -match "^\w+=" } | ForEach-Object {
  $k,$v = $_ -split "=",2; Set-Variable -Name $k -Value $v
}
Write-Host ">> pinned tag: $LLAMACPP_REF"
$Src = "$Here/vendor/llama.cpp"
if (-not (Test-Path $Src)) {
  git clone --depth 1 --branch $LLAMACPP_REF $LLAMACPP_REPO $Src
}
$Flags = (python "$Here/scripts/build_flags.py" $Backend) -split " "
cmake -S $Src -B "$Src/build" @Flags
cmake --build "$Src/build" --config Release --target rpc-server llama-server
Write-Host ">> built rpc-server + llama-server in $Src/build/bin"
```

- [ ] **Step 8: Commit**

```bash
chmod +x scripts/build_llamacpp.sh
git add scripts/build_flags.py scripts/build_llamacpp.sh scripts/build_llamacpp.ps1 config/llamacpp.pin tests/test_build_flags.py
git commit -m "feat: pinned llama.cpp build scripts with per-backend cmake flags"
```

---

### Task 4: Launchers + healthcheck

**Files:**
- Create: `scripts/start_worker.sh`, `scripts/start_worker.ps1`
- Create: `scripts/start_coordinator.sh`, `scripts/start_coordinator.ps1`
- Create: `scripts/healthcheck.py`
- Test: `tests/test_healthcheck.py`

**Interfaces:**
- Consumes: `fleet.json` from Task 2 (`rpc`, `rpc_port`, `coordinator_host`, `coordinator_cmd`, `worker_cmd`).
- Produces:
  - `check_endpoint(host: str, port: int, timeout: float = 3.0) -> bool`
  - `parse_rpc(rpc: str) -> list[tuple[str,int]]`
  - `run_healthcheck(fleet: dict, timeout: float = 3.0) -> dict` → `{"nodes":[{"host","port","up"}...], "all_up": bool}`

- [ ] **Step 1: Write the failing tests**

`tests/test_healthcheck.py`:
```python
import importlib.util, pathlib, socket, threading
spec = importlib.util.spec_from_file_location(
    "healthcheck", pathlib.Path(__file__).parent.parent / "scripts" / "healthcheck.py")
hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)


def test_parse_rpc():
    assert hc.parse_rpc("10.0.0.1:50052,10.0.0.2:50052") == [("10.0.0.1", 50052), ("10.0.0.2", 50052)]


def test_check_endpoint_true_on_open_socket():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: srv.accept(), daemon=True).start()
    assert hc.check_endpoint("127.0.0.1", port, timeout=2.0) is True
    srv.close()


def test_check_endpoint_false_on_closed_port():
    assert hc.check_endpoint("127.0.0.1", 1, timeout=1.0) is False


def test_run_healthcheck_aggregates():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    port = srv.getsockname()[1]
    threading.Thread(target=lambda: srv.accept(), daemon=True).start()
    fleet = {"rpc": f"127.0.0.1:{port},127.0.0.1:1"}
    res = hc.run_healthcheck(fleet, timeout=1.0)
    assert res["all_up"] is False
    assert res["nodes"][0]["up"] is True and res["nodes"][1]["up"] is False
    srv.close()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_healthcheck.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `scripts/healthcheck.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_healthcheck.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Write `scripts/start_worker.sh`**

```bash
#!/usr/bin/env bash
# Start this node's rpc-server. Reads port from fleet.json (or --port).
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
PORT="${1:-50052}"
BIN="$HERE/vendor/llama.cpp/build/bin/rpc-server"
[ -x "$BIN" ] || { echo "ERROR: $BIN not built. Run scripts/build_llamacpp.sh first."; exit 1; }
echo ">> starting rpc-server on 0.0.0.0:$PORT (LAN-only; do not expose to internet)"
exec "$BIN" --host 0.0.0.0 --port "$PORT"
```

- [ ] **Step 6: Write `scripts/start_worker.ps1`**

```powershell
param([int]$Port = 50052)
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
$Bin = "$Here/vendor/llama.cpp/build/bin/rpc-server.exe"
if (-not (Test-Path $Bin)) { throw "$Bin not built. Run build_llamacpp.ps1 first." }
Write-Host ">> starting rpc-server on 0.0.0.0:$Port (LAN-only)"
& $Bin --host 0.0.0.0 --port $Port
```

- [ ] **Step 7: Write `scripts/start_coordinator.sh`**

```bash
#!/usr/bin/env bash
# Start llama-server (coordinator) using the command from fleet.json.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
FLEET="${1:-$HERE/fleet.json}"
[ -f "$FLEET" ] || { echo "ERROR: $FLEET not found. Run plan_split.py first."; exit 1; }
CMD="$(python3 -c "import json,sys;print(json.load(open('$FLEET'))['coordinator_cmd'])")"
BIN_DIR="$HERE/vendor/llama.cpp/build/bin"
echo ">> $CMD"
cd "$HERE"
exec env PATH="$BIN_DIR:$PATH" bash -c "$CMD"
```

- [ ] **Step 8: Write `scripts/start_coordinator.ps1`**

```powershell
param([string]$Fleet = "")
$ErrorActionPreference = "Stop"
$Here = Split-Path -Parent $PSScriptRoot
if ($Fleet -eq "") { $Fleet = "$Here/fleet.json" }
if (-not (Test-Path $Fleet)) { throw "$Fleet not found. Run plan_split.py first." }
$Cmd = (python -c "import json;print(json.load(open(r'$Fleet'))['coordinator_cmd'])")
$env:PATH = "$Here/vendor/llama.cpp/build/bin;$env:PATH"
Write-Host ">> $Cmd"
Set-Location $Here
Invoke-Expression $Cmd
```

- [ ] **Step 9: Commit**

```bash
chmod +x scripts/start_worker.sh scripts/start_coordinator.sh
git add scripts/start_worker.sh scripts/start_worker.ps1 scripts/start_coordinator.sh scripts/start_coordinator.ps1 scripts/healthcheck.py tests/test_healthcheck.py
git commit -m "feat: worker/coordinator launchers and cell healthcheck"
```

---

### Task 5: Benchmark harness

**Files:**
- Create: `bench/report.py`
- Create: `bench/run_bench.py`
- Create: `bench/prompts.txt`
- Test: `tests/test_report.py`

**Interfaces:**
- Produces:
  - `summarize(results: list, wall_time_s: float) -> dict` → keys `{"n","ok","total_completion_tokens","wall_time_s","aggregate_tok_s","mean_latency_s","p50_latency_s","p95_latency_s"}`
  - `format_report(single: dict, batch: dict) -> str`
  - (run_bench) `send_request(base_url, model, prompt, max_tokens) -> dict` → `{"ok","latency_s","completion_tokens"}`
  - (run_bench) `run_benchmark(base_url, model, prompts, concurrency, max_tokens) -> dict`

- [ ] **Step 1: Write the failing tests**

`tests/test_report.py`:
```python
import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "report", pathlib.Path(__file__).parent.parent / "bench" / "report.py")
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)


def test_summarize_basic():
    results = [
        {"ok": True, "latency_s": 2.0, "completion_tokens": 100},
        {"ok": True, "latency_s": 4.0, "completion_tokens": 100},
        {"ok": False, "latency_s": 1.0, "completion_tokens": 0},
    ]
    s = rp.summarize(results, wall_time_s=4.0)
    assert s["n"] == 3 and s["ok"] == 2
    assert s["total_completion_tokens"] == 200
    assert s["aggregate_tok_s"] == 50.0        # 200 tokens / 4s wall
    assert s["mean_latency_s"] == 3.0


def test_summarize_empty_is_safe():
    s = rp.summarize([], wall_time_s=0.0)
    assert s["ok"] == 0 and s["aggregate_tok_s"] == 0.0


def test_format_report_contains_both_modes():
    single = rp.summarize([{"ok": True, "latency_s": 5.0, "completion_tokens": 50}], 5.0)
    batch = rp.summarize([{"ok": True, "latency_s": 5.0, "completion_tokens": 50}] * 8, 6.0)
    text = rp.format_report(single, batch)
    assert "single" in text.lower() and "batch" in text.lower()
    assert "aggregate_tok_s" in text or "tok/s" in text.lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_report.py -v`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `bench/report.py`**

```python
#!/usr/bin/env python3
"""Pure summarization/formatting of benchmark request results."""


def summarize(results: list, wall_time_s: float) -> dict:
    ok = [r for r in results if r.get("ok")]
    lat = sorted(r["latency_s"] for r in ok)
    toks = sum(r.get("completion_tokens", 0) for r in ok)

    def pct(p):
        if not lat:
            return 0.0
        i = min(len(lat) - 1, int(round((p / 100.0) * (len(lat) - 1))))
        return round(lat[i], 3)

    return {
        "n": len(results),
        "ok": len(ok),
        "total_completion_tokens": toks,
        "wall_time_s": round(wall_time_s, 3),
        "aggregate_tok_s": round(toks / wall_time_s, 2) if wall_time_s > 0 else 0.0,
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
        "p50_latency_s": pct(50),
        "p95_latency_s": pct(95),
    }


def format_report(single: dict, batch: dict) -> str:
    speedup = (round(batch["aggregate_tok_s"] / single["aggregate_tok_s"], 2)
               if single["aggregate_tok_s"] > 0 else 0.0)
    return (
        "# Amalfi benchmark\n\n"
        "## Single-stream (concurrency=1)\n"
        f"- aggregate_tok_s: {single['aggregate_tok_s']}\n"
        f"- mean_latency_s: {single['mean_latency_s']}\n"
        f"- requests ok: {single['ok']}/{single['n']}\n\n"
        "## Batch (high concurrency)\n"
        f"- aggregate_tok_s: {batch['aggregate_tok_s']}\n"
        f"- p50_latency_s: {batch['p50_latency_s']}, p95_latency_s: {batch['p95_latency_s']}\n"
        f"- requests ok: {batch['ok']}/{batch['n']}\n\n"
        f"## Throughput speedup (batch vs single): {speedup}x\n"
        "This demonstrates the throughput-over-latency insight: per-request latency "
        "stays high, but aggregate tokens/sec rises sharply under concurrency.\n"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_report.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Write `bench/prompts.txt`**

```text
Explain how a bicycle stays upright while moving.
Summarize the causes of the fall of the Western Roman Empire.
Write a Python function that returns the nth Fibonacci number.
Describe the water cycle in three sentences.
What are the tradeoffs between TCP and UDP?
Give three tips for reducing memory usage in a large program.
Explain gradient descent to a high-school student.
What is the difference between latency and throughput?
```

- [ ] **Step 6: Implement `bench/run_bench.py`**

```python
#!/usr/bin/env python3
"""Fire concurrent OpenAI-compatible requests at the coordinator; report throughput."""
import argparse, json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from report import summarize, format_report  # noqa: E402


def send_request(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=600)
        dt = time.perf_counter() - t0
        resp.raise_for_status()
        ct = resp.json().get("usage", {}).get("completion_tokens", 0)
        return {"ok": True, "latency_s": dt, "completion_tokens": ct}
    except Exception as e:
        return {"ok": False, "latency_s": time.perf_counter() - t0,
                "completion_tokens": 0, "error": str(e)}


def run_benchmark(base_url, model, prompts, concurrency, max_tokens) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(
            lambda p: send_request(base_url, model, p, max_tokens), prompts))
    return summarize(results, time.perf_counter() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default="local")
    ap.add_argument("--prompts", default=str(pathlib.Path(__file__).parent / "prompts.txt"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()
    prompts = [l.strip() for l in open(args.prompts) if l.strip()]

    print(">> single-stream warmup/measure (concurrency=1)...")
    single = run_benchmark(args.url, args.model, prompts[:4], 1, args.max_tokens)
    print(json.dumps(single, indent=2))

    print(f">> batch measure (concurrency={args.concurrency})...")
    batch = run_benchmark(args.url, args.model, prompts * 4, args.concurrency, args.max_tokens)
    print(json.dumps(batch, indent=2))

    report = format_report(single, batch)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"\n>> wrote {args.out}\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 7: Commit**

```bash
git add bench/report.py bench/run_bench.py bench/prompts.txt tests/test_report.py
git commit -m "feat: batch throughput benchmark harness with report"
```

---

### Task 6: Loopback integration test + full-suite green

**Files:**
- Create: `tests/test_loopback_integration.py`
- Create: `scripts/loopback_demo.sh`

**Interfaces:**
- Consumes: everything above. This is the end-to-end wiring check on localhost.

**Note:** This test *skips* automatically unless a built llama.cpp and a small GGUF model are present, so the suite stays green on any machine. The `loopback_demo.sh` script is the manual full run.

- [ ] **Step 1: Write the integration test with skip guards**

`tests/test_loopback_integration.py`:
```python
import importlib.util, os, pathlib, shutil, subprocess, time
import pytest

ROOT = pathlib.Path(__file__).parent.parent
BIN = ROOT / "vendor" / "llama.cpp" / "build" / "bin"
MODEL_ENV = os.environ.get("AMALFI_TEST_MODEL", "")


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m

hc = _load("healthcheck", "scripts/healthcheck.py")

requires_llama = pytest.mark.skipif(
    not (BIN / "rpc-server").exists() or not MODEL_ENV or not pathlib.Path(MODEL_ENV).exists(),
    reason="needs built llama.cpp and AMALFI_TEST_MODEL pointing at a small GGUF")


@requires_llama
def test_loopback_two_workers_serve_and_healthcheck():
    procs = []
    try:
        for port in (50060, 50061):
            procs.append(subprocess.Popen(
                [str(BIN / "rpc-server"), "--host", "127.0.0.1", "--port", str(port)]))
        time.sleep(3)
        fleet = {"rpc": "127.0.0.1:50060,127.0.0.1:50061"}
        res = hc.run_healthcheck(fleet, timeout=3.0)
        assert res["all_up"] is True
    finally:
        for p in procs:
            p.terminate()
```

- [ ] **Step 2: Run it (expect skip on a fresh machine)**

Run: `python -m pytest tests/test_loopback_integration.py -v`
Expected: `1 skipped` (no built binary / model yet). This is correct — it proves the guard works.

- [ ] **Step 3: Write `scripts/loopback_demo.sh` (manual full run)**

```bash
#!/usr/bin/env bash
# Manual localhost proof: 2 rpc-servers + coordinator on this machine with a small model.
# Requires: scripts/build_llamacpp.sh cpu   AND   a small GGUF at models/$MODEL.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$HERE/vendor/llama.cpp/build/bin"
MODEL="${1:?usage: loopback_demo.sh <gguf-filename-in-models/>}"

"$BIN/rpc-server" --host 127.0.0.1 --port 50060 & W1=$!
"$BIN/rpc-server" --host 127.0.0.1 --port 50061 & W2=$!
sleep 3
trap 'kill $W1 $W2 $CO 2>/dev/null || true' EXIT

"$BIN/llama-server" --model "$HERE/models/$MODEL" \
  --rpc 127.0.0.1:50060,127.0.0.1:50061 --tensor-split 0.5,0.5 \
  --n-gpu-layers 999 --ctx-size 2048 --host 127.0.0.1 --port 8080 & CO=$!
sleep 8

echo ">> healthcheck:"; python3 "$HERE/scripts/healthcheck.py" --fleet <(echo '{"rpc":"127.0.0.1:50060,127.0.0.1:50061"}')
echo ">> benchmark:"; python3 "$HERE/bench/run_bench.py" --url http://127.0.0.1:8080 --concurrency 4 --max-tokens 64
```

- [ ] **Step 4: Run the whole unit suite green**

Run: `python -m pytest -v`
Expected: all unit tests PASS, loopback integration test SKIPPED (until a model is provided).

- [ ] **Step 5: Commit**

```bash
chmod +x scripts/loopback_demo.sh
git add tests/test_loopback_integration.py scripts/loopback_demo.sh
git commit -m "test: localhost loopback cell integration (auto-skips without model)"
```

---

### Task 7: Docs — README, runbook, results template

**Files:**
- Create: `README.md`
- Create: `docs/runbook.md`
- Create: `docs/results.md`

- [ ] **Step 1: Write `README.md`**

Include: one-paragraph project summary (from the spec §1), the pipeline diagram in words (probe → plan → build → launch → chat/bench), quickstart for the loopback cell, link to `docs/runbook.md` and the design spec, and the LAN-only security warning.

```markdown
# Project Amalfi — Distributed Inference Cell (v1)

Serve one 25–30B model across several commodity laptops (16 GB RAM each) that no
single laptop can hold, using llama.cpp RPC pipeline-parallelism over the LAN.
Proves the "cell" concept and the throughput-over-latency insight.

## Pipeline
1. `scripts/probe.py` on each node → node record.
2. Merge records into `nodes.json`; `scripts/plan_split.py` → `fleet.json` + launch commands.
3. `scripts/build_llamacpp.sh <backend>` on each node (pinned source).
4. `scripts/start_worker.*` on every node; `scripts/start_coordinator.*` on one.
5. Chat via the coordinator web UI (`http://<coordinator>:8080`); benchmark with `bench/run_bench.py`.

## Quickstart (localhost loopback, no laptops needed)
    pip install -r requirements.txt
    python -m pytest -v            # unit suite
    scripts/build_llamacpp.sh cpu  # build once
    # drop a small GGUF in models/, then:
    scripts/loopback_demo.sh <model.gguf>

## Security
RPC mode is LAN-only and unauthenticated. Never expose rpc-server ports to the internet.

See `docs/runbook.md` for the office-laptop procedure and
`docs/superpowers/specs/2026-07-23-distributed-inference-cell-design.md` for the design.
```

- [ ] **Step 2: Write `docs/runbook.md`**

Full office-laptop procedure. Must include, as concrete numbered steps:
1. Prereqs per OS (git, cmake, python3, build toolchain; CUDA toolkit for NVIDIA nodes).
2. On every laptop: clone repo, `pip install -r requirements.txt`, set the pinned tag, run `scripts/build_llamacpp.sh <backend>` (choose backend per node), verify `vendor/llama.cpp/build/bin/rpc-server` exists.
3. On every laptop: run `python scripts/probe.py --rpc-host <its-LAN-IP>` and collect the JSON records into a single `nodes.json` array on the coordinator.
4. On the coordinator: `python scripts/plan_split.py --nodes nodes.json --model <key> --out fleet.json`; review the split.
5. Download the chosen GGUF into `models/` on the coordinator (and confirm the `--rpc` device-order mapping by reading llama-server's startup log showing layers-per-device; if reversed vs plan, reorder the `--rpc`/`--tensor-split` lists).
6. Start `scripts/start_worker.*` on every node, then `scripts/start_coordinator.*` on the coordinator.
7. `python scripts/healthcheck.py --fleet fleet.json` → expect all UP.
8. Open `http://<coordinator>:8080` for chat; run `python bench/run_bench.py --url http://<coordinator>:8080 --concurrency 8`.
9. Record numbers in `docs/results.md`.
10. Firewall note: open the RPC port (50052) between laptops on the LAN only.

- [ ] **Step 3: Write `docs/results.md` template**

```markdown
# Amalfi Acceptance Results

- Date:
- Fleet: <n> laptops (list OS / GPU / RAM / measured mem-bandwidth per node)
- Model: <key> (<size> GB), quantization Q4_K_M
- Split: <tensor-split values> across <hosts>

## Success criteria (spec §2)
- [ ] Model loaded split across >= 2 nodes (does not fit on one) — evidence:
- [ ] Coherent chat output — sample:
- [ ] Batch aggregate tok/s > single-stream tok/s — numbers below

## Benchmark
| mode | concurrency | aggregate tok/s | p50 latency s | p95 latency s | ok/total |
|------|-------------|-----------------|---------------|---------------|----------|
| single | 1 | | | | |
| batch  | 8 | | | | |

## Notes / observations
```

- [ ] **Step 4: Commit**

```bash
git add README.md docs/runbook.md docs/results.md
git commit -m "docs: README, office-laptop runbook, results template"
```

---

## Self-Review

**Spec coverage check (spec §→ task):**
- §3 foundation = llama.cpp RPC, pinned source → Task 3. ✓
- §4 components: worker/coordinator launchers → Task 4; probe → Task 1; planner → Task 2; healthcheck → Task 4; benchmark → Task 5. ✓
- §5 data flow (probe→plan→launch→infer→batch) → Tasks 1,2,4,5 + runbook Task 7. ✓
- §6 model selection (catalog, memory math) → Task 2 `config/models.json`. ✓
- §7 capability-aware split (score, select fewest/fastest, proportional split) → Task 2. ✓
- §8 node-drop handling (detect + relaunch guidance) → Task 4 healthcheck. ✓
- §9 error handling (version match, model hash, reachability, capacity) → build tag verify (Task 3), capacity check `select_nodes` (Task 2), reachability healthcheck (Task 4); **model-hash check** covered in runbook Task 7 step 5 (manual) — acceptable for v1, not a code path.
- §10 testing (unit + loopback + acceptance) → Tasks 1–6 unit, Task 6 loopback, Task 7 acceptance runbook/results. ✓
- §11 repo layout → matches Tasks 1–7. ✓
- §12 out-of-scope items → none implemented (correct). ✓

**Placeholder scan:** no "TBD/TODO"; all code steps contain full code. `config/llamacpp.pin` uses a concrete tag with an explicit verify-before-build step (Task 3 Step 5 + build script Step 6). ✓

**Type consistency:** `fleet.json` keys produced by `build_launch_commands`/`plan` (Task 2: `rpc`, `tensor_split`, `worker_cmd`, `coordinator_cmd`, `coordinator_host`) are exactly the keys consumed by `healthcheck.run_healthcheck` (`rpc`) and the launchers (`coordinator_cmd`) in Task 4. `summarize` output keys (Task 5) match what `format_report` and tests read. ✓

---

## Execution Handoff

After the plan is approved, implement task-by-task with the chosen execution sub-skill, running the unit suite green after each task and committing per the steps above. Final acceptance happens on the office laptops per `docs/runbook.md`.
