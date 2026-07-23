#!/usr/bin/env python3
"""Amalfi capability probe. Run on each node; emits a JSON node record."""
import argparse, json, platform, shutil, socket, subprocess, sys, time
import numpy as np
import psutil

GPU_KEYS = ("type", "name", "vram_gb")
BACKEND_FOR_GPU = {"cuda": "cuda", "metal": "metal", "vulkan": "vulkan", "none": "cpu"}


def detect_backend() -> str:
    """Map this machine's GPU type to the llama.cpp build backend."""
    return BACKEND_FOR_GPU.get(detect_gpu()["type"], "cpu")


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
    ap.add_argument("--print-backend", action="store_true",
                    help="print the llama.cpp build backend for this machine and exit")
    ap.add_argument("--print-ip", action="store_true",
                    help="print this machine's primary LAN IP and exit")
    args = ap.parse_args()
    if args.print_backend:
        print(detect_backend())
        return 0
    if args.print_ip:
        print(_primary_ip())
        return 0
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
