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
