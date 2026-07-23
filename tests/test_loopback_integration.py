import importlib.util, os, pathlib, subprocess, time
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
