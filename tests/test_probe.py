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
