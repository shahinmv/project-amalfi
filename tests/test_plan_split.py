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
