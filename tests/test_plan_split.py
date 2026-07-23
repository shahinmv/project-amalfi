import importlib.util, pathlib, pytest
spec = importlib.util.spec_from_file_location(
    "plan_split", pathlib.Path(__file__).parent.parent / "scripts" / "plan_split.py")
ps = importlib.util.module_from_spec(spec); spec.loader.exec_module(ps)


def node(host, bw, free, gpu="none", vram=0.0, total=None):
    return {"rpc_host": host, "mem_bandwidth_gbps": bw, "free_ram_gb": free,
            "total_ram_gb": total if total is not None else free,
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


def test_capped_budget_limited_by_cap_and_free_ram():
    # cap 4, overhead 1 -> budget min(4, capacity) - 1
    assert ps.capped_budget_gb(node("a", 40, 16.0), 4.0, 1.0) == 3.0   # capped at 4
    assert ps.capped_budget_gb(node("b", 40, 2.7), 4.0, 1.0) == 1.7    # limited by free RAM


def test_capped_split_keeps_every_node_under_cap():
    nodes = [node(f"n{i}", 40, 16.0) for i in range(7)]
    model_size = 18.6
    sel = ps.select_nodes_capped(nodes, model_size, cap_gb=4.0, overhead_gb=1.0)
    split = ps.compute_tensor_split_capped(sel, 4.0, 1.0)
    assert abs(sum(split) - 1.0) < 1e-6
    for s in split:
        assert s * model_size <= 3.0 + 1e-6   # weights per node never exceed the 3GB budget


def test_capped_raises_when_too_few_nodes():
    nodes = [node(f"n{i}", 40, 16.0) for i in range(4)]  # 4 * 3GB = 12 < 18.6
    with pytest.raises(ValueError):
        ps.select_nodes_capped(nodes, 18.6, cap_gb=4.0, overhead_gb=1.0)


def test_dynamic_budget_half_of_total_bounded_by_free():
    # total 16, free 12, fraction .5 -> min(8, 12) - 1 overhead = 7
    assert ps.dynamic_budget_gb(node("a", 40, 12.0, total=16.0), 0.5, 1.0) == 7.0
    # busy node: total 16, free 3 -> min(8, 3) - 1 = 2  (bounded by free RAM, not the half)
    assert ps.dynamic_budget_gb(node("b", 40, 3.0, total=16.0), 0.5, 1.0) == 2.0


def test_plan_default_uses_half_of_ram():
    nodes = [node(f"192.168.0.{i}", 40, 12.0, total=16.0) for i in range(4)]
    catalog = {"m": {"gguf": "m.gguf", "size_gb": 18.6, "ctx_size": 4096}}
    fleet = ps.plan(nodes, "m", catalog)  # default ram_fraction=0.5
    assert fleet["cap_policy"] == "ram_fraction=0.5"
    for e in fleet["est_ram_per_node"]:
        assert e["total_gb"] <= 0.5 * 16.0 + 1e-6   # never more than half of a 16GB laptop


def test_plan_capped_reports_per_node_ram_under_cap():
    nodes = [node(f"192.168.0.{i}", 40, 16.0) for i in range(7)]
    catalog = {"m": {"gguf": "m.gguf", "size_gb": 18.6, "ctx_size": 4096}}
    fleet = ps.plan(nodes, "m", catalog, max_ram_gb=4.0, ram_overhead_gb=1.0)
    assert all(e["total_gb"] <= 4.0 + 1e-6 for e in fleet["est_ram_per_node"])


def test_build_launch_commands_shape():
    sel = [node("192.168.1.10", 60, 12), node("192.168.1.11", 20, 12)]
    model = {"gguf": "m.gguf", "ctx_size": 4096}
    cmd = ps.build_launch_commands(sel, model, rpc_port=50052, api_port=8080)
    assert cmd["rpc"] == "192.168.1.10:50052,192.168.1.11:50052"
    assert "llama-server" in cmd["coordinator_cmd"]
    assert "--tensor-split" in cmd["coordinator_cmd"]
    assert "rpc-server" in cmd["worker_cmd"]
    assert cmd["coordinator_host"] == "192.168.1.10"
