import importlib.util, pathlib, pytest
spec = importlib.util.spec_from_file_location(
    "merge_nodes", pathlib.Path(__file__).parent.parent / "scripts" / "merge_nodes.py")
mn = importlib.util.module_from_spec(spec); spec.loader.exec_module(mn)


def rec(host, port=50052):
    return {"hostname": host, "rpc_host": host, "rpc_port": port, "gpu": {"type": "none"}}


def test_merge_flattens_objects_and_arrays():
    out = mn.merge_node_objs([rec("a"), [rec("b"), rec("c")]])
    hosts = sorted(r["rpc_host"] for r in out)
    assert hosts == ["a", "b", "c"]


def test_merge_dedups_by_host_port_keeping_last():
    first = rec("a"); first["free_ram_gb"] = 10.0
    second = rec("a"); second["free_ram_gb"] = 12.0
    out = mn.merge_node_objs([first, second])
    assert len(out) == 1
    assert out[0]["free_ram_gb"] == 12.0


def test_merge_rejects_bad_type():
    with pytest.raises(ValueError):
        mn.merge_node_objs(["not-a-record"])
