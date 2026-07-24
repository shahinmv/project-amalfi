import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "supervisor", pathlib.Path(__file__).parent.parent / "scripts" / "supervisor.py")
sup = importlib.util.module_from_spec(spec); spec.loader.exec_module(sup)


def test_no_reform_when_nothing_online():
    assert sup.needs_reform(active=set(), reachable=set(), healthy=False, loading=False) is False


def test_forms_when_nodes_appear_and_none_active():
    assert sup.needs_reform(set(), {"a", "b"}, healthy=False, loading=False) is True


def test_reforms_when_a_node_drops():
    assert sup.needs_reform({"a", "b", "c"}, {"a", "b"}, healthy=True, loading=False) is True


def test_reforms_when_a_node_joins():
    assert sup.needs_reform({"a", "b"}, {"a", "b", "c"}, healthy=True, loading=False) is True


def test_reforms_when_coordinator_dead_and_not_loading():
    assert sup.needs_reform({"a", "b"}, {"a", "b"}, healthy=False, loading=False) is True


def test_no_reform_while_loading_even_if_unhealthy():
    # during the post-launch grace period, an unhealthy coordinator is just still loading
    assert sup.needs_reform({"a", "b"}, {"a", "b"}, healthy=False, loading=True) is False


def test_stable_healthy_cell_is_left_alone():
    assert sup.needs_reform({"a", "b"}, {"a", "b"}, healthy=True, loading=False) is False
