import importlib.util, pathlib, socket, threading
spec = importlib.util.spec_from_file_location(
    "healthcheck", pathlib.Path(__file__).parent.parent / "scripts" / "healthcheck.py")
hc = importlib.util.module_from_spec(spec); spec.loader.exec_module(hc)


def _accept_once(srv):
    """Accept one connection, swallowing the abort raised when srv is closed."""
    try:
        conn, _ = srv.accept()
        conn.close()
    except OSError:
        pass


def _listening_socket():
    srv = socket.socket(); srv.bind(("127.0.0.1", 0)); srv.listen(1)
    threading.Thread(target=_accept_once, args=(srv,), daemon=True).start()
    return srv, srv.getsockname()[1]


def test_parse_rpc():
    assert hc.parse_rpc("10.0.0.1:50052,10.0.0.2:50052") == [("10.0.0.1", 50052), ("10.0.0.2", 50052)]


def test_check_endpoint_true_on_open_socket():
    srv, port = _listening_socket()
    try:
        assert hc.check_endpoint("127.0.0.1", port, timeout=2.0) is True
    finally:
        srv.close()


def test_check_endpoint_false_on_closed_port():
    assert hc.check_endpoint("127.0.0.1", 1, timeout=1.0) is False


def test_run_healthcheck_aggregates():
    srv, port = _listening_socket()
    try:
        fleet = {"rpc": f"127.0.0.1:{port},127.0.0.1:1"}
        res = hc.run_healthcheck(fleet, timeout=1.0)
        assert res["all_up"] is False
        assert res["nodes"][0]["up"] is True and res["nodes"][1]["up"] is False
    finally:
        srv.close()
