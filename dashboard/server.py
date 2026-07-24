#!/usr/bin/env python3
"""Amalfi live dashboard — visualizes real per-node data transfer in the running cell.

Serves index.html and a /state JSON endpoint. A background thread samples the
coordinator's per-connection byte counters (via `nettop -x`, which is slow ~5s) and
caches live bytes/sec to each worker; /state returns the cache instantly so the UI
animates the actual relay traffic. macOS only.

Run:  ./.venv/bin/python dashboard/server.py [--port 8090] [--api http://127.0.0.1:8080]
Open: http://<this-mac-ip>:<port>
"""
import argparse, json, os, re, subprocess, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS = {
    "127.0.0.1":     ("This Mac (worker)", "metal"),
    "192.168.1.89":  ("Mac Mini",          "metal"),
    "192.168.1.140": ("Laptop 1",          "cpu"),
    "192.168.1.236": ("Laptop 2",          "cpu"),
}
RPC_PORT = "50052"
_rates, _lock = {}, threading.Lock()


def _coord_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "llama-server"], text=True).split()
        return int(out[0]) if out else None
    except Exception:
        return None


def _sample(pid):
    """remote_ip -> (bytes_in, bytes_out) for the coordinator's RPC connections."""
    res = {}
    try:
        out = subprocess.check_output(
            ["nettop", "-x", "-L", "1", "-m", "tcp", "-p", str(pid)],
            text=True, timeout=12, stderr=subprocess.DEVNULL)
    except Exception:
        return res
    for line in out.splitlines():
        f = line.split(",")
        if len(f) < 6:
            continue
        m = re.search(r"<->(\d+\.\d+\.\d+\.\d+):" + RPC_PORT, f[1])
        if not m:
            continue
        try:
            res[m.group(1)] = (int(f[4]), int(f[5]))
        except ValueError:
            continue
    return res


def _sampler_loop():
    prev, prev_t = {}, None
    while True:
        pid = _coord_pid()
        conns = _sample(pid) if pid else {}
        now = time.time()
        dt = (now - prev_t) if prev_t else None
        new = {}
        for ip, (bi, bo) in conns.items():
            pbi, pbo = prev.get(ip, (bi, bo))
            name, backend = LABELS.get(ip, (ip, "?"))
            new[ip] = {
                "ip": ip, "name": name, "backend": backend,
                "in_bps": max(0.0, (bi - pbi) / dt) if dt and dt > 0 else 0.0,
                "out_bps": max(0.0, (bo - pbo) / dt) if dt and dt > 0 else 0.0,
                "tot_in": bi, "tot_out": bo,
            }
        with _lock:
            _rates.clear()
            _rates.update(new)
        prev, prev_t = conns, now
        time.sleep(1)


def _state(api):
    with _lock:
        nodes = list(_rates.values())
    nodes.sort(key=lambda n: (n["ip"] != "127.0.0.1", n["ip"]))
    healthy = False
    try:
        urllib.request.urlopen(api + "/health", timeout=2)
        healthy = True
    except Exception:
        pass
    return {"healthy": healthy, "running": _coord_pid() is not None,
            "nodes": nodes, "ts": time.time()}


def make_handler(api):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/state"):
                body = json.dumps(_state(api)).encode()
                ctype = "application/json"
            else:
                try:
                    with open(os.path.join(HERE, "index.html"), "rb") as fh:
                        body = fh.read()
                except OSError:
                    body = b"index.html missing"
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--api", default="http://127.0.0.1:8080")
    args = ap.parse_args()
    threading.Thread(target=_sampler_loop, daemon=True).start()
    srv = ThreadingHTTPServer(("0.0.0.0", args.port), make_handler(args.api))
    print(f">> Amalfi dashboard on http://0.0.0.0:{args.port} (coordinator API: {args.api})")
    srv.serve_forever()


if __name__ == "__main__":
    main()
