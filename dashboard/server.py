#!/usr/bin/env python3
"""Amalfi live dashboard — real per-node data transfer + cell metadata + a token-rate probe.

Endpoints:
  /            index.html
  /state       JSON: live per-node bytes/sec (sampled via `nettop -x`, background thread),
               enriched with cell metadata (dashboard/cell.json), health, cell summary.
  /generate    runs one short generation on the coordinator and returns measured tok/s.

macOS only (uses nettop). Run:
  ./.venv/bin/python dashboard/server.py [--port 8090] [--api http://127.0.0.1:8080]
"""
import argparse, json, os, re, subprocess, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RPC_PORT = "50052"
_rates, _lock = {}, threading.Lock()
_FALLBACK = {"127.0.0.1": ("This Mac (worker)", "metal")}


def _cell():
    try:
        with open(os.path.join(HERE, "cell.json")) as f:
            return json.load(f)
    except Exception:
        return {"nodes": {}}


def _coord_pid():
    try:
        out = subprocess.check_output(["pgrep", "-f", "llama-server"], text=True).split()
        return int(out[0]) if out else None
    except Exception:
        return None


def _sample(pid):
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
            new[ip] = {
                "ip": ip,
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
    cell = _cell()
    meta = cell.get("nodes", {})
    with _lock:
        live = {ip: dict(v) for ip, v in _rates.items()}
    nodes = []
    for ip, v in live.items():
        m = meta.get(ip, {})
        name, backend = _FALLBACK.get(ip, (ip, "?"))
        v.update({
            "name": m.get("name", name), "backend": m.get("backend", backend),
            "pct": m.get("pct"), "cores": m.get("cores"), "ram_gb": m.get("ram_gb"),
            "bw": m.get("bw"),
        })
        nodes.append(v)
    nodes.sort(key=lambda n: (n["ip"] != "127.0.0.1", n["ip"]))
    healthy = False
    try:
        urllib.request.urlopen(api + "/health", timeout=2)
        healthy = True
    except Exception:
        pass
    pooled = sum(n.get("ram_gb") or 0 for n in nodes)
    return {
        "healthy": healthy, "running": _coord_pid() is not None, "nodes": nodes,
        "cell": {"model": cell.get("model"), "quant": cell.get("quant"),
                 "model_size_gb": cell.get("model_size_gb"), "ctx_size": cell.get("ctx_size"),
                 "node_count": len(nodes), "pooled_ram_gb": round(pooled, 1)},
        "ts": time.time(),
    }


def _generate(api):
    payload = json.dumps({
        "model": "local",
        "messages": [{"role": "user", "content": "Write two sentences about distributed systems."}],
        "max_tokens": 64, "temperature": 0.4,
    }).encode()
    req = urllib.request.Request(api + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.load(r)
        dt = time.time() - t0
        toks = d.get("usage", {}).get("completion_tokens", 0)
        return {"ok": True, "tok_s": round(toks / dt, 2) if dt > 0 else 0.0,
                "tokens": toks, "latency_s": round(dt, 2),
                "text": d["choices"][0]["message"]["content"]}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def make_handler(api):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, body, ctype="application/json"):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/state"):
                self._send(json.dumps(_state(api)).encode())
            elif self.path.startswith("/generate"):
                self._send(json.dumps(_generate(api)).encode())
            else:
                try:
                    with open(os.path.join(HERE, "index.html"), "rb") as fh:
                        body = fh.read()
                except OSError:
                    body = b"index.html missing"
                self._send(body, "text/html; charset=utf-8")
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
