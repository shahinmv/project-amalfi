#!/usr/bin/env python3
"""Fire concurrent OpenAI-compatible requests at the coordinator; report throughput."""
import argparse, json, pathlib, sys, time
from concurrent.futures import ThreadPoolExecutor
import requests

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from report import summarize, format_report  # noqa: E402


def send_request(base_url: str, model: str, prompt: str, max_tokens: int) -> dict:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": 0.7}
    t0 = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=600)
        dt = time.perf_counter() - t0
        resp.raise_for_status()
        ct = resp.json().get("usage", {}).get("completion_tokens", 0)
        return {"ok": True, "latency_s": dt, "completion_tokens": ct}
    except Exception as e:
        return {"ok": False, "latency_s": time.perf_counter() - t0,
                "completion_tokens": 0, "error": str(e)}


def run_benchmark(base_url, model, prompts, concurrency, max_tokens) -> dict:
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        results = list(ex.map(
            lambda p: send_request(base_url, model, p, max_tokens), prompts))
    return summarize(results, time.perf_counter() - t0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--model", default="local")
    ap.add_argument("--prompts", default=str(pathlib.Path(__file__).parent / "prompts.txt"))
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--out", default="docs/results.md")
    args = ap.parse_args()
    prompts = [l.strip() for l in open(args.prompts) if l.strip()]

    print(">> single-stream warmup/measure (concurrency=1)...")
    single = run_benchmark(args.url, args.model, prompts[:4], 1, args.max_tokens)
    print(json.dumps(single, indent=2))

    print(f">> batch measure (concurrency={args.concurrency})...")
    batch = run_benchmark(args.url, args.model, prompts * 4, args.concurrency, args.max_tokens)
    print(json.dumps(batch, indent=2))

    report = format_report(single, batch)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"\n>> wrote {args.out}\n")
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
