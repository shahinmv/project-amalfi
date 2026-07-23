#!/usr/bin/env python3
"""Pure summarization/formatting of benchmark request results."""


def summarize(results: list, wall_time_s: float) -> dict:
    ok = [r for r in results if r.get("ok")]
    lat = sorted(r["latency_s"] for r in ok)
    toks = sum(r.get("completion_tokens", 0) for r in ok)

    def pct(p):
        if not lat:
            return 0.0
        i = min(len(lat) - 1, int(round((p / 100.0) * (len(lat) - 1))))
        return round(lat[i], 3)

    return {
        "n": len(results),
        "ok": len(ok),
        "total_completion_tokens": toks,
        "wall_time_s": round(wall_time_s, 3),
        "aggregate_tok_s": round(toks / wall_time_s, 2) if wall_time_s > 0 else 0.0,
        "mean_latency_s": round(sum(lat) / len(lat), 3) if lat else 0.0,
        "p50_latency_s": pct(50),
        "p95_latency_s": pct(95),
    }


def format_report(single: dict, batch: dict) -> str:
    speedup = (round(batch["aggregate_tok_s"] / single["aggregate_tok_s"], 2)
               if single["aggregate_tok_s"] > 0 else 0.0)
    if speedup >= 1.05:
        verdict = (f"Batch throughput is {speedup}x single-stream: concurrency raises aggregate "
                   f"tokens/sec (the throughput-over-latency win) — the nodes had spare parallel "
                   f"capacity to fill.")
    elif speedup <= 0.95:
        verdict = (f"Batch throughput is {speedup}x single-stream: concurrency did NOT help here — "
                   f"aggregate tokens/sec fell. This is expected when nodes are already saturated by "
                   f"a single stream (CPU-only / low memory bandwidth): extra concurrent requests add "
                   f"contention and KV-cache pressure instead of filling idle compute. The batch win "
                   f"shows up on nodes with parallel headroom (GPUs / higher bandwidth).")
    else:
        verdict = f"Batch and single-stream throughput are within noise ({speedup}x)."
    return (
        "# Amalfi benchmark\n\n"
        "## Single-stream (concurrency=1)\n"
        f"- aggregate_tok_s: {single['aggregate_tok_s']}\n"
        f"- mean_latency_s: {single['mean_latency_s']}\n"
        f"- requests ok: {single['ok']}/{single['n']}\n\n"
        "## Batch (high concurrency)\n"
        f"- aggregate_tok_s: {batch['aggregate_tok_s']}\n"
        f"- p50_latency_s: {batch['p50_latency_s']}, p95_latency_s: {batch['p95_latency_s']}\n"
        f"- requests ok: {batch['ok']}/{batch['n']}\n\n"
        f"## Throughput speedup (batch vs single): {speedup}x\n"
        f"{verdict}\n"
    )
