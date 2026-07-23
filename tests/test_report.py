import importlib.util, pathlib
spec = importlib.util.spec_from_file_location(
    "report", pathlib.Path(__file__).parent.parent / "bench" / "report.py")
rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)


def test_summarize_basic():
    results = [
        {"ok": True, "latency_s": 2.0, "completion_tokens": 100},
        {"ok": True, "latency_s": 4.0, "completion_tokens": 100},
        {"ok": False, "latency_s": 1.0, "completion_tokens": 0},
    ]
    s = rp.summarize(results, wall_time_s=4.0)
    assert s["n"] == 3 and s["ok"] == 2
    assert s["total_completion_tokens"] == 200
    assert s["aggregate_tok_s"] == 50.0        # 200 tokens / 4s wall
    assert s["mean_latency_s"] == 3.0


def test_summarize_empty_is_safe():
    s = rp.summarize([], wall_time_s=0.0)
    assert s["ok"] == 0 and s["aggregate_tok_s"] == 0.0


def test_format_report_contains_both_modes():
    single = rp.summarize([{"ok": True, "latency_s": 5.0, "completion_tokens": 50}], 5.0)
    batch = rp.summarize([{"ok": True, "latency_s": 5.0, "completion_tokens": 50}] * 8, 6.0)
    text = rp.format_report(single, batch)
    assert "single" in text.lower() and "batch" in text.lower()
    assert "aggregate_tok_s" in text or "tok/s" in text.lower()
