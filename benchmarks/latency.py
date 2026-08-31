"""Per-sample inference latency (plan v4 §11).

Locked latency contract:
    * timed unit  : ``adapter.predict(text)`` wall time only
    * clock       : ``time.perf_counter_ns()``
    * batch size  : 1
    * warm-up     : first N runs discarded (plan v4 §11)
    * CUDA        : ``torch.cuda.synchronize()`` immediately before and after
                    each timed call
    * EXCLUDED    : model load, dataset read, evaluation, tokenization->safe
                    text, metric computation, result writing
"""
from __future__ import annotations

import time
from typing import Callable, Sequence


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = pct / 100.0 * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def cuda_sync(device: str):
    return _cuda_sync(device)


def summarize(per_sample_ms: Sequence[float], *, warmup: int = 0) -> dict:
    """Build the latency report from an already-collected list of per-sample ms."""
    vals = list(per_sample_ms)
    total_ms = sum(vals)
    return {
        "count": len(vals),
        "warmup_runs": warmup,
        "batch_size": 1,
        "clock": "time.perf_counter_ns",
        "avg_ms": total_ms / len(vals) if vals else 0.0,
        "p50_ms": _percentile(vals, 50),
        "p95_ms": _percentile(vals, 95),
        "p99_ms": _percentile(vals, 99),
        "min_ms": min(vals) if vals else 0.0,
        "max_ms": max(vals) if vals else 0.0,
        "total_ms": total_ms,
        "per_sample_ms": vals,
    }


def _cuda_sync(device: str):
    if not device.startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def measure_latency(
    predict: Callable[[str], object],
    texts: Sequence[str],
    *,
    device: str = "cpu",
    warmup: int = 10,
) -> dict:
    """Run ``predict`` over every text once (batch=1) and summarize latency.

    Returns milliseconds. ``per_sample_ms`` is aligned with ``texts`` order
    (warm-up entries excluded).
    """
    warmup = max(0, min(warmup, len(texts)))
    for i in range(warmup):
        predict(texts[i])

    per_sample_ms: list[float] = []
    for text in texts:
        _cuda_sync(device)
        t0 = time.perf_counter_ns()
        predict(text)
        _cuda_sync(device)
        t1 = time.perf_counter_ns()
        per_sample_ms.append((t1 - t0) / 1e6)

    total_ms = sum(per_sample_ms)
    return {
        "count": len(per_sample_ms),
        "warmup_runs": warmup,
        "batch_size": 1,
        "clock": "time.perf_counter_ns",
        "avg_ms": total_ms / len(per_sample_ms) if per_sample_ms else 0.0,
        "p50_ms": _percentile(per_sample_ms, 50),
        "p95_ms": _percentile(per_sample_ms, 95),
        "p99_ms": _percentile(per_sample_ms, 99),
        "min_ms": min(per_sample_ms) if per_sample_ms else 0.0,
        "max_ms": max(per_sample_ms) if per_sample_ms else 0.0,
        "total_ms": total_ms,
        "per_sample_ms": per_sample_ms,
    }
