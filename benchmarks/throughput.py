"""Throughput (plan v4 §16.4)."""
from __future__ import annotations


def throughput_from_latency(latency_report: dict) -> dict:
    total_s = latency_report.get("total_ms", 0.0) / 1000.0
    n = latency_report.get("count", 0)
    return {
        "samples": n,
        "total_seconds": total_s,
        "samples_per_second": (n / total_s) if total_s > 0 else 0.0,
    }
