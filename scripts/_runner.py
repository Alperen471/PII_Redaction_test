"""Shared inference driver for the semantic and system runners (plan v5 §15, §25, §26).

Handles: model load timing, cold start, warm-up (discarded), and the measured
batch=1 pass that captures both predictions and per-sample latency under the
locked latency contract (§15).
"""
from __future__ import annotations

import time
from typing import Sequence

from benchmarks.latency import cuda_sync, summarize
from benchmarks.memory import MemoryTracker, reset_cuda_peak


def run_inference(
    adapter,
    dataset: Sequence[dict],
    *,
    device: str,
    warmup: int,
) -> dict:
    texts = [s["text"] for s in dataset]

    mem = MemoryTracker(device)
    mem.snapshot("baseline")
    reset_cuda_peak(device)

    t0 = time.perf_counter_ns()
    adapter.load()
    model_load_time_ms = (time.perf_counter_ns() - t0) / 1e6
    mem.snapshot("after_load")

    # cold start: first inference on a freshly loaded model (plan v5 §26)
    t_cs = time.perf_counter_ns()
    if texts:
        adapter.predict(texts[0])
    first_inference_ms = (time.perf_counter_ns() - t_cs) / 1e6

    # warm-up (discarded, plan v5 §15)
    n_warm = min(max(warmup, 0), len(texts))
    for i in range(n_warm):
        adapter.predict(texts[i])

    # measured pass: batch=1, capture predictions + latency together
    raw_records: list[dict] = []
    preds_by_id: dict[str, list] = {}
    per_sample_ms: list[float] = []

    for sample, text in zip(dataset, texts):
        cuda_sync(device)
        ts = time.perf_counter_ns()
        preds = adapter.predict(text)
        cuda_sync(device)
        latency_ms = (time.perf_counter_ns() - ts) / 1e6

        per_sample_ms.append(latency_ms)
        preds_by_id[sample["id"]] = preds
        raw_records.append(
            {
                "id": sample["id"],
                "dataset_type": sample.get("dataset_type"),
                "predictions": preds,
                "latency_ms": latency_ms,
            }
        )

    mem.snapshot("after_run")

    return {
        "raw_records": raw_records,
        "preds_by_id": preds_by_id,
        "latency": summarize(per_sample_ms, warmup=n_warm),
        "cold_start": {
            "model_load_time_ms": model_load_time_ms,
            "first_inference_ms": first_inference_ms,
            "time_to_first_inference_ms": model_load_time_ms + first_inference_ms,
        },
        "memory": mem.report(),
    }
