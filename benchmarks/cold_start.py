"""Cold start / model load timing (plan v4 §12).

``model_load_time_ms``       : construct -> load() returns
``time_to_first_inference_ms``: construct -> first predict() returns
Both are kept OUT of the per-sample latency figure (plan v4 §11, §14).
"""
from __future__ import annotations

import time
from typing import Callable


def measure_cold_start(
    build_adapter: Callable[[], object],
    sample_text: str,
) -> dict:
    t0 = time.perf_counter_ns()
    adapter = build_adapter()
    t_built = time.perf_counter_ns()
    adapter.load()
    t_loaded = time.perf_counter_ns()
    adapter.predict(sample_text)
    t_first = time.perf_counter_ns()
    return {
        "adapter": getattr(adapter, "name", "unknown"),
        "construct_ms": (t_built - t0) / 1e6,
        "model_load_time_ms": (t_loaded - t_built) / 1e6,
        "first_inference_ms": (t_first - t_loaded) / 1e6,
        "time_to_first_inference_ms": (t_first - t0) / 1e6,
        "_adapter_obj": adapter,
    }
