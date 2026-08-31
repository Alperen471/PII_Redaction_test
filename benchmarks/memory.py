"""Process + GPU memory sampling (plan v4 §13)."""
from __future__ import annotations

import os


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return 0.0


def _cuda_mem_mb(device: str) -> dict:
    if not device.startswith("cuda"):
        return {}
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "vram_allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "vram_reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "vram_max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    except Exception:
        return {}


def reset_cuda_peak(device: str) -> None:
    if not device.startswith("cuda"):
        return
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


class MemoryTracker:
    """Snapshot RSS/VRAM at baseline, after load, and after the run."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = device
        self.samples: dict[str, dict] = {}

    def snapshot(self, tag: str) -> None:
        self.samples[tag] = {"rss_mb": _rss_mb(), **_cuda_mem_mb(self.device)}

    def report(self) -> dict:
        base = self.samples.get("baseline", {}).get("rss_mb", 0.0)
        after_load = self.samples.get("after_load", {}).get("rss_mb", 0.0)
        after_run = self.samples.get("after_run", {}).get("rss_mb", 0.0)
        out = {
            "snapshots": self.samples,
            "ram_load_delta_mb": max(0.0, after_load - base),
            "ram_peak_mb": max(base, after_load, after_run),
        }
        cuda_run = self.samples.get("after_run", {})
        if "vram_max_allocated_mb" in cuda_run:
            out["vram_mb"] = cuda_run["vram_max_allocated_mb"]
        return out
