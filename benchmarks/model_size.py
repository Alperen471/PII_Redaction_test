"""Best-effort on-disk model size (plan v5 §5.3, §19).

Returns megabytes, or ``None`` when the location cannot be determined. Never
raises -- size is a reporting nicety, not a gate.
"""
from __future__ import annotations

import os
from pathlib import Path

_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".onnx", ".h5", ".ckpt", ".msgpack"}


def _dir_size_mb(path: Path, *, weights_only: bool = False) -> float:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            if weights_only and Path(f).suffix.lower() not in _WEIGHT_SUFFIXES:
                continue
            try:
                total += (Path(root) / f).stat().st_size
            except OSError:
                pass
    return total / (1024 * 1024)


def _hf_hub_dir(repo_id: str) -> Path | None:
    cache = os.environ.get("HF_HOME")
    hub = Path(cache) / "hub" if cache else Path.home() / ".cache" / "huggingface" / "hub"
    folder = hub / ("models--" + repo_id.replace("/", "--"))
    if not folder.is_dir():
        return None
    snaps = folder / "snapshots"
    if snaps.is_dir():
        subs = sorted(snaps.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if subs:
            return subs[0]
    return folder


def model_size_mb(model_name: str | None, *, local_path: str | None = None) -> float | None:
    try:
        if local_path and Path(local_path).exists():
            return round(_dir_size_mb(Path(local_path)), 2)
        if not model_name:
            return None
        p = Path(model_name)
        if p.exists():
            return round(_dir_size_mb(p), 2)
        hub = _hf_hub_dir(model_name)
        if hub is not None:
            return round(_dir_size_mb(hub, weights_only=True), 2)
    except Exception:
        return None
    return None
