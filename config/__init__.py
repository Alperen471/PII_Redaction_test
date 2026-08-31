"""Configuration loading (plan v4 §16.7)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_YAML = _ROOT / "config" / "models.yaml"
<<<<<<< HEAD
DEFAULT_SYSTEMS_YAML = _ROOT / "config" / "systems.yaml"
=======
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53


def load_config(path: str | Path = DEFAULT_MODELS_YAML) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("dataset", {})
    cfg.setdefault("run", {})
    cfg.setdefault("models", {})
    return cfg


<<<<<<< HEAD
def load_systems_config(path: str | Path = DEFAULT_SYSTEMS_YAML) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg.setdefault("structured_labels", [])
    cfg.setdefault("semantic_labels", [])
    cfg.setdefault("semantic_gap_labels", [])
    cfg.setdefault("systems", {})
    return cfg


=======
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
def model_config(cfg: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in cfg["models"]:
        raise KeyError(
            f"Model '{name}' not in config. Available: {sorted(cfg['models'])}"
        )
    entry = dict(cfg["models"][name])
    entry.setdefault("adapter", name)
    return entry
