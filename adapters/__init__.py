"""Detector adapters (plan v4 §16.2).

Public factory: :func:`get_adapter`.
"""
from __future__ import annotations

from adapters.base import BasePIIAdapter

_REGISTRY = {
    "regex": ("adapters.regex_adapter", "RegexAdapter"),
    "stanza": ("adapters.stanza_adapter", "StanzaAdapter"),
    "berturk": ("adapters.berturk_adapter", "BERTurkAdapter"),
    "gliner_tr": ("adapters.gliner_tr_adapter", "GlinerTrAdapter"),
    "gliner_edge": ("adapters.gliner_edge_adapter", "GlinerEdgeAdapter"),
    "gliner_stream": ("adapters.gliner_stream_adapter", "GlinerStreamAdapter"),
}


def get_adapter(name: str, config: dict) -> BasePIIAdapter:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown adapter '{name}'. Known: {sorted(_REGISTRY)}")
    module_path, cls_name = _REGISTRY[name]
    module = __import__(module_path, fromlist=[cls_name])
    return getattr(module, cls_name)(config)
