"""Common adapter interface (plan v4 §16.2).

Contract for every adapter:
    * ``predict`` takes ONLY the raw text string -- never the sample dict, never
      ``dataset_type`` / ``entities`` / ``notes`` / ``canonical_value``
      (plan v4 §8.4.4, §14).
    * returned spans use Python ``str`` code-point offsets, ``[start, end)``, on
      the *original* text (plan v4 §9.1). If the model needs normalized text for
      inference, the adapter must map offsets back (plan v4 §9.2).
    * returned labels are canonical taxonomy labels (``normalize_label``).
    * ``score`` in ``[0, 1]``; use ``1.0`` when the detector is deterministic.

``predict`` output is validated by :meth:`finalize`, which enforces the span
invariant ``text[start:end] == span["text"]`` and drops malformed spans.
"""
from __future__ import annotations

import unicodedata
from typing import Any

from common.taxonomy import normalize_label
from evaluation.spans import Span


class BasePIIAdapter:
    name: str = "base"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})
        self.device = str(self.config.get("device", "cpu"))
        self._loaded = False

    # ---- lifecycle ------------------------------------------------------- #
    def load(self) -> None:
        raise NotImplementedError

    def predict(self, text: str) -> list[Span]:
        raise NotImplementedError

    def unload(self) -> None:
        self._loaded = False

    # ---- helpers ------------------------------------------------------- #
    def finalize(self, text: str, raw_spans: list[dict], *, strict: bool = False) -> list[Span]:
        """Normalize labels, enforce the span invariant, sort, de-duplicate."""
        out: list[Span] = []
        seen: set[tuple[str, int, int]] = set()
        for s in raw_spans:
            try:
                start = int(s["start"])
                end = int(s["end"])
            except (KeyError, TypeError, ValueError):
                if strict:
                    raise
                continue
            if not (0 <= start < end <= len(text)):
                if strict:
                    raise ValueError(f"span out of range: {s!r}")
                continue
            label = normalize_label(str(s.get("label", "")))
            surface = text[start:end]
            reported = s.get("text")
            if reported is not None:
                if unicodedata.normalize("NFC", str(reported)) != unicodedata.normalize(
                    "NFC", surface
                ):
                    if strict:
                        raise ValueError(
                            f"span invariant broken: text[{start}:{end}]={surface!r} "
                            f"!= {reported!r}"
                        )
                    # trust offsets, keep the true surface
            key = (label, start, end)
            if key in seen:
                continue
            seen.add(key)
            score = s.get("score", 1.0)
            try:
                score = max(0.0, min(1.0, float(score)))
            except (TypeError, ValueError):
                score = 1.0
            out.append(
                {"text": surface, "label": label, "start": start, "end": end, "score": score}
            )
        out.sort(key=lambda x: (x["start"], x["end"], x["label"]))
        return out
