"""Composite (system) adapter for the System-Level PII benchmark (plan v5 §21, §22).

Parallel architecture:

        TEXT
     ┌───┴────┐
     ▼        ▼
   REGEX    GLiNER
     │        │
  structured  semantic + structured-gap
     └───┬────┘
         ▼   label-domain-priority MERGE

Merge policy (label-domain priority):
    * Regex owns the character territory of every span it produced. Any GLiNER
      span (semantic OR structured-domain) that overlaps that territory is
      dropped -- regex is authoritative for structured PII, so a GLiNER
      "PERSON [15,26)" over a regex "CUSTOMER_ID [15,26)" disappears.
    * GLiNER spans that fall in the gaps regex left are kept (gap-fill for the
      structured labels regex cannot express, and for PERSON/LOCATION/ADDRESS).
    * identical (label, start, end) from both sides -> keep the higher score.
Evaluation runs on this merged set.
"""
from __future__ import annotations

from typing import Iterable

from adapters.base import BasePIIAdapter
from evaluation.spans import Span, covered_intervals, overlaps_intervals


def _score(p: Span) -> float:
    try:
        return float(p.get("score", 1.0))
    except (TypeError, ValueError):
        return 1.0


def merge_predictions(structured: Iterable[Span], semantic: Iterable[Span]) -> list[Span]:
    """Label-domain-priority merge: regex territory wins, GLiNER fills the gaps."""
    structured = list(structured)
    struct_keys = {(p["label"], int(p["start"]), int(p["end"])) for p in structured}
    territory = covered_intervals(structured)

    kept: list[Span] = list(structured)
    for p in semantic:
        key = (p["label"], int(p["start"]), int(p["end"]))
        if key in struct_keys:
            kept.append(p)  # exact agreement -> dedupe below keeps the higher score
        elif overlaps_intervals(int(p["start"]), int(p["end"]), territory):
            continue  # regex owns this territory (any label) -> drop GLiNER span
        else:
            kept.append(p)  # gap fill

    best: dict[tuple[str, int, int], Span] = {}
    for p in kept:
        key = (p["label"], int(p["start"]), int(p["end"]))
        cur = best.get(key)
        if cur is None or _score(p) > _score(cur):
            best[key] = p
    merged = list(best.values())
    merged.sort(key=lambda s: (int(s["start"]), int(s["end"]), s["label"]))
    return merged


class CompositePIIAdapter(BasePIIAdapter):
    name = "composite"

    def __init__(
        self,
        structured_adapter: BasePIIAdapter,
        semantic_adapter: BasePIIAdapter | None,
        structured_labels: Iterable[str],
        semantic_labels: Iterable[str],
        *,
        name: str | None = None,
    ) -> None:
        super().__init__({})
        self.structured_adapter = structured_adapter
        self.semantic_adapter = semantic_adapter
        self.structured_labels = set(structured_labels)
        # labels the semantic (GLiNER) detector is allowed to contribute:
        # PERSON/LOCATION/ADDRESS plus the structured-gap labels it is prompted for
        self.semantic_labels = set(semantic_labels)
        if name:
            self.name = name

    def load(self) -> None:
        self.structured_adapter.load()
        if self.semantic_adapter is not None:
            self.semantic_adapter.load()
        self._loaded = True

    def predict(self, text: str) -> list[Span]:
        structured = [
            p for p in self.structured_adapter.predict(text)
            if p["label"] in self.structured_labels
        ]
        semantic = (
            []
            if self.semantic_adapter is None
            else [
                p for p in self.semantic_adapter.predict(text)
                if p["label"] in self.semantic_labels
            ]
        )
        return merge_predictions(structured, semantic)

    def unload(self) -> None:
        self.structured_adapter.unload()
        if self.semantic_adapter is not None:
            self.semantic_adapter.unload()
        super().unload()
