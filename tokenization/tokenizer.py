"""Redaction layer: tokenization / pseudonymization (plan v4 §3.2, §4).

This layer is deliberately separate from the evaluator (plan v4 §4): it turns
detector spans into safe tokens, it never sees the ground truth.

Locked rules (implementation contract + plan v4 §4):
    * token key   = ``(label, unicodedata.normalize("NFC", detected_text).strip())``
    * same key    -> same token within one text
    * numbering   = per-label, 1-based, in order of first appearance
    * NO coreference / alias / fuzzy resolution (plan v4 §4.2, §4.5)
    * overlapping predictions: keep higher score, then earlier start (deterministic)
"""
from __future__ import annotations

import unicodedata
from typing import Sequence

from evaluation.spans import Span


def _key(label: str, surface: str) -> tuple[str, str]:
    return label, unicodedata.normalize("NFC", surface).strip()


def resolve_overlaps(preds: Sequence[Span]) -> list[Span]:
    """Drop overlapping predictions deterministically.

    Sort by (score desc, start asc, end asc, label) and greedily keep a
    prediction only if it does not overlap an already-kept one.
    """
    ordered = sorted(
        preds,
        key=lambda p: (
            -float(p.get("score", 1.0)),
            int(p["start"]),
            int(p["end"]),
            str(p.get("label", "")),
        ),
    )
    kept: list[Span] = []
    for p in ordered:
        if any(not (p["end"] <= k["start"] or p["start"] >= k["end"]) for k in kept):
            continue
        kept.append(p)
    kept.sort(key=lambda p: (int(p["start"]), int(p["end"])))
    return kept


def tokenize(text: str, preds: Sequence[Span]) -> tuple[str, dict, list[Span]]:
    """Return ``(safe_text, token_map, applied_spans)``.

    ``token_map`` maps ``"LABEL||surface"`` -> ``"<LABEL_n>"``. ``applied_spans``
    is the overlap-resolved span list actually substituted (also what the
    leakage/coverage metrics consume).
    """
    applied = resolve_overlaps(preds)
    token_map: dict[str, str] = {}
    counters: dict[str, int] = {}
    token_for_span: list[str] = []

    for p in applied:
        surface = text[p["start"]:p["end"]]
        label = p["label"]
        k = _key(label, surface)
        flat = f"{k[0]}||{k[1]}"
        if flat not in token_map:
            counters[label] = counters.get(label, 0) + 1
            token_map[flat] = f"<{label}_{counters[label]}>"
        token_for_span.append(token_map[flat])

    # substitute right-to-left so earlier offsets stay valid
    safe = text
    for p, token in sorted(
        zip(applied, token_for_span), key=lambda pt: pt[0]["start"], reverse=True
    ):
        safe = safe[:p["start"]] + token + safe[p["end"]:]

    return safe, token_map, applied
