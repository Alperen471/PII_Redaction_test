"""Semantic NER benchmark metrics: PERSON + LOCATION only (plan v5 §5).

Wraps :func:`evaluation.metrics.compute_metrics` with ``label_subset =
{PERSON, LOCATION}`` and adds the per-label PERSON / LOCATION breakout that the
semantic leaderboard needs (plan v5 §5.3).
"""
from __future__ import annotations

from typing import Sequence

from evaluation.metrics import MetricsResult, compute_metrics
from evaluation.spans import Span

SEMANTIC_LABELS = frozenset({"PERSON", "LOCATION"})


def _label_block(result: MetricsResult, label: str) -> dict:
    info = result.per_label.get(label)
    if info is None:
        return {
            "support": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0,
            "exact_f1": 0.0,
        }
    return {
        "support": info["support"],
        "precision": info["relaxed"]["precision"],
        "recall": info["relaxed"]["recall"],
        "f1": info["relaxed"]["f1"],
        "exact_precision": info["exact"]["precision"],
        "exact_recall": info["exact"]["recall"],
        "exact_f1": info["exact"]["f1"],
    }


def compute_semantic_metrics(
    records: Sequence[dict],
    predictions_by_id: dict[str, Sequence[Span]],
) -> dict:
    result = compute_metrics(records, predictions_by_id, label_subset=SEMANTIC_LABELS)
    return {
        "labels": sorted(SEMANTIC_LABELS),
        "sample_count": result.sample_count,
        "gold_entity_count": result.gold_entity_count,
        "relaxed_micro": result.micro_relaxed,
        "relaxed_macro": result.macro_relaxed,
        "exact_micro": result.micro_exact,
        "exact_macro": result.macro_exact,
        "macro_supported": result.macro_supported,
        "person": _label_block(result, "PERSON"),
        "location": _label_block(result, "LOCATION"),
        "counts": {
            "relaxed_fp": result.false_positive_count,
            "relaxed_fn": result.false_negative_count,
            "exact_fp": result.exact_false_positive_count,
            "exact_fn": result.exact_false_negative_count,
            "partial_span_count": result.partial_span_count,
        },
        "dataset_type": result.per_dataset_type,
        "per_label": result.per_label,
    }
