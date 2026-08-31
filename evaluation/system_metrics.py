"""System-Level PII benchmark metrics: full 13-label taxonomy (plan v5 §6, §7).

Measures production *system* coverage. A label the system does not detect at all
turns every matching gold entity into an FN -- correct here, because the unit
under test is the system, not a model (plan v5 §7).

Adds, on top of :func:`evaluation.metrics.compute_metrics`:
    * PII leakage rate + tokenization coverage (plan v5 §17)
    * hard-negative false-positive analysis (plan v5 §18, §29)
"""
from __future__ import annotations

from typing import Sequence

from common.taxonomy import EVALUATION_LABELS, is_in_scope
from evaluation.entity_metrics import entity_table
from evaluation.leakage import leakage_report
from evaluation.metrics import compute_metrics
from evaluation.spans import Span


def _is_negative(sample: dict) -> bool:
    """No in-scope, non-out-of-scope gold entity (plan v5 §18, runtime rule)."""
    return not any(
        is_in_scope(e["label"]) and not e.get("out_of_scope", False)
        for e in sample.get("entities", [])
    )


def hard_negative_analysis(
    records: Sequence[dict],
    predictions_by_id: dict[str, Sequence[Span]],
) -> dict:
    negatives = [s for s in records if _is_negative(s)]
    hard = [s for s in negatives if s.get("dataset_type") == "Ambiguous / Hard Negative"]

    def block(samples: list[dict]) -> dict:
        if not samples:
            return {"count": 0, "false_positive_count": 0, "false_positive_rate": 0.0,
                    "clean_pass_rate": 0.0, "predicted_labels": {}}
        fp = 0
        clean = 0
        labels: dict[str, int] = {}
        for s in samples:
            preds = [p for p in predictions_by_id.get(s["id"], []) if is_in_scope(p["label"])]
            fp += len(preds)
            clean += 1 if not preds else 0
            for p in preds:
                labels[p["label"]] = labels.get(p["label"], 0) + 1
        return {
            "count": len(samples),
            "false_positive_count": fp,
            "false_positive_rate": fp / len(samples),
            "clean_pass_rate": clean / len(samples),
            "predicted_labels": dict(sorted(labels.items(), key=lambda kv: -kv[1])),
        }

    return {
        "all_negative_samples": block(negatives),
        "hard_negative_category": block(hard),
    }


def compute_system_metrics(
    records: Sequence[dict],
    predictions_by_id: dict[str, Sequence[Span]],
    applied_spans_by_id: dict[str, Sequence[Span]],
) -> dict:
    result = compute_metrics(records, predictions_by_id)  # full taxonomy
    leak = leakage_report(records, applied_spans_by_id)

    positives = sum(1 for s in records if not _is_negative(s))
    return {
        "labels": sorted(EVALUATION_LABELS),
        "sample_count": result.sample_count,
        "positive_samples": positives,
        "negative_samples": result.sample_count - positives,
        "gold_entity_count": result.gold_entity_count,
        "relaxed_micro": result.micro_relaxed,
        "relaxed_macro": result.macro_relaxed,
        "exact_micro": result.micro_exact,
        "exact_macro": result.macro_exact,
        "macro_supported": result.macro_supported,
        "counts": {
            "relaxed_fp": result.false_positive_count,
            "relaxed_fn": result.false_negative_count,
            "exact_fp": result.exact_false_positive_count,
            "exact_fn": result.exact_false_negative_count,
            "partial_span_count": result.partial_span_count,
        },
        "entity_level": {
            "relaxed": entity_table(result, mode="relaxed"),
            "exact": entity_table(result, mode="exact"),
        },
        "dataset_type": result.per_dataset_type,
        "leakage": {k: v for k, v in leak.items() if k != "leaked_examples"},
        "leaked_examples": leak["leaked_examples"],
        "hard_negative": hard_negative_analysis(records, predictions_by_id),
        "per_label": result.per_label,
    }
