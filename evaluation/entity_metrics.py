"""Entity-level (per-label) reporting (plan v4 §10.9, §8.4.2).

Every row carries ``support`` (gold entity count) and ``low_support``
(``support < 30``). Low-support rows must not be read as strong evidence.
"""
from __future__ import annotations

from common.taxonomy import LOW_SUPPORT_THRESHOLD
from evaluation.metrics import MetricsResult

_COLUMNS = ("label", "support", "low_support", "precision", "recall", "f1")


def entity_table(result: MetricsResult, *, mode: str = "relaxed") -> list[dict]:
    """Flat, sorted-by-label rows for ``results/metrics/*.json`` and the report."""
    rows: list[dict] = []
    for label in sorted(result.per_label):
        info = result.per_label[label]
        cell = info[mode]
        rows.append(
            {
                "label": label,
                "support": info["support"],
                "low_support": info["low_support"],
                "precision": cell["precision"],
                "recall": cell["recall"],
                "f1": cell["f1"],
                "tp": cell["tp"],
                "fp": cell["fp"],
                "fn": cell["fn"],
            }
        )
    return rows


def render_entity_table(result: MetricsResult, *, mode: str = "relaxed") -> str:
    rows = entity_table(result, mode=mode)
    head = f"{'Entity':<16}{'Support':>8}{'Low':>5}{'Prec':>8}{'Recall':>8}{'F1':>8}"
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(
            f"{r['label']:<16}{r['support']:>8}{'Y' if r['low_support'] else 'N':>5}"
            f"{r['precision']:>8.3f}{r['recall']:>8.3f}{r['f1']:>8.3f}"
        )
    return "\n".join(lines)


def low_support_labels(result: MetricsResult) -> list[str]:
    return sorted(
        l for l, info in result.per_label.items()
        if info["support"] < LOW_SUPPORT_THRESHOLD
    )
