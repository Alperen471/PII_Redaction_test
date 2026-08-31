"""Aggregate benchmark metrics (plan v4 §10).

Locked scoring contract:
    PRIMARY  = relaxed micro Precision / Recall / F1
               (relaxed match = same label + >0 char overlap; optimal 1:1 align)
    SECONDARY:
        relaxed macro P/R/F1
        exact   micro P/R/F1        (label + exact [start,end))
        exact   macro P/R/F1
        macro-supported P/R/F1      (labels with support >= 30 only)

Model-selection order (contract): relaxed micro recall -> leakage rate ->
exact micro F1 -> macro-supported recall -> P95 latency.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from common.taxonomy import LOW_SUPPORT_THRESHOLD, is_in_scope
<<<<<<< HEAD
from evaluation.alignment import align_exact, align_relaxed
from evaluation.spans import Span
=======
from evaluation.spans import Span, align_exact, align_relaxed
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "Counts") -> None:
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def support(self) -> int:
        return self.tp + self.fn

    def prf(self) -> tuple[float, float, float]:
        precision = self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0
        recall = self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall)
            else 0.0
        )
        return precision, recall, f1


@dataclass
class RecordScore:
    id: str
    dataset_type: str
    relaxed: Counts
    exact: Counts
    n_pred_in_scope: int
    n_gold: int


@dataclass
class MetricsResult:
    micro_relaxed: dict
    macro_relaxed: dict
    micro_exact: dict
    macro_exact: dict
    macro_supported: dict
    per_label: dict[str, dict]
    per_dataset_type: dict[str, dict]
    false_negative_count: int
    false_positive_count: int
    exact_false_negative_count: int
    exact_false_positive_count: int
    partial_span_count: int
    sample_count: int
    gold_entity_count: int
    per_record: list[dict] = field(default_factory=list)


def _prf_dict(c: Counts) -> dict:
    p, r, f = c.prf()
    return {
        "precision": p,
        "recall": r,
        "f1": f,
        "tp": c.tp,
        "fp": c.fp,
        "fn": c.fn,
        "support": c.support,
    }


def _macro(label_counts: dict[str, Counts], labels: Iterable[str]) -> dict:
    labels = [l for l in labels]
    if not labels:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "labels": []}
    ps, rs, fs = [], [], []
    for l in labels:
        p, r, f = label_counts[l].prf()
        ps.append(p); rs.append(r); fs.append(f)
    n = len(labels)
    return {
        "precision": sum(ps) / n,
        "recall": sum(rs) / n,
        "f1": sum(fs) / n,
        "labels": sorted(labels),
    }


def _score_record(
    preds: Sequence[Span],
    golds: Sequence[Span],
    relaxed_label_counts: dict[str, Counts],
    exact_label_counts: dict[str, Counts],
) -> tuple[Counts, Counts, int]:
    """Score one record; updates per-label counters in place.

    FP is charged to the predicted label, FN to the gold label. Partial-match
    count is returned for reporting (plan v4 §10.7).
    """
    rel = Counts()
    exa = Counts()

    rel_pairs = set(align_relaxed(preds, golds))
    matched_pred = {i for i, _ in rel_pairs}
    matched_gold = {j for _, j in rel_pairs}
    partials = 0
    for i, j in rel_pairs:
        rel.tp += 1
        relaxed_label_counts[golds[j]["label"]].tp += 1
        if not (
            preds[i]["start"] == golds[j]["start"]
            and preds[i]["end"] == golds[j]["end"]
        ):
            partials += 1
    for i, p in enumerate(preds):
        if i not in matched_pred:
            rel.fp += 1
            relaxed_label_counts[p["label"]].fp += 1
    for j, g in enumerate(golds):
        if j not in matched_gold:
            rel.fn += 1
            relaxed_label_counts[g["label"]].fn += 1

    exa_pairs = set(align_exact(preds, golds))
    ematched_pred = {i for i, _ in exa_pairs}
    ematched_gold = {j for _, j in exa_pairs}
    for _, j in exa_pairs:
        exa.tp += 1
        exact_label_counts[golds[j]["label"]].tp += 1
    for i, p in enumerate(preds):
        if i not in ematched_pred:
            exa.fp += 1
            exact_label_counts[p["label"]].fp += 1
    for j, g in enumerate(golds):
        if j not in ematched_gold:
            exa.fn += 1
            exact_label_counts[g["label"]].fn += 1

    return rel, exa, partials


def compute_metrics(
    records: Sequence[dict],
    predictions_by_id: dict[str, Sequence[Span]],
<<<<<<< HEAD
    *,
    label_subset: frozenset[str] | set[str] | None = None,
) -> MetricsResult:
    """Compute all benchmark metrics (plan v5 §11).

    ``records`` are raw dataset samples (with ``entities``). ``predictions_by_id``
    maps sample id -> prediction spans.

    ``label_subset`` restricts scoring to those canonical labels (used by the
    Semantic NER benchmark with ``{PERSON, LOCATION}``, plan v5 §5.1). ``None``
    means the full in-scope taxonomy (plan v5 §4). Out-of-scope labels
    (``ORGANIZATION``) are always excluded from primary/secondary metrics.
    """
    def _keep(label: str) -> bool:
        if not is_in_scope(label):
            return False
        return label_subset is None or label in label_subset

=======
) -> MetricsResult:
    """Compute all benchmark metrics.

    ``records`` are raw dataset samples (with ``entities``). ``predictions_by_id``
    maps sample id -> in-scope prediction spans (out-of-scope labels must already
    be filtered, plan v4 §8.4.1).
    """
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
    relaxed_label_counts: dict[str, Counts] = defaultdict(Counts)
    exact_label_counts: dict[str, Counts] = defaultdict(Counts)
    micro_rel = Counts()
    micro_exa = Counts()
    partial_total = 0
    per_dtype_rel: dict[str, Counts] = defaultdict(Counts)
    per_dtype_exa: dict[str, Counts] = defaultdict(Counts)
    per_record: list[dict] = []
    gold_total = 0

    for sample in records:
        sid = sample["id"]
        dtype = sample.get("dataset_type", "UNKNOWN")
        golds: list[Span] = [
            {
                "text": e["text"],
                "label": e["label"],
                "start": e["start"],
                "end": e["end"],
            }
            for e in sample.get("entities", [])
<<<<<<< HEAD
            if _keep(e["label"]) and not e.get("out_of_scope", False)
        ]
        gold_total += len(golds)
        # out-of-scope / off-subset predictions never enter the metric
        # (plan v5 §4, §5.1); runners also pre-filter, this guards direct callers.
        preds = [
            p for p in predictions_by_id.get(sid, [])
            if _keep(p["label"])
=======
            if is_in_scope(e["label"]) and not e.get("out_of_scope", False)
        ]
        gold_total += len(golds)
        # defensive: out-of-scope predictions never enter the primary metric
        # (plan v4 §8.4.1); run_model already filters, this guards direct callers.
        preds = [
            p for p in predictions_by_id.get(sid, [])
            if is_in_scope(p["label"])
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
        ]

        rel, exa, partials = _score_record(
            preds, golds, relaxed_label_counts, exact_label_counts
        )
        micro_rel.add(rel)
        micro_exa.add(exa)
        partial_total += partials
        per_dtype_rel[dtype].add(rel)
        per_dtype_exa[dtype].add(exa)
        per_record.append(
            {
                "id": sid,
                "dataset_type": dtype,
                "relaxed": {"tp": rel.tp, "fp": rel.fp, "fn": rel.fn},
                "exact": {"tp": exa.tp, "fp": exa.fp, "fn": exa.fn},
                "n_pred": len(preds),
                "n_gold": len(golds),
            }
        )

    # macro is averaged over labels that actually occur in the gold (support > 0);
    # the per-label table additionally surfaces in-scope labels that only ever
    # appear as false positives (support 0, fp > 0) so they are not hidden.
    gold_labels = sorted(
        l for l, c in relaxed_label_counts.items() if c.support > 0 and is_in_scope(l)
    )
    supported_labels = [
        l for l in gold_labels if relaxed_label_counts[l].support >= LOW_SUPPORT_THRESHOLD
    ]
    report_labels = sorted(
        l
        for l, c in relaxed_label_counts.items()
        if is_in_scope(l) and (c.support > 0 or c.fp > 0)
    )

    per_label: dict[str, dict] = {}
    for l in report_labels:
        rc = relaxed_label_counts[l]
        ec = exact_label_counts[l]
        support = rc.support
        per_label[l] = {
            "support": support,
            "low_support": support < LOW_SUPPORT_THRESHOLD,
            "relaxed": _prf_dict(rc),
            "exact": _prf_dict(ec),
        }

    per_dtype: dict[str, dict] = {}
    for dt in sorted(per_dtype_rel):
        per_dtype[dt] = {
            "relaxed": _prf_dict(per_dtype_rel[dt]),
            "exact": _prf_dict(per_dtype_exa[dt]),
        }

    return MetricsResult(
        micro_relaxed=_prf_dict(micro_rel),
        macro_relaxed=_macro(relaxed_label_counts, gold_labels),
        micro_exact=_prf_dict(micro_exa),
        macro_exact=_macro(exact_label_counts, gold_labels),
        macro_supported=_macro(relaxed_label_counts, supported_labels),
        per_label=per_label,
        per_dataset_type=per_dtype,
        false_negative_count=micro_rel.fn,
        false_positive_count=micro_rel.fp,
        exact_false_negative_count=micro_exa.fn,
        exact_false_positive_count=micro_exa.fp,
        partial_span_count=partial_total,
        sample_count=len(records),
        gold_entity_count=gold_total,
        per_record=per_record,
    )
