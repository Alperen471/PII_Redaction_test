"""Post-tokenization safety metrics (plan v4 §10.10, §10.11).

Both metrics are computed on the *interval* level against the overlap-resolved
predicted spans that the tokenizer actually substituted:

    pii_leakage_rate     = gold entities not fully hidden        / gold entities
    tokenization_coverage= gold entities hidden by a same-label token / gold entities

A gold entity that is fully redacted but by a wrong-label token counts as an
error: it is NOT covered (plan v4 §10.11) though it is also NOT leaked.
Target: pii_leakage_rate == 0 (plan v4 §10.10).
"""
from __future__ import annotations

from typing import Sequence

from common.taxonomy import is_in_scope
from evaluation.spans import Span, covered_intervals


def _fully_covered(gold: Span, intervals: list[tuple[int, int]]) -> bool:
    cursor = gold["start"]
    for lo, hi in intervals:
        if lo > cursor:
            break
        if hi > cursor:
            cursor = hi
        if cursor >= gold["end"]:
            return True
    return cursor >= gold["end"]


def leakage_report(
    records: Sequence[dict],
    applied_spans_by_id: dict[str, Sequence[Span]],
) -> dict:
    total = 0
    leaked = 0
    covered = 0
    wrong_label = 0
    leaked_examples: list[dict] = []
    records_with_leak = 0

    for sample in records:
        sid = sample["id"]
        applied = list(applied_spans_by_id.get(sid, []))
        any_all = covered_intervals(applied)
        by_label: dict[str, list[tuple[int, int]]] = {}
        for p in applied:
            by_label.setdefault(p["label"], []).append((p["start"], p["end"]))
        by_label = {k: covered_intervals(
            [{"start": a, "end": b} for a, b in v]
        ) for k, v in by_label.items()}

        record_leaked = False
        for e in sample.get("entities", []):
            if not is_in_scope(e["label"]) or e.get("out_of_scope", False):
                continue
            gold: Span = {
                "text": e["text"],
                "label": e["label"],
                "start": e["start"],
                "end": e["end"],
            }
            total += 1
            hidden_any = _fully_covered(gold, any_all)
            hidden_same = _fully_covered(gold, by_label.get(e["label"], []))
            if not hidden_any:
                leaked += 1
                record_leaked = True
                if len(leaked_examples) < 50:
                    leaked_examples.append(
                        {"id": sid, "label": e["label"], "text": e["text"]}
                    )
            elif hidden_same:
                covered += 1
            else:
                wrong_label += 1
        if record_leaked:
            records_with_leak += 1

    return {
        "gold_entity_count": total,
        "leaked_count": leaked,
        "pii_leakage_rate": (leaked / total) if total else 0.0,
        "correctly_tokenized_count": covered,
        "tokenization_coverage": (covered / total) if total else 0.0,
        "wrong_label_tokenization_count": wrong_label,
        "records_with_leak": records_with_leak,
        "leaked_examples": leaked_examples,
    }
