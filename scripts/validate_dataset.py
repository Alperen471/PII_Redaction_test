"""Freeze-check the benchmark dataset (plan v4 §8, §9.1).

Hard failures (exit 1):
    * root not a JSON array
    * sample count != 1000
    * ids not exactly PII-0000..PII-0999, unique and ordered
    * span invariant broken: text[start:end] != entity["text"]
    * entity missing a required field / bad offsets

Soft warnings (exit 0, printed):
    * label support differs from the plan's §8.4 table
    * entity total != 1685
    * dataset_type counts differ from §8.2
    * entity_count field disagrees with len(entities)

Usage:
    python -m scripts.validate_dataset [path/to/dataset.json]
"""
from __future__ import annotations

import sys
import unicodedata
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.io import (  # noqa: E402
    DEFAULT_DATASET_PATH,
    EXPECTED_DATASET_TYPE_COUNTS,
    EXPECTED_ENTITY_TOTAL,
    EXPECTED_LABEL_SUPPORT,
    EXPECTED_SAMPLES,
    load_dataset,
)

REQUIRED_ENTITY_FIELDS = ("text", "label", "start", "end")


def validate(path: str) -> int:
    errors: list[str] = []
    warnings: list[str] = []

    dataset = load_dataset(path)

    n = len(dataset)
    if n != EXPECTED_SAMPLES:
        errors.append(f"sample count = {n}, expected {EXPECTED_SAMPLES}")

    ids = [s.get("id") for s in dataset]
    expected_ids = [f"PII-{i:04d}" for i in range(n)]
    if ids != expected_ids:
        first_bad = next(
            (i for i, (a, b) in enumerate(zip(ids, expected_ids)) if a != b), None
        )
        errors.append(
            f"id sequence broken (unique/ordered PII-0000..). first mismatch at "
            f"index {first_bad}: got {ids[first_bad] if first_bad is not None else '?'!r}"
        )
    if len(set(ids)) != len(ids):
        dups = [k for k, v in Counter(ids).items() if v > 1]
        errors.append(f"duplicate ids: {dups[:10]}")

    label_support: Counter = Counter()
    dtype_counts: Counter = Counter()
    entity_total = 0

    for s in dataset:
        sid = s.get("id", "?")
        text = s.get("text")
        if not isinstance(text, str):
            errors.append(f"{sid}: 'text' missing or not a string")
            continue
        dtype_counts[s.get("dataset_type", "UNKNOWN")] += 1
        entities = s.get("entities", [])
        if s.get("entity_count") is not None and s["entity_count"] != len(entities):
            warnings.append(
                f"{sid}: entity_count={s['entity_count']} != len(entities)={len(entities)}"
            )
        for k, e in enumerate(entities):
            missing = [f for f in REQUIRED_ENTITY_FIELDS if f not in e]
            if missing:
                errors.append(f"{sid} entity#{k}: missing fields {missing}")
                continue
            if e.get("out_of_scope"):
                # out-of-scope helper rows may omit valid offsets (plan v4 §8.4.1)
                label_support[e["label"]] += 1
                entity_total += 1
                continue
            start, end = e["start"], e["end"]
            if not (isinstance(start, int) and isinstance(end, int)):
                errors.append(f"{sid} entity#{k}: start/end not int")
                continue
            if not (0 <= start < end <= len(text)):
                errors.append(
                    f"{sid} entity#{k}: offsets [{start},{end}) out of range 0..{len(text)}"
                )
                continue
            surface = text[start:end]
            if unicodedata.normalize("NFC", surface) != unicodedata.normalize(
                "NFC", e["text"]
            ):
                errors.append(
                    f"{sid} entity#{k}: span invariant broken "
                    f"text[{start}:{end}]={surface!r} != {e['text']!r}"
                )
            label_support[e["label"]] += 1
            entity_total += 1

    if entity_total != EXPECTED_ENTITY_TOTAL:
        warnings.append(
            f"entity total = {entity_total}, plan §8.4 expects {EXPECTED_ENTITY_TOTAL}"
        )
    for lab, exp in EXPECTED_LABEL_SUPPORT.items():
        got = label_support.get(lab, 0)
        if got != exp:
            warnings.append(f"label {lab}: support {got}, plan §8.4 expects {exp}")
    extra = set(label_support) - set(EXPECTED_LABEL_SUPPORT) - {"ORGANIZATION"}
    if extra:
        warnings.append(f"unexpected labels present: {sorted(extra)}")
    for dt, exp in EXPECTED_DATASET_TYPE_COUNTS.items():
        got = dtype_counts.get(dt, 0)
        if got != exp:
            warnings.append(f"dataset_type {dt!r}: {got}, plan §8.2 expects {exp}")

    print(f"dataset: {path}")
    print(f"samples: {n}")
    print(f"entities (incl. out_of_scope): {entity_total}")
    print("label support:")
    for lab in sorted(label_support):
        flag = "  (LOW)" if label_support[lab] < 30 else ""
        print(f"  {lab:<16} {label_support[lab]:>4}{flag}")
    print("dataset_type counts:")
    for dt in sorted(dtype_counts):
        print(f"  {dt:<28} {dtype_counts[dt]:>4}")

    if warnings:
        print(f"\n{len(warnings)} WARNING(S):")
        for w in warnings:
            print(f"  ! {w}")
    if errors:
        print(f"\n{len(errors)} ERROR(S):")
        for e in errors:
            print(f"  x {e}")
        print("\nVALIDATION FAILED")
        return 1
    print("\nVALIDATION OK")
    return 0


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_DATASET_PATH
    raise SystemExit(validate(target))
