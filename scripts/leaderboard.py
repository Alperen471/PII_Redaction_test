<<<<<<< HEAD
"""Leaderboard CSV upsert helpers (plan v5 §5.3, §6, §27, §28)."""
=======
"""leaderboard.csv upsert (plan v4 §16.5)."""
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
from __future__ import annotations

import csv
from pathlib import Path

<<<<<<< HEAD

def _fmt(v):
    if isinstance(v, float):
        return f"{v:.6f}"
    return "" if v is None else v


def upsert_csv(path: str | Path, columns: list[str], row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    key = columns[0]  # first column is the identity ("model" / "system")
    if path.is_file():
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r[key]] = r
    rows[row[key]] = row
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columns)
        w.writeheader()
        for ident in sorted(rows):
            src = rows[ident]
            w.writerow({c: _fmt(src.get(c, "")) for c in columns})


# --------------------------------------------------------------------------- #
# Semantic NER benchmark (plan v5 §5.3)
# --------------------------------------------------------------------------- #
SEMANTIC_COLUMNS = [
    "model",
    "relaxed_micro_precision", "relaxed_micro_recall", "relaxed_micro_f1",
    "relaxed_macro_precision", "relaxed_macro_recall", "relaxed_macro_f1",
    "exact_micro_precision", "exact_micro_recall", "exact_micro_f1",
    "person_precision", "person_recall", "person_f1", "person_exact_f1",
    "location_precision", "location_recall", "location_f1", "location_exact_f1",
    "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "model_size_mb", "ram_mb", "vram_mb", "cold_start_ms", "throughput_sps",
]


def semantic_row(summary: dict) -> dict:
    s = summary["semantic"]
    lat = summary["latency_ms"]
    p, loc = s["person"], s["location"]
    return {
        "model": summary["model"],
        "relaxed_micro_precision": s["relaxed_micro"]["precision"],
        "relaxed_micro_recall": s["relaxed_micro"]["recall"],
        "relaxed_micro_f1": s["relaxed_micro"]["f1"],
        "relaxed_macro_precision": s["relaxed_macro"]["precision"],
        "relaxed_macro_recall": s["relaxed_macro"]["recall"],
        "relaxed_macro_f1": s["relaxed_macro"]["f1"],
        "exact_micro_precision": s["exact_micro"]["precision"],
        "exact_micro_recall": s["exact_micro"]["recall"],
        "exact_micro_f1": s["exact_micro"]["f1"],
        "person_precision": p["precision"], "person_recall": p["recall"],
        "person_f1": p["f1"], "person_exact_f1": p.get("exact_f1", 0.0),
        "location_precision": loc["precision"], "location_recall": loc["recall"],
        "location_f1": loc["f1"], "location_exact_f1": loc.get("exact_f1", 0.0),
        "latency_p50_ms": lat["p50_ms"], "latency_p95_ms": lat["p95_ms"],
        "latency_p99_ms": lat["p99_ms"],
        "model_size_mb": summary["resources"].get("model_size_mb"),
        "ram_mb": summary["resources"].get("ram_load_delta_mb"),
        "vram_mb": summary["resources"].get("vram_mb"),
        "cold_start_ms": summary["cold_start"]["model_load_time_ms"],
        "throughput_sps": summary["throughput"]["samples_per_second"],
    }


def update_semantic_leaderboard(path: str | Path, summary: dict) -> None:
    upsert_csv(path, SEMANTIC_COLUMNS, semantic_row(summary))


# --------------------------------------------------------------------------- #
# System-Level PII benchmark (plan v5 §6, §28)
# --------------------------------------------------------------------------- #
SYSTEM_COLUMNS = [
    "system",
    "relaxed_micro_precision", "relaxed_micro_recall", "relaxed_micro_f1",
    "relaxed_macro_f1",
    "exact_micro_precision", "exact_micro_recall", "exact_micro_f1",
    "macro_supported_recall", "macro_supported_f1",
    "pii_leakage_rate", "tokenization_coverage",
    "hard_negative_fp_rate", "hard_negative_clean_pass_rate",
    "partial_span_count",
    "latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "throughput_sps", "model_load_time_ms",
    "ram_mb", "vram_mb", "model_size_mb", "cold_start_ms", "sample_count",
]


def system_row(summary: dict) -> dict:
    s = summary["system"]
    lat = summary["latency_ms"]
    hn = s["hard_negative"]["hard_negative_category"]
    return {
        "system": summary["system_name"],
        "relaxed_micro_precision": s["relaxed_micro"]["precision"],
        "relaxed_micro_recall": s["relaxed_micro"]["recall"],
        "relaxed_micro_f1": s["relaxed_micro"]["f1"],
        "relaxed_macro_f1": s["relaxed_macro"]["f1"],
        "exact_micro_precision": s["exact_micro"]["precision"],
        "exact_micro_recall": s["exact_micro"]["recall"],
        "exact_micro_f1": s["exact_micro"]["f1"],
        "macro_supported_recall": s["macro_supported"]["recall"],
        "macro_supported_f1": s["macro_supported"]["f1"],
        "pii_leakage_rate": s["leakage"]["pii_leakage_rate"],
        "tokenization_coverage": s["leakage"]["tokenization_coverage"],
        "hard_negative_fp_rate": hn["false_positive_rate"],
        "hard_negative_clean_pass_rate": hn["clean_pass_rate"],
        "partial_span_count": s["counts"]["partial_span_count"],
        "latency_avg_ms": lat["avg_ms"], "latency_p50_ms": lat["p50_ms"],
        "latency_p95_ms": lat["p95_ms"], "latency_p99_ms": lat["p99_ms"],
        "throughput_sps": summary["throughput"]["samples_per_second"],
        "model_load_time_ms": summary["cold_start"]["model_load_time_ms"],
        "ram_mb": summary["resources"].get("ram_load_delta_mb"),
        "vram_mb": summary["resources"].get("vram_mb"),
        "model_size_mb": summary["resources"].get("model_size_mb"),
        "cold_start_ms": summary["cold_start"]["time_to_first_inference_ms"],
        "sample_count": s["sample_count"],
    }


def update_system_leaderboard(path: str | Path, summary: dict) -> None:
    upsert_csv(path, SYSTEM_COLUMNS, system_row(summary))


# --------------------------------------------------------------------------- #
# Legacy full-taxonomy single-model leaderboard (scripts/run_model.py)
# --------------------------------------------------------------------------- #
COLUMNS = [
    "model", "relaxed_micro_precision", "relaxed_micro_recall", "relaxed_micro_f1",
    "relaxed_macro_precision", "relaxed_macro_recall", "relaxed_macro_f1",
    "exact_micro_precision", "exact_micro_recall", "exact_micro_f1", "exact_macro_f1",
    "macro_supported_recall", "macro_supported_f1", "pii_leakage_rate",
    "tokenization_coverage", "hard_negative_clean_pass_rate", "partial_span_count",
    "latency_avg_ms", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms",
    "throughput_sps", "model_load_time_ms", "ram_load_delta_mb", "vram_mb",
    "model_size_mb", "sample_count",
]


def update_leaderboard(path: str | Path, summary: dict) -> None:
    sec = summary["secondary"]
    lat = summary["latency_ms"]
    row = {
=======
COLUMNS = [
    "model",
    "relaxed_micro_precision",
    "relaxed_micro_recall",
    "relaxed_micro_f1",
    "relaxed_macro_precision",
    "relaxed_macro_recall",
    "relaxed_macro_f1",
    "exact_micro_precision",
    "exact_micro_recall",
    "exact_micro_f1",
    "exact_macro_f1",
    "macro_supported_recall",
    "macro_supported_f1",
    "pii_leakage_rate",
    "tokenization_coverage",
    "hard_negative_clean_pass_rate",
    "partial_span_count",
    "latency_avg_ms",
    "latency_p50_ms",
    "latency_p95_ms",
    "latency_p99_ms",
    "throughput_sps",
    "model_load_time_ms",
    "ram_load_delta_mb",
    "vram_mb",
    "model_size_mb",
    "sample_count",
]


def _row(summary: dict) -> dict:
    sec = summary["secondary"]
    lat = summary["latency_ms"]
    return {
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
        "model": summary["model"],
        "relaxed_micro_precision": summary["primary"]["relaxed_micro_precision"],
        "relaxed_micro_recall": summary["primary"]["relaxed_micro_recall"],
        "relaxed_micro_f1": summary["primary"]["relaxed_micro_f1"],
        "relaxed_macro_precision": sec["relaxed_macro"]["precision"],
        "relaxed_macro_recall": sec["relaxed_macro"]["recall"],
        "relaxed_macro_f1": sec["relaxed_macro"]["f1"],
        "exact_micro_precision": sec["exact_micro"]["precision"],
        "exact_micro_recall": sec["exact_micro"]["recall"],
        "exact_micro_f1": sec["exact_micro"]["f1"],
        "exact_macro_f1": sec["exact_macro"]["f1"],
        "macro_supported_recall": sec["macro_supported"]["recall"],
        "macro_supported_f1": sec["macro_supported"]["f1"],
        "pii_leakage_rate": summary["leakage"]["pii_leakage_rate"],
        "tokenization_coverage": summary["leakage"]["tokenization_coverage"],
        "hard_negative_clean_pass_rate": summary["hard_negative"].get("clean_pass_rate"),
        "partial_span_count": summary["counts"]["partial_span_count"],
<<<<<<< HEAD
        "latency_avg_ms": lat["avg_ms"], "latency_p50_ms": lat["p50_ms"],
        "latency_p95_ms": lat["p95_ms"], "latency_p99_ms": lat["p99_ms"],
=======
        "latency_avg_ms": lat["avg_ms"],
        "latency_p50_ms": lat["p50_ms"],
        "latency_p95_ms": lat["p95_ms"],
        "latency_p99_ms": lat["p99_ms"],
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
        "throughput_sps": summary["throughput"]["samples_per_second"],
        "model_load_time_ms": summary["cold_start"]["model_load_time_ms"],
        "ram_load_delta_mb": summary["resources"].get("ram_load_delta_mb"),
        "vram_mb": summary["resources"].get("vram_mb"),
        "model_size_mb": summary["resources"].get("model_size_mb"),
        "sample_count": summary["sample_count"],
    }
<<<<<<< HEAD
    upsert_csv(path, COLUMNS, row)
=======


def update_leaderboard(path: str | Path, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict] = {}
    if path.is_file():
        with open(path, "r", encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["model"]] = r
    rows[summary["model"]] = _row(summary)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.6f}"
        return "" if v is None else v

    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        for model in sorted(rows):
            src = rows[model]
            w.writerow({c: fmt(src.get(c, "")) for c in COLUMNS})
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
