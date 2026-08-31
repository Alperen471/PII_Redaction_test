"""Benchmark a single detector end-to-end (plan v4 §16.6, §15).

Flow:
    config -> json.load dataset -> verify 1000 -> build adapter -> cold start
    -> warm-up -> per-sample predict (batch=1, timed) -> evaluate vs entities
    -> tokenize -> accuracy + leakage + latency metrics -> write raw + metrics
    -> update leaderboard.

Usage:
    python -m scripts.run_model --model regex
    python -m scripts.run_model --model regex --limit 50 --device cpu
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters import get_adapter  # noqa: E402
from benchmarks.latency import cuda_sync, summarize  # noqa: E402
from benchmarks.memory import MemoryTracker, reset_cuda_peak  # noqa: E402
from benchmarks.throughput import throughput_from_latency  # noqa: E402
from common.io import EXPECTED_SAMPLES, load_dataset, save_json  # noqa: E402
from common.taxonomy import is_in_scope  # noqa: E402
from config import load_config, model_config  # noqa: E402
from evaluation.entity_metrics import entity_table  # noqa: E402
from evaluation.leakage import leakage_report  # noqa: E402
from evaluation.metrics import compute_metrics  # noqa: E402
from scripts.leaderboard import update_leaderboard  # noqa: E402
from tokenization.tokenizer import tokenize  # noqa: E402


def _split_scope(spans):
    in_scope, out_scope = [], []
    for s in spans:
        (in_scope if is_in_scope(s["label"]) else out_scope).append(s)
    return in_scope, out_scope


def _clean_pass_stats(records, in_scope_by_id):
    """Hard-negative behaviour (plan v4 §10.5): records with no gold entities."""
    neg = [r for r in records if not [
        e for e in r.get("entities", [])
        if is_in_scope(e["label"]) and not e.get("out_of_scope")
    ]]
    if not neg:
        return {"negative_records": 0}
    clean = sum(1 for r in neg if not in_scope_by_id.get(r["id"]))
    fp_spans = sum(len(in_scope_by_id.get(r["id"], [])) for r in neg)
    by_dtype: dict[str, dict] = {}
    for r in neg:
        d = by_dtype.setdefault(r.get("dataset_type", "UNKNOWN"), {"n": 0, "clean": 0})
        d["n"] += 1
        d["clean"] += 0 if in_scope_by_id.get(r["id"]) else 1
    return {
        "negative_records": len(neg),
        "clean_pass_records": clean,
        "clean_pass_rate": clean / len(neg),
        "false_positive_spans": fp_spans,
        "by_dataset_type": by_dtype,
    }


def run(model_name: str, args) -> dict:
    cfg = load_config(args.config)
    mcfg = model_config(cfg, model_name)
    if args.device:
        mcfg["device"] = args.device
    device = str(mcfg.get("device", "cpu"))
    warmup = int(cfg.get("run", {}).get("warmup_runs", 10))
    dataset_path = args.dataset or cfg["dataset"].get("path")

    dataset = load_dataset(dataset_path)
    if args.limit:
        dataset = dataset[: args.limit]
    elif len(dataset) != EXPECTED_SAMPLES:
        print(f"WARNING: dataset has {len(dataset)} samples, expected {EXPECTED_SAMPLES}")

    texts = [s["text"] for s in dataset]

    mem = MemoryTracker(device)
    mem.snapshot("baseline")
    reset_cuda_peak(device)

    adapter = get_adapter(mcfg["adapter"], mcfg)
    t0 = time.perf_counter_ns()
    adapter.load()
    model_load_time_ms = (time.perf_counter_ns() - t0) / 1e6
    mem.snapshot("after_load")

    # cold start: time to first inference from a freshly loaded model
    t_cs = time.perf_counter_ns()
    _ = adapter.predict(texts[0]) if texts else []
    first_inference_ms = (time.perf_counter_ns() - t_cs) / 1e6

    # warm-up (discarded, plan v4 §11)
    for i in range(min(warmup, len(texts))):
        adapter.predict(texts[i])

    # measured pass: batch=1, capture predictions + latency together
    raw_records = []
    in_scope_by_id: dict[str, list] = {}
    applied_by_id: dict[str, list] = {}
    per_sample_ms: list[float] = []

    for sample, text in zip(dataset, texts):
        cuda_sync(device)
        ts = time.perf_counter_ns()
        preds = adapter.predict(text)
        cuda_sync(device)
        latency_ms = (time.perf_counter_ns() - ts) / 1e6
        per_sample_ms.append(latency_ms)

        in_scope, out_scope = _split_scope(preds)
        in_scope_by_id[sample["id"]] = in_scope
        _safe, _map, applied = tokenize(text, in_scope)
        applied_by_id[sample["id"]] = applied

        raw_records.append(
            {
                "id": sample["id"],
                "dataset_type": sample.get("dataset_type"),
                "predictions": preds,
                "predictions_in_scope": in_scope,
                "predictions_out_of_scope": out_scope,
                "latency_ms": latency_ms,
            }
        )

    mem.snapshot("after_run")

    latency = summarize(per_sample_ms, warmup=min(warmup, len(texts)))
    metrics = compute_metrics(dataset, in_scope_by_id)
    leak = leakage_report(dataset, applied_by_id)
    clean = _clean_pass_stats(dataset, in_scope_by_id)

    summary = {
        "model": model_name,
        "adapter": mcfg["adapter"],
        "dataset": Path(dataset_path).name,
        "sample_count": len(dataset),
        "gold_entity_count": metrics.gold_entity_count,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "device": device,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "device": device,
            "warmup_runs": warmup,
            "threshold": mcfg.get("threshold"),
            "threshold_locked": mcfg.get("threshold_locked"),
            "threshold_source": mcfg.get("threshold_source"),
            "labels": mcfg.get("labels"),
            "model_name": mcfg.get("model_name"),
        },
        "primary": {
            "relaxed_micro_precision": metrics.micro_relaxed["precision"],
            "relaxed_micro_recall": metrics.micro_relaxed["recall"],
            "relaxed_micro_f1": metrics.micro_relaxed["f1"],
        },
        "secondary": {
            "relaxed_macro": metrics.macro_relaxed,
            "exact_micro": metrics.micro_exact,
            "exact_macro": metrics.macro_exact,
            "macro_supported": metrics.macro_supported,
        },
        "counts": {
            "relaxed_tp": metrics.micro_relaxed["tp"],
            "relaxed_fp": metrics.false_positive_count,
            "relaxed_fn": metrics.false_negative_count,
            "exact_fp": metrics.exact_false_positive_count,
            "exact_fn": metrics.exact_false_negative_count,
            "partial_span_count": metrics.partial_span_count,
        },
        "entity_level": {
            "relaxed": entity_table(metrics, mode="relaxed"),
            "exact": entity_table(metrics, mode="exact"),
        },
        "dataset_type": metrics.per_dataset_type,
        "hard_negative": clean,
        "leakage": {k: v for k, v in leak.items() if k != "leaked_examples"},
        "leaked_examples": leak["leaked_examples"],
        "latency_ms": {k: v for k, v in latency.items() if k != "per_sample_ms"},
        "throughput": throughput_from_latency(latency),
        "cold_start": {
            "model_load_time_ms": model_load_time_ms,
            "first_inference_ms": first_inference_ms,
            "time_to_first_inference_ms": model_load_time_ms + first_inference_ms,
        },
        "resources": {**mem.report(), "model_size_mb": mcfg.get("model_size_mb")},
    }

    out_dir = Path(args.out_dir)
    save_json(out_dir / "raw" / f"{model_name}_predictions.json", raw_records)
    save_json(out_dir / "metrics" / f"{model_name}_metrics.json", summary)
    save_json(out_dir / "metrics" / f"{model_name}_latency.json", {
        "model": model_name, "per_sample_ms": latency["per_sample_ms"],
    })
    update_leaderboard(out_dir / "leaderboard.csv", summary)

    adapter.unload()

    p = summary["primary"]
    print(
        f"[{model_name}] relaxed micro  P={p['relaxed_micro_precision']:.3f} "
        f"R={p['relaxed_micro_recall']:.3f} F1={p['relaxed_micro_f1']:.3f}  "
        f"leakage={leak['pii_leakage_rate']:.3f}  "
        f"exactF1={metrics.micro_exact['f1']:.3f}  "
        f"P95={latency['p95_ms']:.2f}ms"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default=None, help="override config device")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on first N records")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    run(args.model, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
