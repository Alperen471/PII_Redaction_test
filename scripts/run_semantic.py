"""Semantic NER benchmark for one model: PERSON + LOCATION (plan v5 §5, §25).

    python -m scripts.run_semantic --model gliner_tr
    python -m scripts.run_semantic --model stanza --limit 50 --device cpu
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters import get_adapter  # noqa: E402
from benchmarks.model_size import model_size_mb  # noqa: E402
from benchmarks.throughput import throughput_from_latency  # noqa: E402
from common.io import EXPECTED_SAMPLES, load_dataset, save_json  # noqa: E402
from config import load_config, model_config  # noqa: E402
from evaluation.semantic_metrics import compute_semantic_metrics  # noqa: E402
from scripts._runner import run_inference  # noqa: E402
from scripts.leaderboard import update_semantic_leaderboard  # noqa: E402


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
        print(f"WARNING: {len(dataset)} samples, expected {EXPECTED_SAMPLES}")

    adapter = get_adapter(mcfg["adapter"], mcfg)
    res = run_inference(adapter, dataset, device=device, warmup=warmup)
    adapter.unload()

    semantic = compute_semantic_metrics(dataset, res["preds_by_id"])
    latency = res["latency"]

    summary = {
        "benchmark": "semantic",
        "model": model_name,
        "adapter": mcfg["adapter"],
        "dataset": Path(dataset_path).name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "device": device,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "device": device,
            "warmup_runs": warmup,
            "model_name": mcfg.get("model_name"),
            "threshold": mcfg.get("threshold"),
            "threshold_locked": mcfg.get("threshold_locked"),
            "threshold_source": mcfg.get("threshold_source"),
            "labels": mcfg.get("labels"),
            "semantic_labels": sorted(semantic["labels"]),
        },
        "semantic": semantic,
        "latency_ms": {k: v for k, v in latency.items() if k != "per_sample_ms"},
        "throughput": throughput_from_latency(latency),
        "cold_start": res["cold_start"],
        "resources": {
            **res["memory"],
            "model_size_mb": model_size_mb(mcfg.get("model_name")),
        },
    }

    out = Path(args.out_dir)
    save_json(out / "raw" / "semantic" / f"{model_name}_predictions.json", res["raw_records"])
    save_json(out / "metrics" / "semantic" / f"{model_name}_metrics.json", summary)
    save_json(out / "metrics" / "semantic" / f"{model_name}_latency.json",
              {"model": model_name, "per_sample_ms": latency["per_sample_ms"]})
    update_semantic_leaderboard(out / "semantic_leaderboard.csv", summary)

    rm = semantic["relaxed_micro"]
    print(
        f"[semantic:{model_name}] relaxed micro P={rm['precision']:.3f} "
        f"R={rm['recall']:.3f} F1={rm['f1']:.3f}  "
        f"PERSON R={semantic['person']['recall']:.3f}  "
        f"LOCATION R={semantic['location']['recall']:.3f}  "
        f"exactF1={semantic['exact_micro']['f1']:.3f}  P95={latency['p95_ms']:.2f}ms"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    run(args.model, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
