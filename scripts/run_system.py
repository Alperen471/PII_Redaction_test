"""System-Level PII benchmark for one system (plan v5 §6, §7, §26).

    python -m scripts.run_system --system regex_gliner_tr
    python -m scripts.run_system --system regex_only --limit 50
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adapters import get_adapter  # noqa: E402
from adapters.composite_adapter import CompositePIIAdapter  # noqa: E402
from benchmarks.model_size import model_size_mb  # noqa: E402
from benchmarks.throughput import throughput_from_latency  # noqa: E402
from common.io import EXPECTED_SAMPLES, load_dataset, save_json  # noqa: E402
from common.taxonomy import gliner_prompt_labels, is_in_scope  # noqa: E402
from config import load_config, load_systems_config, model_config  # noqa: E402
from evaluation.system_metrics import compute_system_metrics  # noqa: E402
from scripts._runner import run_inference  # noqa: E402
from scripts.leaderboard import update_system_leaderboard  # noqa: E402
from tokenization.tokenizer import tokenize  # noqa: E402


def _build_system(system_name: str, scfg: dict, cfg: dict, device_override):
    sysdef = scfg["systems"][system_name]
    structured_labels = set(scfg["structured_labels"])
    # GLiNER is allowed to contribute PERSON/LOCATION/ADDRESS + the structured-gap
    # labels; regex still wins any overlap (label-domain-priority merge).
    gliner_canonical = list(scfg["semantic_labels"]) + list(scfg.get("semantic_gap_labels", []))
    gliner_labels = set(gliner_canonical)

    smcfg = model_config(cfg, sysdef["structured_adapter"])
    if device_override:
        smcfg["device"] = device_override
    structured = get_adapter(smcfg["adapter"], smcfg)

    semantic = None
    sem_name = sysdef.get("semantic_adapter")
    sem_device = None
    if sem_name:
        semcfg = model_config(cfg, sem_name)
        if device_override:
            semcfg["device"] = device_override
        # frozen, specific GLiNER prompt for the system's allowed label set (§12)
        semcfg["labels"] = gliner_prompt_labels(gliner_canonical)
        sem_device = str(semcfg.get("device", "cpu"))
        semantic = get_adapter(semcfg["adapter"], semcfg)

    composite = CompositePIIAdapter(
        structured, semantic, structured_labels, gliner_labels, name=system_name
    )
    device = sem_device or str(smcfg.get("device", "cpu"))
    meta = {
        "structured_adapter": sysdef["structured_adapter"],
        "semantic_adapter": sem_name,
        "semantic_model_name": (semcfg.get("model_name") if sem_name else None),
        "threshold": (semcfg.get("threshold") if sem_name else None),
        "threshold_locked": (semcfg.get("threshold_locked") if sem_name else None),
        "threshold_source": (semcfg.get("threshold_source") if sem_name else None),
        "semantic_prompt_labels": (semcfg.get("labels") if sem_name else None),
    }
    return composite, device, meta


def run(system_name: str, args) -> dict:
    cfg = load_config(args.config)
    scfg = load_systems_config(args.systems)
    if system_name not in scfg["systems"]:
        raise KeyError(f"Unknown system '{system_name}'. Known: {sorted(scfg['systems'])}")

    warmup = int(cfg.get("run", {}).get("warmup_runs", 10))
    dataset_path = args.dataset or cfg["dataset"].get("path")
    dataset = load_dataset(dataset_path)
    if args.limit:
        dataset = dataset[: args.limit]
    elif len(dataset) != EXPECTED_SAMPLES:
        print(f"WARNING: {len(dataset)} samples, expected {EXPECTED_SAMPLES}")

    composite, device, meta = _build_system(system_name, scfg, cfg, args.device)
    res = run_inference(composite, dataset, device=device, warmup=warmup)
    composite.unload()

    # tokenization -> applied spans feed leakage / coverage (plan v5 §16, §17)
    applied_by_id: dict[str, list] = {}
    for sample in dataset:
        in_scope = [p for p in res["preds_by_id"].get(sample["id"], []) if is_in_scope(p["label"])]
        _safe, _map, applied = tokenize(sample["text"], in_scope)
        applied_by_id[sample["id"]] = applied

    system = compute_system_metrics(dataset, res["preds_by_id"], applied_by_id)
    latency = res["latency"]
    size = None if meta["semantic_adapter"] is None else model_size_mb(meta["semantic_model_name"])

    summary = {
        "benchmark": "system",
        "system_name": system_name,
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
            "structured_labels": sorted(scfg["structured_labels"]),
            "semantic_labels": sorted(scfg["semantic_labels"]),
            "semantic_gap_labels": sorted(scfg.get("semantic_gap_labels", [])),
            "merge_policy": "label-domain-priority (regex territory wins overlaps)",
            **meta,
        },
        "system": system,
        "latency_ms": {k: v for k, v in latency.items() if k != "per_sample_ms"},
        "throughput": throughput_from_latency(latency),
        "cold_start": res["cold_start"],
        "resources": {**res["memory"], "model_size_mb": size},
    }

    out = Path(args.out_dir)
    save_json(out / "raw" / "system" / f"{system_name}_predictions.json", res["raw_records"])
    save_json(out / "metrics" / "system" / f"{system_name}_metrics.json", summary)
    save_json(out / "metrics" / "system" / f"{system_name}_latency.json",
              {"system": system_name, "per_sample_ms": latency["per_sample_ms"]})
    update_system_leaderboard(out / "system_leaderboard.csv", summary)

    rm = system["relaxed_micro"]
    hn = system["hard_negative"]["hard_negative_category"]
    print(
        f"[system:{system_name}] relaxed micro P={rm['precision']:.3f} "
        f"R={rm['recall']:.3f} F1={rm['f1']:.3f}  "
        f"leakage={system['leakage']['pii_leakage_rate']:.3f}  "
        f"cov={system['leakage']['tokenization_coverage']:.3f}  "
        f"exactF1={system['exact_micro']['f1']:.3f}  "
        f"hardneg_FP_rate={hn['false_positive_rate']:.3f}  P95={latency['p95_ms']:.2f}ms"
    )
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", required=True)
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--systems", default="config/systems.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    run(args.system, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
