"""Inspect raw model / system predictions against the ground truth.

    # legacy full-taxonomy single model
    python -m scripts.show_predictions --model regex --id PII-0007
    python -m scripts.show_predictions --model gliner_tr --filter fp --limit 5

    # semantic NER benchmark (PERSON / LOCATION only)
    python -m scripts.show_predictions --semantic gliner_tr --id PII-0007

    # system benchmark
    python -m scripts.show_predictions --system regex_gliner_tr --filter fn --limit 10

Filters: tp | fp | fn | partial | leak | any (default: show every requested sample)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json  # noqa: E402

from common.io import load_dataset  # noqa: E402
from common.taxonomy import is_in_scope  # noqa: E402
from evaluation.alignment import align_relaxed  # noqa: E402
from evaluation.semantic_metrics import SEMANTIC_LABELS  # noqa: E402
from evaluation.spans import exact_match  # noqa: E402

RAW = {
    "model": "results/raw/{name}_predictions.json",
    "semantic": "results/raw/semantic/{name}_predictions.json",
    "system": "results/raw/system/{name}_predictions.json",
}


def _load_raw(kind: str, name: str) -> dict[str, list]:
    path = Path(RAW[kind].format(name=name))
    if not path.is_file():
        sys.exit(f"not found: {path}  (run the matching benchmark first)")
    with open(path, "r", encoding="utf-8") as fh:
        rows = json.load(fh)
    out = {}
    for r in rows:
        preds = r.get("predictions_in_scope") or r.get("predictions") or []
        out[r["id"]] = preds
    return out


def _fmt(sp: dict) -> str:
    sc = sp.get("score")
    tail = f"  ({sc:.2f})" if isinstance(sc, (int, float)) else ""
    return f"{sp['label']:<14} [{sp['start']:>4},{sp['end']:>4})  {sp['text']!r}{tail}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", help="legacy full-taxonomy raw (results/raw/)")
    g.add_argument("--semantic", help="semantic benchmark raw (results/raw/semantic/)")
    g.add_argument("--system", help="system benchmark raw (results/raw/system/)")
    ap.add_argument("--dataset", default="data/pii_benchmark_merged_fixed.json")
    ap.add_argument("--id", action="append", help="specific sample id(s)")
    ap.add_argument("--filter", default="any", choices=["any", "tp", "fp", "fn", "partial", "leak"])
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    kind, name = next(
        (k, v) for k, v in (("model", args.model), ("semantic", args.semantic), ("system", args.system))
        if v
    )
    subset = SEMANTIC_LABELS if kind == "semantic" else None

    dataset = load_dataset(args.dataset)
    raw = _load_raw(kind, name)

    wanted_ids = set(args.id) if args.id else None
    shown = 0
    for sample in dataset:
        sid = sample["id"]
        if wanted_ids is not None and sid not in wanted_ids:
            continue

        golds = [
            {"label": e["label"], "start": e["start"], "end": e["end"], "text": e["text"]}
            for e in sample.get("entities", [])
            if is_in_scope(e["label"]) and not e.get("out_of_scope")
            and (subset is None or e["label"] in subset)
        ]
        preds = [
            p for p in raw.get(sid, [])
            if is_in_scope(p["label"]) and (subset is None or p["label"] in subset)
        ]

        pairs = align_relaxed(preds, golds)
        mp = {i for i, _ in pairs}
        mg = {j for _, j in pairs}
        tp = [(preds[i], golds[j]) for i, j in pairs]
        partial = [(p, g) for p, g in tp if not exact_match(p, g)]
        fp = [preds[i] for i in range(len(preds)) if i not in mp]
        fn = [golds[j] for j in range(len(golds)) if j not in mg]
        leak = [g for g in fn] + [g for p, g in partial]

        buckets = {"any": True, "tp": tp, "fp": fp, "fn": fn, "partial": partial, "leak": leak}
        if not buckets[args.filter]:
            continue

        shown += 1
        print(f"\n=== {sid}  [{sample.get('dataset_type')}] ===")
        print(f"text: {sample['text']}")
        print("GOLD:")
        for g in golds:
            print(f"  {_fmt(g)}")
        print("PRED:")
        for p in preds:
            print(f"  {_fmt(p)}")
        if tp:
            print("TP:")
            for p, gg in tp:
                mark = "exact" if exact_match(p, gg) else "PARTIAL"
                print(f"  {gg['label']:<14} gold[{gg['start']},{gg['end']}) ~ pred[{p['start']},{p['end']})  {mark}")
        if fn:
            print("FN (missed):")
            for g in fn:
                print(f"  {_fmt(g)}")
        if fp:
            print("FP (spurious):")
            for p in fp:
                print(f"  {_fmt(p)}")

        if shown >= args.limit:
            break

    if shown == 0:
        print("(no matching samples)")
    else:
        print(f"\n-- shown {shown} sample(s) from {kind}:{name}, filter={args.filter}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
