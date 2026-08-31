"""Benchmark every model in the config, in the plan's order (plan v4 §16.6, §18).

Usage:
    python -m scripts.run_all
    python -m scripts.run_all --only regex,berturk --device cuda
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config  # noqa: E402
from scripts.run_model import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--only", default=None, help="comma-separated subset of models")
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    order = cfg.get("run", {}).get("order") or list(cfg["models"])
    if args.only:
        wanted = {m.strip() for m in args.only.split(",")}
        order = [m for m in order if m in wanted]

    failures = []
    for model in order:
        print(f"\n=== {model} ===")
        try:
            run(model, args)
        except Exception:
            failures.append(model)
            traceback.print_exc()
            if not args.continue_on_error:
                return 1
    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print(f"\nleaderboard -> {Path(args.out_dir) / 'leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
