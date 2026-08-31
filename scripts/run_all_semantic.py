"""Run the Semantic NER benchmark for every semantic model (plan v5 §25).

    python -m scripts.run_all_semantic
    python -m scripts.run_all_semantic --only gliner_tr,stanza --device cpu
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config  # noqa: E402
from scripts.run_semantic import run  # noqa: E402
from dotenv import load_dotenv

load_dotenv()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--only", default=None)
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    models = cfg.get("semantic_models") or [
        m for m in cfg["models"] if m != "regex"
    ]
    if args.only:
        wanted = {m.strip() for m in args.only.split(",")}
        models = [m for m in models if m in wanted]

    failures = []
    for m in models:
        print(f"\n=== semantic: {m} ===")
        try:
            run(m, args)
        except Exception:
            failures.append(m)
            traceback.print_exc()
            if not args.continue_on_error:
                return 1
    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print(f"\nsemantic leaderboard -> {Path(args.out_dir) / 'semantic_leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
