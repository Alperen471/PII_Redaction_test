"""Run the System-Level PII benchmark for every configured system (plan v5 §26).

    python -m scripts.run_all_system
    python -m scripts.run_all_system --only regex_only,regex_gliner_tr
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_systems_config  # noqa: E402
from scripts.run_system import run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config/models.yaml")
    ap.add_argument("--systems", default="config/systems.yaml")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--only", default=None)
    ap.add_argument("--continue-on-error", action="store_true")
    args = ap.parse_args()

    scfg = load_systems_config(args.systems)
    systems = list(scfg["systems"])
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        systems = [s for s in systems if s in wanted]

    failures = []
    for s in systems:
        print(f"\n=== system: {s} ===")
        try:
            run(s, args)
        except Exception:
            failures.append(s)
            traceback.print_exc()
            if not args.continue_on_error:
                return 1
    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print(f"\nsystem leaderboard -> {Path(args.out_dir) / 'system_leaderboard.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
