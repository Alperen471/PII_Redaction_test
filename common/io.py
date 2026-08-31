"""Dataset loading and result persistence (plan v4 §8.6, §15)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Canonical dataset facts from the plan (v4 §8.1, §8.3, §8.4, §8.2).
EXPECTED_SAMPLES = 1000
FIRST_ID = "PII-0000"
LAST_ID = "PII-0999"
EXPECTED_ENTITY_TOTAL = 1685
EXPECTED_LABEL_SUPPORT = {
    "ADDRESS": 110,
    "CLAIM_ID": 125,
    "CREDIT_CARD": 10,
    "CUSTOMER_ID": 110,
    "DATE_OF_BIRTH": 50,
    "EMAIL": 131,
    "IBAN": 121,
    "LOCATION": 45,
    "PERSON": 410,
    "PHONE": 171,
    "POLICY_ID": 166,
    "TCKN": 130,
    "VEHICLE_PLATE": 106,
}
EXPECTED_DATASET_TYPE_COUNTS = {
    "Clean PII": 300,
    "STT-like PII": 250,
    "Ambiguous / Hard Negative": 150,
    "Insurance-domain": 200,
    "Multi-PII / Complex": 100,
}

DEFAULT_DATASET_PATH = "data/pii_benchmark_merged_fixed.json"


def load_dataset(path: str | os.PathLike = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Load the benchmark dataset.

    The file is a single JSON array (not JSONL), so ``json.load`` is used
    directly (plan v4 §8.6). The raw text is NOT normalized here; span indices
    are evaluated against this exact string (plan v4 §9.2).
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Benchmark dataset not found at '{p}'. Place the frozen 1000-record "
            f"file there (plan v4 §8.1)."
        )
    with open(p, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a JSON array (plan v4 §8.1).")
    return data


def save_json(path: str | os.PathLike, obj: Any) -> None:
    """Atomically write ``obj`` as UTF-8 JSON (ensure_ascii=False)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=p.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
        os.replace(tmp, p)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
