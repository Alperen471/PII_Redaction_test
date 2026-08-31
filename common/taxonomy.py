"""Label taxonomy + normalization (plan v4 §16.1, §8.4.1).

All adapter/model outputs are normalized to a canonical label set before they
reach the evaluator. Labels that normalize to something outside
``EVALUATION_LABELS`` (e.g. ``ORGANIZATION``) are kept in the raw prediction
dump but filtered out before the primary metric is computed.
"""
from __future__ import annotations

import json
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_TAXONOMY_PATH = _DATA_DIR / "taxonomy.json"

# Strips IOB / BILOU style prefixes: "B-PER", "I-PERSON", "L-LOC", ...
_IOB_PREFIX = re.compile(r"^(?:[BILUES])-", re.IGNORECASE)


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(_TAXONOMY_PATH, "r", encoding="utf-8") as fh:
        return json.load(fh)


@lru_cache(maxsize=1)
def canonical_labels() -> tuple[str, ...]:
    return tuple(_load()["canonical_labels"])


@lru_cache(maxsize=1)
def evaluation_labels() -> frozenset[str]:
    return frozenset(_load()["evaluation_labels"])


@lru_cache(maxsize=1)
def out_of_scope_labels() -> frozenset[str]:
    return frozenset(_load().get("out_of_scope_labels", ()))


@lru_cache(maxsize=1)
def low_support_threshold() -> int:
    return int(_load().get("low_support_threshold", 30))


@lru_cache(maxsize=1)
def _alias_map() -> dict[str, str]:
    raw = _load()["aliases"]
    out: dict[str, str] = {}
    for k, v in raw.items():
        out[_alias_key(k)] = v
    # Canonical labels always map to themselves.
    for lab in _load()["canonical_labels"]:
        out[_alias_key(lab)] = lab
    for lab in _load().get("out_of_scope_labels", ()):
        out[_alias_key(lab)] = lab
    return out


def _alias_key(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).strip().lower()
    text = _IOB_PREFIX.sub("", text)
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_label(raw_label: str) -> str:
    """Map a model-specific label to a canonical benchmark label.

    Unknown labels are returned upper-cased and space/dash-collapsed to an
    underscore form so they remain visible in the raw dump and are then dropped
    by :func:`is_in_scope`.
    """
    key = _alias_key(raw_label)
    mapped = _alias_map().get(key)
    if mapped is not None:
        return mapped
    return re.sub(r"\s+", "_", key).upper() or "UNKNOWN"


def is_in_scope(label: str) -> bool:
    """True if ``label`` (already canonical) is part of the primary evaluation."""
    return label in evaluation_labels()


EVALUATION_LABELS: frozenset[str] = evaluation_labels()
LOW_SUPPORT_THRESHOLD: int = low_support_threshold()
<<<<<<< HEAD


# Frozen canonical -> GLiNER prompt phrasing (plan v5 §12: labels must be
# specific, mapping locked before the run). Every value must round-trip back
# through ``normalize_label`` to its key -- guarded by test_taxonomy.
GLINER_PROMPT: dict[str, str] = {
    "PERSON": "person",
    "LOCATION": "location",
    "ADDRESS": "address",
    "CUSTOMER_ID": "customer number",
    "POLICY_ID": "insurance policy number",
    "CLAIM_ID": "insurance claim number",
    "VEHICLE_PLATE": "vehicle license plate",
    "DATE_OF_BIRTH": "date of birth",
    "TCKN": "tc kimlik no",
    "IBAN": "iban",
    "EMAIL": "email address",
    "PHONE": "phone number",
    "CREDIT_CARD": "credit card number",
}


def gliner_prompt_labels(canonical_labels) -> list[str]:
    """Map canonical labels to their frozen GLiNER prompt phrasings, order kept."""
    return [GLINER_PROMPT[lbl] for lbl in canonical_labels]
=======
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
