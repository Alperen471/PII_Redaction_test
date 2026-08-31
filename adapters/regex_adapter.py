"""Regex + Validator baseline (plan v4 §5.1, §7).

Deterministic, CPU-only, no model download. Emits a span only when its format
validator passes; validated spans get ``score = 1.0``.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from adapters.base import BasePIIAdapter
from adapters.validators import VALIDATORS
from common.taxonomy import normalize_label
from evaluation.spans import Span

_DEFAULT_PATTERNS = Path(__file__).resolve().parent.parent / "config" / "patterns.yaml"

# pattern-key -> canonical label
_KEY_TO_LABEL = {
    "tckn": "TCKN",
    "phone": "PHONE",
    "email": "EMAIL",
    "iban": "IBAN",
    "credit_card": "CREDIT_CARD",
    "vehicle_plate": "VEHICLE_PLATE",
    "date_of_birth": "DATE_OF_BIRTH",
    "customer_id": "CUSTOMER_ID",
    "policy_id": "POLICY_ID",
    "claim_id": "CLAIM_ID",
}


class RegexAdapter(BasePIIAdapter):
    name = "regex"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.device = "cpu"
        self.patterns_path = Path(
            self.config.get("patterns_path", _DEFAULT_PATTERNS)
        )
        self._rules: list[dict] = []

    def load(self) -> None:
        with open(self.patterns_path, "r", encoding="utf-8") as fh:
            spec = yaml.safe_load(fh)
        rules: list[dict] = []
        for key, cfg in spec.items():
            if not cfg or not cfg.get("enabled"):
                continue
            pattern = cfg.get("regex") or ""
            if not pattern:
                continue
            rules.append(
                {
                    "label": normalize_label(_KEY_TO_LABEL.get(key, key)),
                    "regex": re.compile(pattern),
                    "validator": VALIDATORS.get(cfg.get("validator", "none"), VALIDATORS["none"]),
                    "score": float(cfg.get("score", 1.0)),
                    "require_context": bool(cfg.get("require_context", False)),
                    "context_window": int(cfg.get("context_window", 40)),
                    "context_keywords": [k.lower() for k in cfg.get("context_keywords", [])],
                }
            )
        self._rules = rules
        self._loaded = True

    def _has_context(self, text: str, start: int, end: int, rule: dict) -> bool:
        if not rule["require_context"]:
            return True
        w = rule["context_window"]
        window = text[max(0, start - w): min(len(text), end + w)].lower()
        return any(k in window for k in rule["context_keywords"])

    def predict(self, text: str) -> list[Span]:
        if not self._loaded:
            self.load()
        raw: list[dict] = []
        for rule in self._rules:
            for m in rule["regex"].finditer(text):
                s, e = m.start(), m.end()
                surface = text[s:e]
                if not rule["validator"](surface):
                    continue
                if not self._has_context(text, s, e, rule):
                    continue
                raw.append(
                    {
                        "text": surface,
                        "label": rule["label"],
                        "start": s,
                        "end": e,
                        "score": rule["score"],
                    }
                )
        return self.finalize(text, raw)
