"""Shared GLiNER adapter logic (plan v4 §5.4-5.6).

The three GLiNER variants (Turkish PII / PII-Edge / Stream-PII) differ only in
default model name; label set and threshold come from config and are LOCKED
before the run (plan v4 §8.4.3). Streaming is not used -- all variants run plain
text inference (plan v4 §5.6).
"""
from __future__ import annotations

from typing import Any

from adapters.base import BasePIIAdapter
from evaluation.spans import Span

# Natural-language labels; normalize_label maps them to canonical taxonomy.
DEFAULT_LABELS = [
    "person",
    "location",
    "address",
    "phone number",
    "email",
    "national id",
    "bank account",
    "credit card",
    "customer id",
    "policy id",
    "claim id",
    "vehicle plate",
    "date of birth",
]


class GlinerBaseAdapter(BasePIIAdapter):
    default_model: str = ""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.model_name = self.config.get("model_name", self.default_model)
        self.labels = list(self.config.get("labels", DEFAULT_LABELS))
        self.threshold = float(self.config.get("threshold", 0.5))
        self.threshold_locked = bool(self.config.get("threshold_locked", True))
        self.flat_ner = bool(self.config.get("flat_ner", True))
        self.multi_label = bool(self.config.get("multi_label", False))
        self._model = None

    def load(self) -> None:
        from gliner import GLiNER  # lazy

        self._model = GLiNER.from_pretrained(self.model_name)
        if self.device.startswith("cuda"):
            self._model = self._model.to("cuda")
        self._model.eval()
        self._loaded = True

    def predict(self, text: str) -> list[Span]:
        if not self._loaded:
            self.load()
        ents = self._model.predict_entities(
            text,
            self.labels,
            threshold=self.threshold,
            flat_ner=self.flat_ner,
            multi_label=self.multi_label,
        )
        raw = [
            {
                "text": text[e["start"]:e["end"]],
                "label": e["label"],
                "start": int(e["start"]),
                "end": int(e["end"]),
                "score": float(e.get("score", 1.0)),
            }
            for e in ents
        ]
        return self.finalize(text, raw)

    def unload(self) -> None:
        self._model = None
        super().unload()
