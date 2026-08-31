"""BERTurk NER adapter (plan v4 §5.3).

Model: ``savasy/bert-base-turkish-ner-cased`` via a transformers
token-classification pipeline with word-level aggregation. Fast-tokenizer
offsets are character indices into the input string, i.e. the original text
(plan v4 §9.1) -- no back-mapping needed.
"""
from __future__ import annotations

from typing import Any

from adapters.base import BasePIIAdapter
from evaluation.spans import Span


class BERTurkAdapter(BasePIIAdapter):
    name = "berturk"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.model_name = self.config.get(
            "model_name", "savasy/bert-base-turkish-ner-cased"
        )
        self.aggregation = self.config.get("aggregation_strategy", "simple")
        self._pipe = None

    def load(self) -> None:
        from transformers import (  # lazy
            AutoModelForTokenClassification,
            AutoTokenizer,
            pipeline,
        )

        device = 0 if self.device.startswith("cuda") else -1
        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        mdl = AutoModelForTokenClassification.from_pretrained(self.model_name)
        self._pipe = pipeline(
            "token-classification",
            model=mdl,
            tokenizer=tok,
            aggregation_strategy=self.aggregation,
            device=device,
        )
        self._loaded = True

    def predict(self, text: str) -> list[Span]:
        if not self._loaded:
            self.load()
        raw = []
        for ent in self._pipe(text):
            raw.append(
                {
                    "text": text[ent["start"]:ent["end"]],
                    "label": ent.get("entity_group", ent.get("entity", "")),
                    "start": int(ent["start"]),
                    "end": int(ent["end"]),
                    "score": float(ent.get("score", 1.0)),
                }
            )
        return self.finalize(text, raw)

    def unload(self) -> None:
        self._pipe = None
        super().unload()
