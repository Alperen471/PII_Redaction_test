"""Stanza Turkish NER adapter (plan v4 §5.2).

Heavy dependency (``stanza`` + language model). Code is runtime-identical on CPU
and CUDA; only ``device`` changes. Not exercised in the offline dev environment.
"""
from __future__ import annotations

from typing import Any

from adapters.base import BasePIIAdapter
from evaluation.spans import Span


class StanzaAdapter(BasePIIAdapter):
    name = "stanza"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        super().__init__(config)
        self.lang = self.config.get("lang", "tr")
        self._nlp = None

    def load(self) -> None:
        import stanza  # lazy: only needed on the benchmark machine

        use_gpu = self.device.startswith("cuda")
        if self.config.get("download", False):
            stanza.download(self.lang, processors="tokenize,ner", verbose=False)
        self._nlp = stanza.Pipeline(
            lang=self.lang,
            processors="tokenize,ner",
            use_gpu=use_gpu,
            tokenize_no_ssplit=False,
            verbose=False,
        )
        self._loaded = True

    def predict(self, text: str) -> list[Span]:
        if not self._loaded:
            self.load()
        doc = self._nlp(text)
        raw = [
            {
                "text": ent.text,
                "label": ent.type,
                # Stanza char offsets are Python str code-point indices into `text`
                "start": ent.start_char,
                "end": ent.end_char,
                "score": 1.0,
            }
            for ent in doc.ents
        ]
        return self.finalize(text, raw)

    def unload(self) -> None:
        self._nlp = None
        super().unload()
