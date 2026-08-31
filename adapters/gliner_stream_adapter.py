"""GLiNER Stream PII adapter (plan v4 §5.6).

Streaming is NOT used in this benchmark; the model is run with plain text
inference like every other adapter.
"""
from __future__ import annotations

from adapters._gliner_base import GlinerBaseAdapter


class GlinerStreamAdapter(GlinerBaseAdapter):
    name = "gliner_stream"
    default_model = "knowledgator/gliner-stream-pii-v1.0"
