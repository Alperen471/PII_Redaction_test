"""GLiNER PII Edge adapter (plan v4 §5.5)."""
from __future__ import annotations

from adapters._gliner_base import GlinerBaseAdapter


class GlinerEdgeAdapter(GlinerBaseAdapter):
    name = "gliner_edge"
    default_model = "knowledgator/gliner-pii-edge-v1.0"
