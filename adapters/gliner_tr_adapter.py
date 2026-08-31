"""Turkish GLiNER PII adapter (plan v4 §5.4)."""
from __future__ import annotations

from adapters._gliner_base import GlinerBaseAdapter


class GlinerTrAdapter(GlinerBaseAdapter):
    name = "gliner_tr"
    default_model = "omeryentur/gliner-pam-pii-large"
