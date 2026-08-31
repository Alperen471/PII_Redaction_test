from adapters.base import BasePIIAdapter
from adapters.composite_adapter import CompositePIIAdapter, merge_predictions


def sp(label, start, end, score=1.0):
    return {"label": label, "start": start, "end": end, "score": score, "text": "x"}


class _Fake(BasePIIAdapter):
    def __init__(self, spans):
        super().__init__({})
        self._spans = spans
        self.loaded = False

    def load(self):
        self.loaded = True

    def predict(self, text):
        return list(self._spans)


# --- merge_predictions: label-domain priority (regex territory wins) --------- #
def test_merge_keeps_higher_confidence_on_identical_span():
    merged = merge_predictions([sp("PERSON", 0, 5, 0.4)], [sp("PERSON", 0, 5, 0.9)])
    assert len(merged) == 1 and merged[0]["score"] == 0.9


def test_merge_regex_territory_suppresses_overlapping_gliner_any_label():
    # GLiNER "PERSON" over a regex CUSTOMER_ID span -> dropped (plan: cross-domain,
    # structured wins). This is the CUST-497346 -> PERSON false positive fix.
    merged = merge_predictions(
        [sp("CUSTOMER_ID", 15, 26, 1.0)], [sp("PERSON", 15, 26, 0.98)]
    )
    assert [p["label"] for p in merged] == ["CUSTOMER_ID"]


def test_merge_gliner_gap_fill_is_kept():
    # regex silent here -> GLiNER's structured-domain guess survives
    merged = merge_predictions(
        [sp("EMAIL", 0, 10)], [sp("CUSTOMER_ID", 20, 30, 0.7), sp("PERSON", 40, 52, 0.9)]
    )
    assert sorted(p["label"] for p in merged) == ["CUSTOMER_ID", "EMAIL", "PERSON"]


def test_merge_partial_overlap_with_territory_also_drops():
    merged = merge_predictions([sp("IBAN", 10, 36)], [sp("PERSON", 30, 45, 0.8)])
    assert [p["label"] for p in merged] == ["IBAN"]


# --- CompositePIIAdapter --------------------------------------------------- #
def test_composite_routes_labels_and_suppresses_structured_region():
    structured = _Fake([sp("CUSTOMER_ID", 0, 11), sp("PERSON", 20, 25)])   # PERSON not a structured label
    semantic = _Fake([sp("PERSON", 0, 11), sp("PERSON", 40, 52), sp("ORGANIZATION", 60, 63)])
    comp = CompositePIIAdapter(
        structured, semantic,
        structured_labels={"CUSTOMER_ID", "PHONE", "EMAIL"},
        semantic_labels={"PERSON", "LOCATION", "ADDRESS", "CUSTOMER_ID"},
        name="regex_gliner_tr",
    )
    comp.load()
    assert structured.loaded and semantic.loaded
    out = comp.predict("...")
    # CUSTOMER_ID from regex; PERSON[0,11] suppressed (regex territory);
    # PERSON[40,52] kept; regex PERSON[20,25] dropped (not a structured label);
    # ORG dropped (not in semantic_labels)
    assert sorted((p["label"], p["start"]) for p in out) == [("CUSTOMER_ID", 0), ("PERSON", 40)]


def test_composite_none_semantic_adapter():
    comp = CompositePIIAdapter(
        _Fake([sp("EMAIL", 0, 5)]), None,
        structured_labels={"EMAIL"}, semantic_labels={"PERSON"}, name="regex_only",
    )
    comp.load()
    assert [p["label"] for p in comp.predict("x")] == ["EMAIL"]
