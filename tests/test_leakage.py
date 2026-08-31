from evaluation.leakage import leakage_report


def rec(rid, text, entities):
    return {"id": rid, "dataset_type": "Clean PII", "text": text, "entities": entities}


def ent(label, start, end, text):
    return {"label": label, "start": start, "end": end, "text": text}


def sp(label, start, end):
    return {"label": label, "start": start, "end": end, "score": 1.0, "text": "x"}


def test_full_cover_same_label_is_covered_not_leaked():
    r = rec("PII-0000", "Ahmet Yilmaz aradi", [ent("PERSON", 0, 12, "Ahmet Yilmaz")])
    out = leakage_report([r], {"PII-0000": [sp("PERSON", 0, 12)]})
    assert out["pii_leakage_rate"] == 0.0
    assert out["tokenization_coverage"] == 1.0
    assert out["wrong_label_tokenization_count"] == 0


def test_partial_cover_leaks():
    r = rec("PII-0000", "Ahmet Yilmaz aradi", [ent("PERSON", 0, 12, "Ahmet Yilmaz")])
    out = leakage_report([r], {"PII-0000": [sp("PERSON", 0, 5)]})  # only "Ahmet"
    assert out["leaked_count"] == 1
    assert out["pii_leakage_rate"] == 1.0
    assert out["tokenization_coverage"] == 0.0


def test_wrong_label_cover_is_error_but_not_leak():
    r = rec("PII-0000", "1234567", [ent("TCKN", 0, 7, "1234567")])
    out = leakage_report([r], {"PII-0000": [sp("PHONE", 0, 7)]})
    assert out["leaked_count"] == 0
    assert out["wrong_label_tokenization_count"] == 1
    assert out["tokenization_coverage"] == 0.0
