"""Semantic + System benchmark metric wrappers (plan v5 §5, §6, §7, §18)."""
from evaluation.semantic_metrics import compute_semantic_metrics
from evaluation.system_metrics import compute_system_metrics


def rec(rid, text, entities, dtype="Clean PII"):
    return {"id": rid, "dataset_type": dtype, "text": text, "entities": entities}


def ent(label, s, e, t):
    return {"label": label, "start": s, "end": e, "text": t}


def pr(label, s, e, score=1.0):
    return {"label": label, "start": s, "end": e, "score": score, "text": "x"}


def test_semantic_ignores_non_semantic_labels():
    records = [
        rec("PII-0000", "x" * 60, [
            ent("PERSON", 0, 10, "x" * 10),
            ent("PHONE", 20, 30, "x" * 10),      # ignored by semantic benchmark
            ent("LOCATION", 40, 48, "x" * 8),
        ])
    ]
    preds = {"PII-0000": [
        pr("PERSON", 0, 10),
        pr("PHONE", 20, 30),                     # not scored
        pr("LOCATION", 40, 44),                  # relaxed hit, not exact
        pr("TCKN", 50, 60),                      # off-subset, ignored
    ]}
    s = compute_semantic_metrics(records, preds)
    assert s["gold_entity_count"] == 2
    assert s["relaxed_micro"]["tp"] == 2
    assert s["relaxed_micro"]["fp"] == 0          # PHONE/TCKN preds excluded
    assert s["person"]["recall"] == 1.0
    assert s["location"]["f1"] == 1.0
    assert s["location"]["exact_f1"] == 0.0
    assert s["exact_micro"]["tp"] == 1


def test_system_unsupported_label_counts_as_fn():
    records = [
        rec("PII-0000", "y" * 40, [
            ent("EMAIL", 0, 10, "y" * 10),
            ent("PERSON", 20, 30, "y" * 10),      # regex_only has no PERSON detector
        ])
    ]
    preds = {"PII-0000": [pr("EMAIL", 0, 10)]}
    applied = {"PII-0000": [pr("EMAIL", 0, 10)]}
    sysm = compute_system_metrics(records, preds, applied)
    assert sysm["relaxed_micro"]["tp"] == 1
    assert sysm["relaxed_micro"]["fn"] == 1       # PERSON missed -> FN (plan v5 §7)
    assert sysm["per_label"]["PERSON"]["relaxed"]["fn"] == 1
    # EMAIL redacted by same-label span -> covered, no leak
    assert sysm["leakage"]["pii_leakage_rate"] == 0.5   # PERSON still visible


def test_system_hard_negative_analysis_runtime_rule():
    records = [
        rec("PII-0000", "no pii here", [], dtype="Ambiguous / Hard Negative"),
        rec("PII-0001", "also clean", [], dtype="Ambiguous / Hard Negative"),
        rec("PII-0002", "has one", [ent("EMAIL", 0, 3, "has")], dtype="Clean PII"),
    ]
    preds = {"PII-0000": [pr("PERSON", 0, 2)], "PII-0001": [], "PII-0002": []}
    applied = {k: [] for k in preds}
    sysm = compute_system_metrics(records, preds, applied)
    assert sysm["negative_samples"] == 2
    assert sysm["positive_samples"] == 1
    hn = sysm["hard_negative"]["hard_negative_category"]
    assert hn["count"] == 2
    assert hn["false_positive_count"] == 1
    assert hn["false_positive_rate"] == 0.5
    assert hn["clean_pass_rate"] == 0.5
    assert hn["predicted_labels"] == {"PERSON": 1}
