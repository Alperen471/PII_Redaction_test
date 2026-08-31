from evaluation.metrics import compute_metrics


def rec(rid, text, entities, dtype="Clean PII"):
    return {"id": rid, "dataset_type": dtype, "text": text, "entities": entities}


def ent(label, start, end, text):
    return {"label": label, "start": start, "end": end, "text": text}


def pr(label, start, end, score=1.0):
    return {"label": label, "start": start, "end": end, "score": score, "text": "x"}


def test_micro_relaxed_counts_and_prf():
    records = [
        rec("PII-0000", "a" * 50, [ent("PERSON", 0, 10, "aaaaaaaaaa"),
                                   ent("PHONE", 20, 30, "aaaaaaaaaa")]),
        rec("PII-0001", "b" * 50, [ent("EMAIL", 0, 10, "bbbbbbbbbb")]),
    ]
    preds = {
        "PII-0000": [pr("PERSON", 2, 8), pr("PHONE", 20, 30), pr("TCKN", 40, 50)],
        "PII-0001": [],
    }
    m = compute_metrics(records, preds)
    # TP: PERSON(overlap) + PHONE(exact) = 2 ; FP: TCKN = 1 ; FN: EMAIL = 1
    assert m.micro_relaxed["tp"] == 2
    assert m.false_positive_count == 1
    assert m.false_negative_count == 1
    assert abs(m.micro_relaxed["precision"] - 2 / 3) < 1e-9
    assert abs(m.micro_relaxed["recall"] - 2 / 3) < 1e-9
    # exact: only PHONE
    assert m.micro_exact["tp"] == 1
    assert m.partial_span_count == 1  # PERSON matched but not exact


def test_fp_charged_to_pred_label_fn_to_gold_label():
    records = [rec("PII-0000", "z" * 40, [ent("PERSON", 0, 10, "zzzzzzzzzz")])]
    preds = {"PII-0000": [pr("LOCATION", 0, 10)]}  # wrong label -> no match
    m = compute_metrics(records, preds)
    assert m.per_label["PERSON"]["relaxed"]["fn"] == 1
    assert m.per_label["LOCATION"]["relaxed"]["fp"] == 1
    assert m.micro_relaxed["tp"] == 0


def test_out_of_scope_gold_ignored():
    records = [rec("PII-0000", "q" * 20,
                   [ent("ORGANIZATION", 0, 5, "qqqqq"),
                    ent("PERSON", 6, 11, "qqqqq")])]
    preds = {"PII-0000": [pr("PERSON", 6, 11)]}
    m = compute_metrics(records, preds)
    assert m.gold_entity_count == 1
    assert m.micro_relaxed["recall"] == 1.0


def test_macro_supported_excludes_low_support():
    # PERSON support 2 (low), EMAIL support 40 via repetition
    records = [rec("PII-0000", "c" * 10, [ent("PERSON", 0, 5, "ccccc")])]
    records += [
        rec(f"PII-{i:04d}", "d" * 10, [ent("EMAIL", 0, 5, "ddddd")])
        for i in range(1, 41)
    ]
    preds = {r["id"]: [] for r in records}
    m = compute_metrics(records, preds)
    assert m.per_label["PERSON"]["low_support"] is True
    assert m.per_label["EMAIL"]["low_support"] is False
    assert "EMAIL" in m.macro_supported["labels"]
    assert "PERSON" not in m.macro_supported["labels"]
