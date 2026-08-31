from common.taxonomy import (
    EVALUATION_LABELS,
<<<<<<< HEAD
    GLINER_PROMPT,
    gliner_prompt_labels,
=======
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
    is_in_scope,
    normalize_label,
)


def test_alias_and_iob_normalization():
    assert normalize_label("B-PER") == "PERSON"
    assert normalize_label("i-person") == "PERSON"
<<<<<<< HEAD
    assert normalize_label("phone number") == "PHONE"
    assert normalize_label("vehicle plate") == "VEHICLE_PLATE"
    assert normalize_label("license plate") == "VEHICLE_PLATE"
    assert normalize_label("PERSON") == "PERSON"
    # specific TCKN label is fine (plan v5 §12)
    assert normalize_label("Turkish national identity number (TCKN)") == "TCKN"


def test_risky_generic_mappings_are_not_applied():
    # plan v5 §12: generic 'national id' / 'bank account' must NOT auto-map
    assert normalize_label("national id") != "TCKN"
    assert normalize_label("bank account") != "IBAN"
    assert is_in_scope(normalize_label("national id")) is False
    assert is_in_scope(normalize_label("bank account")) is False
=======
    assert normalize_label("national id") == "TCKN"
    assert normalize_label("phone number") == "PHONE"
    assert normalize_label("bank account") == "IBAN"
    assert normalize_label("vehicle plate") == "VEHICLE_PLATE"
    assert normalize_label("PERSON") == "PERSON"
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53


def test_organization_is_out_of_scope():
    assert normalize_label("ORG") == "ORGANIZATION"
    assert "ORGANIZATION" not in EVALUATION_LABELS
    assert is_in_scope("ORGANIZATION") is False
    assert is_in_scope("PERSON") is True


def test_unknown_label_kept_but_out_of_scope():
    lab = normalize_label("weird custom thing")
    assert lab == "WEIRD_CUSTOM_THING"
    assert is_in_scope(lab) is False


<<<<<<< HEAD
def test_gliner_prompt_round_trips_to_canonical():
    # plan v5 §12: every GLiNER prompt phrasing must normalize back to its label
    for canon, phrasing in GLINER_PROMPT.items():
        assert normalize_label(phrasing) == canon, (phrasing, normalize_label(phrasing))


def test_gliner_prompt_labels_helper_keeps_order():
    assert gliner_prompt_labels(["PERSON", "CLAIM_ID", "VEHICLE_PLATE"]) == [
        "person", "insurance claim number", "vehicle license plate",
    ]


=======
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
def test_evaluation_label_set_matches_plan():
    assert EVALUATION_LABELS == frozenset({
        "ADDRESS", "CLAIM_ID", "CREDIT_CARD", "CUSTOMER_ID", "DATE_OF_BIRTH",
        "EMAIL", "IBAN", "LOCATION", "PERSON", "PHONE", "POLICY_ID", "TCKN",
        "VEHICLE_PLATE",
    })
