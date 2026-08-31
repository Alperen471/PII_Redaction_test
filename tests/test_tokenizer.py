import unicodedata

from tokenization.tokenizer import resolve_overlaps, tokenize


def sp(label, start, end, score=1.0):
    return {"label": label, "start": start, "end": end, "score": score, "text": "x"}


def test_same_surface_same_token_and_suffix_outside_span():
    text = "Ahmet Yilmaz bugun aradi. Ahmet Yilmaz'in telefonu degisti."
    p1 = text.index("Ahmet Yilmaz")
    p2 = text.index("Ahmet Yilmaz", p1 + 1)
    preds = [sp("PERSON", p1, p1 + 12), sp("PERSON", p2, p2 + 12)]
    safe, tmap, applied = tokenize(text, preds)
    assert safe.count("<PERSON_1>") == 2
    assert "<PERSON_2>" not in safe
    assert safe.startswith("<PERSON_1> bugun aradi.")
    assert "<PERSON_1>'in telefonu" in safe  # Turkish suffix preserved (plan v4 §4.3)


def test_token_key_is_nfc_normalized():
    # same grapheme, different Unicode composition -> one token
    nfc = unicodedata.normalize("NFC", "Şükrü")
    nfd = unicodedata.normalize("NFD", "Şükrü")
    text = f"{nfc} ve {nfd} geldi"
    a = text.index(nfc)
    b = text.index(nfd)
    preds = [sp("PERSON", a, a + len(nfc)), sp("PERSON", b, b + len(nfd))]
    safe, tmap, applied = tokenize(text, preds)
    assert len(tmap) == 1
    assert safe.count("<PERSON_1>") == 2


def test_resolve_overlaps_keeps_higher_score():
    preds = [sp("PERSON", 0, 10, score=0.5), sp("LOCATION", 5, 15, score=0.9)]
    kept = resolve_overlaps(preds)
    assert len(kept) == 1 and kept[0]["label"] == "LOCATION"


def test_per_label_numbering():
    text = "A B C D"  # positions 0,2,4,6
    preds = [sp("PERSON", 0, 1), sp("PHONE", 2, 3), sp("PERSON", 4, 5)]
    safe, tmap, applied = tokenize(text, preds)
    assert "<PERSON_1>" in safe and "<PERSON_2>" in safe and "<PHONE_1>" in safe
