<<<<<<< HEAD
from evaluation.alignment import align_exact, align_relaxed
from evaluation.spans import (
=======
from evaluation.spans import (
    align_exact,
    align_relaxed,
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
    exact_match,
    intersection_length,
    iou,
    relaxed_match,
)


def sp(label, start, end, score=1.0):
    return {"label": label, "start": start, "end": end, "score": score, "text": "x"}


def test_intersection_and_iou():
    a, b = sp("PERSON", 0, 10), sp("PERSON", 5, 15)
    assert intersection_length(a, b) == 5
    assert abs(iou(a, b) - 5 / 15) < 1e-9
    assert intersection_length(sp("X", 0, 3), sp("X", 3, 6)) == 0


def test_relaxed_vs_exact():
    g = sp("PERSON", 20, 32)
    assert relaxed_match(sp("PERSON", 20, 27), g)
    assert not relaxed_match(sp("LOCATION", 20, 27), g)
    assert not relaxed_match(sp("PERSON", 32, 40), g)  # touching, no overlap
    assert exact_match(sp("PERSON", 20, 32), g)
    assert not exact_match(sp("PERSON", 20, 31), g)


def test_align_one_pred_cannot_cover_two_golds():
    preds = [sp("PERSON", 0, 20)]
    golds = [sp("PERSON", 0, 10), sp("PERSON", 11, 20)]
    pairs = align_relaxed(preds, golds)
    assert len(pairs) == 1  # exactly one TP, not two


def test_align_maximizes_tp_count_over_iou():
    # p0 overlaps both g0 (iou high) and g1 (iou low); p1 only overlaps g0.
    preds = [sp("PERSON", 0, 10), sp("PERSON", 2, 8)]
    golds = [sp("PERSON", 0, 10), sp("PERSON", 9, 30)]
    pairs = dict(align_relaxed(preds, golds))
    # must match 2 pairs (max cardinality), so p0->g1 and p1->g0
    assert len(pairs) == 2
    assert pairs[0] == 1 and pairs[1] == 0


def test_align_tiebreak_prefers_higher_score():
    golds = [sp("PERSON", 0, 10)]
    preds = [sp("PERSON", 0, 10, score=0.4), sp("PERSON", 0, 10, score=0.9)]
    pairs = align_relaxed(preds, golds)
    assert pairs == [(1, 0)]  # higher-score prediction wins the single gold


def test_align_exact_requires_boundaries():
    preds = [sp("PERSON", 20, 27)]
    golds = [sp("PERSON", 20, 32)]
    assert align_relaxed(preds, golds) == [(0, 0)]
    assert align_exact(preds, golds) == []
