<<<<<<< HEAD
"""Span geometry (plan v5 §8, §9, §14).

Span convention (locked, plan v5 §14):
    * unit    : Python ``str`` Unicode code-point index
    * interval: ``[start, end)`` -- start inclusive, end exclusive
    * anchor  : the original dataset text, unmodified (no lowercase / casefold /
                whitespace collapse / NFC / NFD before evaluation)

Primary ("relaxed") match (plan v5 §8):
    ``pred.label == gold.label AND intersection_length > 0``   (IoU >= 0.5 is NOT used)

The prediction/gold aligner lives in :mod:`evaluation.alignment`.
"""
from __future__ import annotations

from typing import Iterable, TypedDict
=======
"""Span geometry + prediction/gold alignment (plan v4 §9, §10.6-10.8).

Span convention (locked, plan v4 §9.1):
    * unit    : Python ``str`` Unicode code-point index
    * interval: ``[start, end)`` -- start inclusive, end exclusive
    * anchor  : the original dataset text, unmodified

Primary ("relaxed") match, locked in the implementation contract:
    ``pred.label == gold.label AND intersection_length > 0``

Alignment (locked): optimal 1:1 bipartite matching that
    1. maximizes the number of TP edges,
    2. then maximizes total IoU,
    3. then breaks ties deterministically by (score desc, start asc, end asc).
A single prediction can therefore never be matched to two gold entities.
"""
from __future__ import annotations

from typing import Iterable, Sequence, TypedDict
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53


class Span(TypedDict, total=False):
    text: str
    label: str
    start: int
    end: int
    score: float


<<<<<<< HEAD
=======
# --------------------------------------------------------------------------- #
# Span geometry
# --------------------------------------------------------------------------- #
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53
def intersection_length(a: Span, b: Span) -> int:
    lo = max(a["start"], b["start"])
    hi = min(a["end"], b["end"])
    return max(0, hi - lo)


def union_length(a: Span, b: Span) -> int:
    len_a = a["end"] - a["start"]
    len_b = b["end"] - b["start"]
    return len_a + len_b - intersection_length(a, b)


def iou(a: Span, b: Span) -> float:
    u = union_length(a, b)
    if u <= 0:
        return 0.0
    return intersection_length(a, b) / u


def relaxed_match(pred: Span, gold: Span) -> bool:
    """Primary match: same label + at least one overlapping code point."""
    return pred["label"] == gold["label"] and intersection_length(pred, gold) > 0


def exact_match(pred: Span, gold: Span) -> bool:
    return (
        pred["label"] == gold["label"]
        and pred["start"] == gold["start"]
        and pred["end"] == gold["end"]
    )


def is_partial(pred: Span, gold: Span) -> bool:
<<<<<<< HEAD
    """Overlapping, same label, but not an exact boundary match (plan v5 §9)."""
    return relaxed_match(pred, gold) and not exact_match(pred, gold)


def overlaps_intervals(start: int, end: int, intervals: list[tuple[int, int]]) -> bool:
    """True if ``[start, end)`` shares at least one code point with any interval."""
    return any(not (end <= lo or start >= hi) for lo, hi in intervals)
=======
    """Overlapping, same label, but not an exact boundary match (plan v4 §10.7)."""
    return relaxed_match(pred, gold) and not exact_match(pred, gold)


# --------------------------------------------------------------------------- #
# Min-cost max-cardinality bipartite matching (SSP / SPFA)
# --------------------------------------------------------------------------- #
# scipy is unavailable in the target runtime, so a small self-contained
# min-cost-max-flow is used. Per-record cardinalities are tiny (< ~20), so the
# O(V*E) SPFA augmentation is comfortably fast.

_IOU_SCALE = 10 ** 6          # IoU resolution
_IOU_WEIGHT = 10 ** 12        # keeps IoU strictly above the tie-break term
_START_CAP = 10 ** 6          # clamp for the start-position tie-break


def _tiebreak_key(pred: Span) -> int:
    """Deterministic preference: higher score first, then smaller start/end.

    Encoded as a non-negative integer strictly below ``_IOU_WEIGHT`` so it can
    only decide between solutions that are already tied on (count, total IoU).
    """
    score_milli = int(round(max(0.0, min(1.0, float(pred.get("score", 1.0)))) * 1000))
    start = min(max(int(pred.get("start", 0)), 0), _START_CAP - 1)
    end = min(max(int(pred.get("end", 0)), 0), _START_CAP - 1)
    # score_milli in [0,1000]; start/end in [0,1e6).
    return score_milli * (_START_CAP * _START_CAP) \
        + (_START_CAP - 1 - start) * _START_CAP \
        + (_START_CAP - 1 - end)


def _edge_weight(pred: Span, gold: Span) -> int:
    iou_i = int(round(iou(pred, gold) * _IOU_SCALE))
    return iou_i * _IOU_WEIGHT + _tiebreak_key(pred)


def align(
    preds: Sequence[Span],
    golds: Sequence[Span],
    *,
    matcher=relaxed_match,
) -> list[tuple[int, int]]:
    """Return a list of ``(pred_index, gold_index)`` matched pairs.

    ``matcher`` decides which pairs are eligible edges (``relaxed_match`` for the
    primary metric, ``exact_match`` for the exact-span metric).
    """
    n, m = len(preds), len(golds)
    if n == 0 or m == 0:
        return []

    # Node ids: 0 = source, 1..n = preds, n+1..n+m = golds, n+m+1 = sink.
    S, T = 0, n + m + 1
    V = n + m + 2
    graph: list[list[int]] = [[] for _ in range(V)]
    to: list[int] = []
    cap: list[int] = []
    cost: list[int] = []

    def add_edge(u: int, v: int, c: int, w: int) -> None:
        graph[u].append(len(to)); to.append(v); cap.append(c); cost.append(w)
        graph[v].append(len(to)); to.append(u); cap.append(0); cost.append(-w)

    for i in range(n):
        add_edge(S, 1 + i, 1, 0)
    for j in range(m):
        add_edge(1 + n + j, T, 1, 0)
    for i, p in enumerate(preds):
        for j, g in enumerate(golds):
            if matcher(p, g):
                # maximize weight  ->  minimize negative weight
                add_edge(1 + i, 1 + n + j, 1, -_edge_weight(p, g))

    INF = float("inf")
    while True:
        dist = [INF] * V
        in_queue = [False] * V
        prev_edge = [-1] * V
        dist[S] = 0
        queue = [S]
        in_queue[S] = True
        while queue:
            u = queue.pop(0)
            in_queue[u] = False
            for eid in graph[u]:
                if cap[eid] <= 0:
                    continue
                v = to[eid]
                nd = dist[u] + cost[eid]
                if nd < dist[v]:
                    dist[v] = nd
                    prev_edge[v] = eid
                    if not in_queue[v]:
                        queue.append(v)
                        in_queue[v] = True
        if prev_edge[T] == -1:
            break  # no more augmenting paths -> max cardinality reached
        # augment one unit along the shortest (min-cost) path
        v = T
        while v != S:
            eid = prev_edge[v]
            cap[eid] -= 1
            cap[eid ^ 1] += 1
            v = to[eid ^ 1]

    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for eid in graph[1 + i]:
            v = to[eid]
            if 1 + n <= v <= n + m and cap[eid] == 0 and cost[eid] <= 0:
                # forward pred->gold edge that carries flow
                pairs.append((i, v - 1 - n))
    pairs.sort()
    return pairs


def align_relaxed(preds: Sequence[Span], golds: Sequence[Span]) -> list[tuple[int, int]]:
    return align(preds, golds, matcher=relaxed_match)


def align_exact(preds: Sequence[Span], golds: Sequence[Span]) -> list[tuple[int, int]]:
    return align(preds, golds, matcher=exact_match)
>>>>>>> e59785055f5985db95f4735d0c15d37f3a6cdd53


def covered_intervals(spans: Iterable[Span]) -> list[tuple[int, int]]:
    """Merge spans into sorted, disjoint ``[start, end)`` intervals."""
    ordered = sorted(((s["start"], s["end"]) for s in spans), key=lambda t: t[0])
    merged: list[tuple[int, int]] = []
    for start, end in ordered:
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged
