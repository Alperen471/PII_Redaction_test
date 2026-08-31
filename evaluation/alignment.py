"""Prediction <-> ground-truth alignment (plan v5 §10).

Locked rule:
    1:1 optimal bipartite matching.
      1. build relaxed-match edges
      2. maximum-cardinality matching
      3. tie on cardinality  -> maximize total IoU
      4. still tied          -> higher confidence score
      5. still tied          -> smaller prediction start (deterministic)
    One prediction can match at most one GT; one GT at most one prediction.

``relaxed_match`` requires equal labels, so the matching decomposes per label
automatically (plan v5 §10: "her sample ve label için").

Implementation: self-contained integer min-cost max-flow (successive shortest
augmenting paths). scipy is intentionally not a dependency. Per-record
cardinalities are tiny, so the O(V*E) SPFA augmentation is negligible.
"""
from __future__ import annotations

from typing import Sequence

from evaluation.spans import Span, exact_match, iou, relaxed_match

_IOU_SCALE = 10 ** 6
_IOU_WEIGHT = 10 ** 12          # IoU term strictly dominates the tie-break term
_POS_CAP = 10 ** 6


def _tiebreak_key(pred: Span) -> int:
    """Non-negative int < ``_IOU_WEIGHT``: higher score first, then smaller start/end."""
    score_milli = int(round(max(0.0, min(1.0, float(pred.get("score", 1.0)))) * 1000))
    start = min(max(int(pred.get("start", 0)), 0), _POS_CAP - 1)
    end = min(max(int(pred.get("end", 0)), 0), _POS_CAP - 1)
    return (
        score_milli * (_POS_CAP * _POS_CAP)
        + (_POS_CAP - 1 - start) * _POS_CAP
        + (_POS_CAP - 1 - end)
    )


def _edge_weight(pred: Span, gold: Span) -> int:
    iou_i = int(round(iou(pred, gold) * _IOU_SCALE))
    return iou_i * _IOU_WEIGHT + _tiebreak_key(pred)


def align(
    preds: Sequence[Span],
    golds: Sequence[Span],
    *,
    matcher=relaxed_match,
) -> list[tuple[int, int]]:
    """Return ``(pred_index, gold_index)`` matched pairs, ``pred_index`` ascending."""
    n, m = len(preds), len(golds)
    if n == 0 or m == 0:
        return []

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
            break
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
            if 1 + n <= v <= n + m and cap[eid] == 0:
                pairs.append((i, v - 1 - n))
    pairs.sort()
    return pairs


def align_relaxed(preds: Sequence[Span], golds: Sequence[Span]) -> list[tuple[int, int]]:
    return align(preds, golds, matcher=relaxed_match)


def align_exact(preds: Sequence[Span], golds: Sequence[Span]) -> list[tuple[int, int]]:
    return align(preds, golds, matcher=exact_match)
