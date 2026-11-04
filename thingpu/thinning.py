"""Exact spatial thinning and target-group density on the CPU.

Thinning a species at distance d keeps a maximal independent set of the graph
that joins records closer than d. Every record gets a fixed priority and is
kept when no kept neighbour has a higher one. Processing in rounds, as the GPU
does, returns the same set as the sequential greedy rule (Blelloch, Fineman and
Shun, SPAA 2012). The priorities match the CUDA code bit for bit, so CPU and
GPU results can be compared record by record.
"""
import numpy as np
from scipy.spatial import cKDTree

from .geometry import F32_MARGIN, query_frame

_GOLD = np.uint64(0x9E3779B97F4A7C15)


def splitmix64(x):
    """splitmix64 hash (Steele, Lea and Flood, 2014)."""
    with np.errstate(over="ignore"):
        z = np.asarray(x, dtype=np.uint64) + _GOLD
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def record_hash(ids, seed):
    return splitmix64(np.asarray(ids, dtype=np.uint64) ^ splitmix64(np.uint64(seed)))


def neighbour_graph(xyz, distance_m, precision="f64"):
    """Pairs of records closer than distance_m, as CSR arrays (indptr, indices)."""
    pts, r = query_frame(xyz, distance_m, precision)
    return _graph(pts, r)


def _graph(pts, r):
    n = len(pts)
    if n < 2:
        return np.zeros(n + 1, np.int64), np.zeros(0, np.int64)
    pairs = cKDTree(pts).query_pairs(r, output_type="ndarray")
    src = np.concatenate([pairs[:, 0], pairs[:, 1]])
    dst = np.concatenate([pairs[:, 1], pairs[:, 0]])
    order = np.argsort(src, kind="stable")
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(src, minlength=n), out=indptr[1:])
    return indptr, dst[order].astype(np.int64)


def priorities(indptr, rule, seed, ids):
    """Rank of each record, higher first.

    random:              hash(id) first, then id
    fewest-neighbours:   fewer neighbours first, then hash(id), then id
    `ids` are the global row numbers, so the order does not depend on how the
    records are grouped.
    """
    ids = np.asarray(ids, dtype=np.uint64)
    h = record_hash(ids, seed)
    if rule == "random":
        order = np.lexsort((ids, h))[::-1]
    elif rule == "mindeg":
        order = np.lexsort((ids, h, -np.diff(indptr)))[::-1]
    else:
        raise ValueError(f"unknown rule: {rule}")
    prio = np.empty(len(ids), np.int64)
    prio[order] = np.arange(len(ids), 0, -1, dtype=np.int64)
    return prio


def thin_in_rounds(indptr, indices, prio):
    """Parallel-rounds greedy maximal independent set. Returns (kept, rounds)."""
    n = len(prio)
    src = np.repeat(np.arange(n), np.diff(indptr))
    state = np.zeros(n, np.int8)          # 0 undecided, 1 kept, 2 dropped
    rounds = 0
    while (state == 0).any():
        rounds += 1
        undecided = state == 0
        blocking = undecided[src] & undecided[indices] & (prio[indices] > prio[src])
        blocked = np.bincount(src[blocking], minlength=n) > 0
        state[undecided & ~blocked] = 1
        hit = (state[indices] == 1) & (state[src] == 0)
        state[np.unique(src[hit])] = 2
    return state == 1, rounds


def thin(xyz, group, distance_m, rule="mindeg", seed=0, precision="f64"):
    """Thin every group (species) independently. Records with group < 0 are not kept.

    Returns (kept mask, rounds), where rounds is the maximum over groups: the
    number of parallel rounds the GPU needs to decide every record. With
    precision="f32" the result is the one the GPU must return.
    """
    pts, r = query_frame(xyz, distance_m, precision)
    kept = np.zeros(len(xyz), dtype=bool)
    rounds = 0
    order = np.argsort(group, kind="stable")
    g = group[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(g)]):
        if g[a] < 0:
            continue
        idx = order[a:b]
        indptr, indices = _graph(pts[idx], r)
        k, n_rounds = thin_in_rounds(indptr, indices, priorities(indptr, rule, seed, idx))
        kept[idx[k]] = True
        rounds = max(rounds, n_rounds)
    return kept, rounds


def target_group_density(xyz, distance_m, workers=-1, precision="f64"):
    """For every record, the number of other records closer than distance_m."""
    pts, r = query_frame(xyz, distance_m, precision)
    return cKDTree(pts).query_ball_point(pts, r, return_length=True, workers=workers) - 1


def density_bounds(xyz, distance_m, workers=-1):
    """Lowest and highest counts a float32 (GPU) computation can give: pairs within
    F32_MARGIN of the radius may be counted or not."""
    pts, r = query_frame(xyz, distance_m, "f32")
    tree = cKDTree(pts)
    lo, hi = (tree.query_ball_point(pts, r * f, return_length=True, workers=workers) - 1
              for f in (1 - F32_MARGIN, 1 + F32_MARGIN))
    return lo, hi


def check(xyz, group, kept, distance_m, precision="f64"):
    """(pairs of kept records closer than d, dropped records that could be added back).

    With precision="f32" (a GPU result), pairs within F32_MARGIN of the radius are
    not counted against the result on either side.
    """
    pts, r = query_frame(xyz, distance_m, precision)
    margin = F32_MARGIN if precision == "f32" else 0.0
    too_close = addable = 0
    for gid in np.unique(group[group >= 0]):
        idx = np.flatnonzero(group == gid)
        k = kept[idx]
        indptr, indices = _graph(pts[idx], r * (1 - margin))
        src = np.repeat(np.arange(len(idx)), np.diff(indptr))
        too_close += int((k[src] & k[indices]).sum() // 2)
        if margin:
            indptr, indices = _graph(pts[idx], r * (1 + margin))
            src = np.repeat(np.arange(len(idx)), np.diff(indptr))
        has_kept = np.bincount(src[k[indices]], minlength=len(idx)) > 0
        addable += int((~k & ~has_kept).sum())
    return too_close, addable
