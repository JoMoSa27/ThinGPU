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

from .geometry import STRICT, chord_radius

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


def neighbour_graph(xyz, distance_m):
    """Pairs of records closer than distance_m, as CSR arrays (indptr, indices)."""
    n = len(xyz)
    if n < 2:
        return np.zeros(n + 1, np.int64), np.zeros(0, np.int64)
    pairs = cKDTree(xyz).query_pairs(float(chord_radius(distance_m)) * STRICT, output_type="ndarray")
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


def thin(xyz, group, distance_m, rule="mindeg", seed=0):
    """Thin every group (species) independently. Records with group < 0 are not kept.

    Returns (kept mask, rounds), where rounds is the maximum over groups: the
    number of parallel rounds the GPU needs to decide every record.
    """
    kept = np.zeros(len(xyz), dtype=bool)
    rounds = 0
    order = np.argsort(group, kind="stable")
    g = group[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    for a, b in zip(starts, np.r_[starts[1:], len(g)]):
        if g[a] < 0:
            continue
        idx = order[a:b]
        indptr, indices = neighbour_graph(xyz[idx], distance_m)
        k, r = thin_in_rounds(indptr, indices, priorities(indptr, rule, seed, idx))
        kept[idx[k]] = True
        rounds = max(rounds, r)
    return kept, rounds


def target_group_density(xyz, distance_m, workers=-1):
    """For every record, the number of other records closer than distance_m."""
    r = float(chord_radius(distance_m)) * STRICT
    return cKDTree(xyz).query_ball_point(xyz, r, return_length=True, workers=workers) - 1


def check(xyz, group, kept, distance_m):
    """(pairs of kept records closer than d, dropped records that could be added back)."""
    too_close = addable = 0
    for gid in np.unique(group[group >= 0]):
        idx = np.flatnonzero(group == gid)
        indptr, indices = neighbour_graph(xyz[idx], distance_m)
        k = kept[idx]
        src = np.repeat(np.arange(len(idx)), np.diff(indptr))
        too_close += int((k[src] & k[indices]).sum() // 2)
        has_kept = np.bincount(src[k[indices]], minlength=len(idx)) > 0
        addable += int((~k & ~has_kept).sum())
    return too_close, addable
