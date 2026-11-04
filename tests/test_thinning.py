import numpy as np

from thingpu import thinning
from thingpu.example import synthetic_occurrences
from thingpu.geometry import chord_radius, to_xyz
from thingpu.occurrences import prepare


def _data(n=3000):
    df = prepare(synthetic_occurrences(n), "decimalLongitude", "decimalLatitude", "species")
    use = (df.thingpu_valid & ~df.thingpu_repeat).to_numpy()
    return to_xyz(df._lon[use], df._lat[use]), df.thingpu_group.to_numpy()[use]


def test_chord_matches_great_circle():
    a, b = to_xyz([0.0, 0.0], [0.0, 1.0])
    arc = np.radians(1.0) * 6_371_008.8
    assert abs(np.linalg.norm(a - b) - chord_radius(arc)) < 1e-6


def test_thinning_is_exact_and_maximal():
    xyz, group = _data()
    for rule in ("mindeg", "random"):
        kept, _ = thinning.thin(xyz, group, 5000, rule, 0)
        assert thinning.check(xyz, group, kept, 5000) == (0, 0)


def test_float32_reference_is_exact_in_its_precision():
    xyz, group = _data()
    kept, _ = thinning.thin(xyz, group, 5000, precision="f32")
    assert thinning.check(xyz, group, kept, 5000, "f32") == (0, 0)
    lo, hi = thinning.density_bounds(xyz, 5000)
    c32 = thinning.target_group_density(xyz, 5000, precision="f32")
    assert ((lo <= c32) & (c32 <= hi)).all()


def test_rounds_equal_sequential_greedy():
    xyz, group = _data()
    idx = np.flatnonzero(group == group[0])
    indptr, indices = thinning.neighbour_graph(xyz[idx], 5000)
    prio = thinning.priorities(indptr, "mindeg", 0, idx)
    kept_rounds, _ = thinning.thin_in_rounds(indptr, indices, prio)
    kept_seq = np.zeros(len(idx), bool)
    dropped = np.zeros(len(idx), bool)
    for v in np.argsort(-prio, kind="stable"):
        if not dropped[v]:
            kept_seq[v] = True
            dropped[indices[indptr[v]:indptr[v + 1]]] = True
            dropped[v] = True
    assert np.array_equal(kept_rounds, kept_seq)


def test_species_are_thinned_independently():
    xyz = to_xyz([0.0, 0.0001], [0.0, 0.0])
    kept, _ = thinning.thin(xyz, np.array([0, 1], np.int32), 1000)
    assert kept.all()
