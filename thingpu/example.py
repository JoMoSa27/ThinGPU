"""Synthetic occurrence data for examples and the installation self-test.

Records are clustered around a few "hotspots", as real opportunistic records
are around roads and reserves, so thinning has something to remove. Only
uniform and normal draws are used, so the data are the same across numpy versions.
"""
import numpy as np
import pandas as pd


def synthetic_occurrences(n=20000, n_species=25, seed=1, center=(-84.1, 9.9), spread_deg=1.2):
    rng = np.random.default_rng(seed)
    hotspots = np.asarray(center) + spread_deg * rng.standard_normal((12, 2))
    weights = rng.random(len(hotspots)) ** 3
    which = np.searchsorted(np.cumsum(weights / weights.sum()), rng.random(n), side="right")
    pts = hotspots[np.minimum(which, len(hotspots) - 1)] + 0.05 * rng.standard_normal((n, 2))
    background = rng.random(n) < 0.15
    pts[background] = np.asarray(center) + rng.uniform(-2, 2, size=(background.sum(), 2))
    species = np.floor(n_species * rng.random(n) ** 3).astype(int)     # a few common species, many rare
    df = pd.DataFrame({
        "species": [f"Species {s:02d}" for s in species],
        "decimalLongitude": pts[:, 0].round(5),
        "decimalLatitude": pts[:, 1].round(5),
    })
    repeats = df.iloc[np.flatnonzero(rng.random(n) < 0.2)]     # repeated visits to the same site
    return pd.concat([df, repeats], ignore_index=True)


def main(path="example_occurrences.csv", n=20000):
    df = synthetic_occurrences(n)
    df.to_csv(path, index=False)
    print(f"{len(df):,} synthetic records of {df.species.nunique()} species -> {path}")
