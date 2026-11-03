"""Synthetic occurrence data for examples and the installation self-test.

Records are clustered around a few "hotspots", as real opportunistic records
are around roads and reserves, so thinning has something to remove.
"""
import numpy as np
import pandas as pd


def synthetic_occurrences(n=20000, n_species=25, seed=1, center=(-84.1, 9.9), spread_deg=1.2):
    rng = np.random.default_rng(seed)
    hotspots = np.asarray(center) + rng.normal(0, spread_deg, size=(12, 2))
    weights = rng.dirichlet(np.ones(len(hotspots)) * 0.4)
    which = rng.choice(len(hotspots), size=n, p=weights)
    pts = hotspots[which] + rng.normal(0, 0.05, size=(n, 2))
    background = rng.random(n) < 0.15
    pts[background] = np.asarray(center) + rng.uniform(-2, 2, size=(background.sum(), 2))
    species = rng.zipf(1.6, size=n) % n_species
    df = pd.DataFrame({
        "species": [f"Species {s:02d}" for s in species],
        "decimalLongitude": pts[:, 0].round(5),
        "decimalLatitude": pts[:, 1].round(5),
    })
    repeats = df.sample(frac=0.2, random_state=seed)      # repeated visits to the same site
    return pd.concat([df, repeats], ignore_index=True)


def main(path="example_occurrences.csv", n=20000):
    df = synthetic_occurrences(n)
    df.to_csv(path, index=False)
    print(f"{len(df):,} synthetic records of {df.species.nunique()} species -> {path}")
