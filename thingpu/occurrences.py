"""Reading and cleaning occurrence tables."""
from pathlib import Path

import numpy as np
import pandas as pd

# Column names tried, in order, when the user does not name them.
LON_COLUMNS = ("decimalLongitude", "longitude", "lon", "x")
LAT_COLUMNS = ("decimalLatitude", "latitude", "lat", "y")
SPECIES_COLUMNS = ("species", "scientificName")


def read_table(path):
    """Read a CSV or tab-separated file (GBIF downloads are tab-separated)."""
    path = Path(path)
    with open(path, encoding="utf-8", errors="replace") as f:
        header = f.readline()
    sep = "\t" if header.count("\t") > header.count(",") else ","
    return pd.read_csv(path, sep=sep, low_memory=False, on_bad_lines="warn",
                       quoting=3 if sep == "\t" else 0)


def pick_column(df, requested, candidates, what):
    """Return the column to use for `what`, or raise a clear error."""
    if requested:
        if requested not in df.columns:
            raise SystemExit(f"column '{requested}' ({what}) not found; available: {', '.join(df.columns[:20])}")
        return requested
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"could not find a {what} column; name it with --{what}-col")


def prepare(df, lon_col, lat_col, species_col=None, max_uncertainty_m=None, collapse_repeats=True,
            decimals=5):
    """Validate coordinates and flag records that cannot be used.

    Returns a DataFrame with the original rows plus:
      thingpu_valid    coordinates are usable (and within the uncertainty limit)
      thingpu_group    integer species id, -1 when the species is missing
      thingpu_repeat   same species at the same coordinates (rounded to `decimals`)
                       as an earlier record; such repeats add no spatial information
                       and are left out of thinning when collapse_repeats is True
    """
    out = df.copy()
    lon = pd.to_numeric(out[lon_col], errors="coerce")
    lat = pd.to_numeric(out[lat_col], errors="coerce")
    valid = lon.between(-180, 180) & lat.between(-90, 90) & ~((lon == 0) & (lat == 0))
    if max_uncertainty_m is not None and "coordinateUncertaintyInMeters" in out.columns:
        unc = pd.to_numeric(out["coordinateUncertaintyInMeters"], errors="coerce")
        valid &= unc.isna() | (unc <= max_uncertainty_m)

    if species_col:
        names = out[species_col].astype("string").str.strip()
        names = names.where(names.notna() & (names != ""))
        codes, _ = pd.factorize(names, use_na_sentinel=True)
    else:
        codes = np.zeros(len(out), dtype=np.int64)
    group = np.where(valid, codes, -1).astype(np.int32)

    repeat = np.zeros(len(out), dtype=bool)
    if collapse_repeats:
        key = pd.DataFrame({"g": group, "x": lon.round(decimals), "y": lat.round(decimals)})
        repeat = key.duplicated(keep="first").to_numpy() & (group >= 0)

    out["thingpu_valid"] = valid.to_numpy()
    out["thingpu_group"] = group
    out["thingpu_repeat"] = repeat
    out["_lon"], out["_lat"] = lon.to_numpy(), lat.to_numpy()
    return out
