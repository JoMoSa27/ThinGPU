"""Command line interface: `thingpu thin` and `thingpu effort`."""
import argparse
import sys
import time

import numpy as np

from . import __version__, gpu, thinning
from .geometry import to_xyz
from .occurrences import (LAT_COLUMNS, LON_COLUMNS, SPECIES_COLUMNS, pick_column, prepare,
                          read_table)


def _backend(name):
    binary = gpu.find_binary()
    if name == "gpu" and binary is None:
        sys.exit("--backend gpu: thingpu-gpu not found (build it, or set THINGPU_GPU)")
    use_gpu = name == "gpu" or (name == "auto" and binary is not None)
    return use_gpu, binary


def _load(args, species_required):
    df = read_table(args.input)
    lon = pick_column(df, args.lon_col, LON_COLUMNS, "lon")
    lat = pick_column(df, args.lat_col, LAT_COLUMNS, "lat")
    species = None
    if species_required and not args.no_species:
        species = pick_column(df, args.species_col, SPECIES_COLUMNS, "species")
    return prepare(df, lon, lat, species, max_uncertainty_m=args.max_uncertainty,
                   collapse_repeats=species_required)


def _write(df, path, drop=()):
    sep = "\t" if str(path).endswith((".tsv", ".txt")) else ","
    df.drop(columns=["_lon", "_lat", *drop]).to_csv(path, sep=sep, index=False)


def cmd_thin(args):
    df = _load(args, species_required=True)
    use = (df.thingpu_valid & ~df.thingpu_repeat & (df.thingpu_group >= 0)).to_numpy()
    xyz = to_xyz(df._lon.to_numpy()[use], df._lat.to_numpy()[use])
    group = df.thingpu_group.to_numpy()[use]
    use_gpu, binary = _backend(args.backend)
    print(f"{len(df):,} records, {use.sum():,} unique and valid, {len(np.unique(group)):,} species; "
          f"backend: {'GPU' if use_gpu else 'CPU'}", file=sys.stderr)
    for km in args.distance:
        t = time.perf_counter()
        if use_gpu:
            kept, rounds, _ = gpu.thin(xyz, group, km * 1000, args.rule, args.seed, args.index, binary)
        else:
            kept, rounds = thinning.thin(xyz, group, km * 1000, args.rule, args.seed)
        col = np.zeros(len(df), dtype=bool)
        col[np.flatnonzero(use)[kept]] = True
        df[f"kept_{km:g}km"] = col
        print(f"  {km:g} km: kept {kept.sum():,} of {use.sum():,} in {rounds} rounds "
              f"({time.perf_counter() - t:.2f} s)", file=sys.stderr)
        if args.check:
            close, addable = thinning.check(xyz, group, kept, km * 1000)
            print(f"    check: {close} pairs closer than {km:g} km, {addable} records could be added",
                  file=sys.stderr)
    if args.only_kept:
        if len(args.distance) != 1:
            sys.exit("--only-kept needs a single --distance")
        df = df[df[f"kept_{args.distance[0]:g}km"]]
    _write(df, args.out)
    print(f"-> {args.out}", file=sys.stderr)


def cmd_effort(args):
    df = _load(args, species_required=False)
    use = df.thingpu_valid.to_numpy()
    xyz = to_xyz(df._lon.to_numpy()[use], df._lat.to_numpy()[use])
    use_gpu, binary = _backend(args.backend)
    print(f"{len(df):,} records, {use.sum():,} with valid coordinates; "
          f"backend: {'GPU' if use_gpu else 'CPU'}", file=sys.stderr)
    for km in args.distance:
        t = time.perf_counter()
        if use_gpu:
            counts, _ = gpu.target_group_density(xyz, km * 1000, args.index, binary)
        else:
            counts = thinning.target_group_density(xyz, km * 1000)
        col = np.full(len(df), -1, dtype=np.int64)
        col[use] = counts
        df[f"neighbours_{km:g}km"] = col
        print(f"  {km:g} km: median {np.median(counts):g} neighbours ({time.perf_counter() - t:.2f} s)",
              file=sys.stderr)
    _write(df, args.out, drop=("thingpu_group", "thingpu_repeat"))
    print(f"-> {args.out}", file=sys.stderr)


def cmd_example(args):
    from .example import main as make
    make(args.out, args.n)


def cmd_selftest(args):
    """Check the installation: exact CPU thinning, and GPU = CPU when a GPU is available."""
    from .example import synthetic_occurrences
    df = prepare(synthetic_occurrences(args.n), "decimalLongitude", "decimalLatitude", "species")
    use = (df.thingpu_valid & ~df.thingpu_repeat).to_numpy()
    xyz = to_xyz(df._lon.to_numpy()[use], df._lat.to_numpy()[use])
    group = df.thingpu_group.to_numpy()[use]
    ok = True
    binary = gpu.find_binary()
    print(f"thingpu {__version__}; synthetic data: {use.sum():,} unique records of {len(np.unique(group))} species")
    print(f"GPU program: {binary or 'not found (CPU only)'}")
    for km in (1.0, 5.0, 10.0):
        kept, rounds = thinning.thin(xyz, group, km * 1000, "mindeg", 0)
        close, addable = thinning.check(xyz, group, kept, km * 1000)
        line = f"  {km:g} km  CPU: kept {kept.sum():,} in {rounds} rounds, {close} close pairs, {addable} addable"
        ok &= close == 0 and addable == 0
        if binary:
            for index in ("kdtree", "lbvh", "grid"):
                k_gpu, r_gpu, _ = gpu.thin(xyz, group, km * 1000, "mindeg", 0, index, binary)
                diff = int((k_gpu != kept).sum())
                line += f" | GPU {index}: {diff} differences, {r_gpu} rounds"
                ok &= diff == 0 and r_gpu == rounds
            c_cpu = thinning.target_group_density(xyz, km * 1000)
            c_gpu, _ = gpu.target_group_density(xyz, km * 1000, "kdtree", binary)
            line += f" | effort: {int((c_cpu != c_gpu).sum())} differences"
        print(line)
    print("OK" if ok else "FAILED: please report this with the output above")
    sys.exit(0 if ok else 1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="thingpu", description="Exact spatial thinning and sampling-effort "
                                "density for species occurrence records, on GPU or CPU.")
    p.add_argument("--version", action="version", version=f"thingpu {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(s):
        s.add_argument("input", help="occurrence table (CSV, or a GBIF tab-separated download)")
        s.add_argument("--distance", "-d", type=float, nargs="+", required=True, help="distance(s) in km")
        s.add_argument("--out", "-o", required=True, help="output table (.csv or .tsv)")
        s.add_argument("--lon-col"), s.add_argument("--lat-col")
        s.add_argument("--max-uncertainty", type=float, default=None,
                       help="drop records with coordinateUncertaintyInMeters above this value (m)")
        s.add_argument("--backend", choices=["auto", "gpu", "cpu"], default="auto")
        s.add_argument("--index", choices=["kdtree", "lbvh", "grid"], default="kdtree",
                       help="GPU spatial index (default kdtree, the most robust to clustering)")

    t = sub.add_parser("thin", help="per-species spatial thinning")
    common(t)
    t.add_argument("--species-col")
    t.add_argument("--no-species", action="store_true", help="treat all records as one species")
    t.add_argument("--rule", choices=["mindeg", "random"], default="mindeg",
                   help="priority order: fewest neighbours first (keeps more records) or random")
    t.add_argument("--seed", type=int, default=0, help="breaks ties between equal priorities")
    t.add_argument("--only-kept", action="store_true", help="write only the kept records")
    t.add_argument("--check", action="store_true", help="verify the result on the CPU")
    t.set_defaults(func=cmd_thin)

    e = sub.add_parser("effort", help="target-group density: records of all species within d of each record")
    common(e)
    e.set_defaults(func=cmd_effort)

    x = sub.add_parser("example", help="write a synthetic occurrence table to try the tool")
    x.add_argument("--out", "-o", default="example_occurrences.csv")
    x.add_argument("--n", type=int, default=20000)
    x.set_defaults(func=cmd_example)

    st = sub.add_parser("selftest", help="check that thinning is exact and that GPU and CPU agree")
    st.add_argument("--n", type=int, default=20000)
    st.set_defaults(func=cmd_selftest)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
