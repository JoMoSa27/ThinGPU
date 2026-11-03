"""Run the CUDA program (thingpu-gpu) on arrays prepared in Python."""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def find_binary():
    """Path to thingpu-gpu, or None. Looks at $THINGPU_GPU, the PATH and build/."""
    for candidate in (os.environ.get("THINGPU_GPU"), shutil.which("thingpu-gpu"),
                      str(ROOT / "build" / "thingpu-gpu")):
        if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
            return candidate
    return None


def _centered_f32(xyz):
    # The GPU works in single precision. Subtracting the centroid first keeps
    # about 1 cm of resolution for regional data instead of 0.5 m at Earth scale.
    return np.ascontiguousarray((xyz - xyz.mean(axis=0)).astype(np.float32))


def _run(binary, args, workdir):
    res = subprocess.run([binary, *args], cwd=workdir, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"thingpu-gpu failed:\n{res.stderr.strip()}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def thin(xyz, group, distance_m, rule="mindeg", seed=0, index="kdtree", binary=None):
    """Same contract as thinning.thin, on the GPU. Returns (kept, rounds, info)."""
    binary = binary or find_binary()
    with tempfile.TemporaryDirectory(prefix="thingpu_") as tmp:
        tmp = Path(tmp)
        _centered_f32(xyz).tofile(tmp / "xyz.f32")
        np.ascontiguousarray(group, dtype=np.int32).tofile(tmp / "group.i32")
        info = _run(binary, ["thin", "--n", str(len(xyz)), "--xyz", "xyz.f32", "--group", "group.i32",
                             "--distance-m", repr(float(distance_m)), "--rule", rule, "--seed", str(seed),
                             "--index", index, "--out", "kept.u8"], tmp)
        kept = np.fromfile(tmp / "kept.u8", dtype=np.uint8).astype(bool)
    return kept, info["rounds"], info


def target_group_density(xyz, distance_m, index="kdtree", binary=None):
    """Same contract as thinning.target_group_density, on the GPU. Returns (counts, info)."""
    binary = binary or find_binary()
    with tempfile.TemporaryDirectory(prefix="thingpu_") as tmp:
        tmp = Path(tmp)
        _centered_f32(xyz).tofile(tmp / "xyz.f32")
        info = _run(binary, ["effort", "--n", str(len(xyz)), "--xyz", "xyz.f32",
                             "--distance-m", repr(float(distance_m)), "--index", index,
                             "--out", "count.i32"], tmp)
        counts = np.fromfile(tmp / "count.i32", dtype=np.int32)
    return counts, info
