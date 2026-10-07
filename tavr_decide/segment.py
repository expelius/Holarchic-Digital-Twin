"""Automatic segmentation of the landing zone with TotalSegmentator, and automatic nadirs.

Two modes, decided by whether a TotalSegmentator licence is configured:

* **free** (task ``total``): the aorta only. The LVOT is not separated from the rest of the
  heart in this task, so the landing zone below the annulus is missing and the user must
  complete the segmentation (or use the licensed mode).
* **licensed** (free academic licence, tasks ``heartchambers_highres``, ``aortic_sinuses``,
  ``aorta_annulus``): aorta, left ventricle, LVOT, the three cusps, annulus and sinotubular
  junction. The lumen is their union, and the three nadirs are placed automatically: each
  is the lowest point of its cusp along the annulus normal (pointing to the aorta).

Nothing here is validated against expert landmarks yet; the automatic nadirs are a starting
point for the operator to review, not a replacement.
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .geometry import Volume, fit_plane, load_volume

FREE_TASKS = {"total": ["aorta"]}
LICENSED_TASKS = {"heartchambers_highres": ["aorta", "heart_ventricle_left"],
                  "aortic_sinuses": ["left_ventricular_outflow_tract", "right_coronary_cusp",
                                     "left_coronary_cusp", "non_coronary_cusp"],
                  "aorta_annulus": ["annulus_proper", "sinotubular_junction"]}
LUMEN_CLASSES = {"aorta", "heart_ventricle_left", "left_ventricular_outflow_tract",
                 "right_coronary_cusp", "left_coronary_cusp", "non_coronary_cusp"}
CUSP_LABELS = {"non_coronary_cusp": "nadir_N", "left_coronary_cusp": "nadir_L", "right_coronary_cusp": "nadir_R"}


def find_totalsegmentator() -> str | None:
    """Locate the TotalSegmentator executable: PATH, this interpreter's Scripts folder, or a
    user-level Python install on Windows."""
    exe = shutil.which("TotalSegmentator")
    if exe:
        return exe
    import sys
    cands = [os.path.join(os.path.dirname(sys.executable), "Scripts", "TotalSegmentator.exe"),
             os.path.join(os.path.dirname(sys.executable), "TotalSegmentator")]
    cands += glob.glob(os.path.expandvars(r"%LOCALAPPDATA%\Programs\Python\Python3*\Scripts\TotalSegmentator.exe"))
    return next((c for c in cands if os.path.isfile(c)), None)


def has_licence(exe: str | None = None) -> bool:
    cfg = Path.home() / ".totalsegmentator" / "config.json"
    if not cfg.exists():
        return False
    import json
    try:
        return bool(json.loads(cfg.read_text(encoding="utf-8")).get("license_number"))
    except (OSError, ValueError):
        return False


def set_licence(number: str, exe: str) -> None:
    ext = ".exe" if exe.lower().endswith(".exe") else ""
    tool = os.path.join(os.path.dirname(exe), "totalseg_set_license" + ext)
    subprocess.run([tool, "-l", number], check=True, capture_output=True, env=clean_env())


def clean_env() -> dict:
    """Environment for an external Python tool. Launched from inside 3D Slicer, the process
    inherits PYTHONHOME/PYTHONPATH pointing at Slicer's interpreter, and an executable that
    belongs to another Python install then fails to import its own package."""
    env = dict(os.environ)
    for k in ("PYTHONHOME", "PYTHONPATH", "PYTHONNOUSERSITE", "PYTHONSTARTUP", "PYTHONEXECUTABLE"):
        env.pop(k, None)
    return env


def run_tasks(ct_path: str | Path, out_dir: str | Path, exe: str, licensed: bool, device: str = "cpu",
              log=None) -> dict[str, Path]:
    """Run the TotalSegmentator tasks of the chosen mode; return {class name: mask path}."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tasks = LICENSED_TASKS if licensed else FREE_TASKS
    found: dict[str, Path] = {}
    for task, classes in tasks.items():
        d = out / task
        cmd = [exe, "-i", str(ct_path), "-o", str(d), "-ta", task, "-d", device]
        if task == "total":
            cmd += ["--roi_subset", *classes]
        if log:
            log(f"TotalSegmentator task '{task}'")
        r = subprocess.run(cmd, capture_output=True, text=True, errors="replace", env=clean_env())
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip().splitlines()[-6:]
            raise RuntimeError(f"TotalSegmentator task '{task}' failed (exit {r.returncode}): " + " | ".join(tail))
        for c in classes:
            p = d / f"{c}.nii.gz"
            if p.exists():
                found[c] = p
    return found


def lumen_from_masks(masks: dict[str, Path]) -> Volume:
    vols = [load_volume(p) for c, p in masks.items() if c in LUMEN_CLASSES]
    if not vols:
        raise ValueError("no lumen classes were segmented")
    data = np.zeros_like(vols[0].data, dtype=np.float32)
    for v in vols:
        data = np.maximum(data, (v.data > 0).astype(np.float32))
    return Volume(data, vols[0].affine)


def auto_nadirs(cusps: dict[str, Volume], toward_aorta: np.ndarray | None = None,
                annulus: Volume | None = None) -> dict[str, tuple[float, float, float]]:
    """Lowest point of each cusp along the root axis.

    The axis is the normal of the annulus mask (PCA) when given, oriented toward
    ``toward_aorta`` (e.g. the aorta centroid); otherwise the normal of the plane through the
    three cusp centroids. Each nadir is the mean of the cusp voxels within 0.5 mm of the cusp's
    lowest level, which is less sensitive to a single noisy voxel than the minimum alone.
    """
    pts = {}
    for name, v in cusps.items():
        idx = np.argwhere(v.data > 0)
        if len(idx) == 0:
            raise ValueError(f"cusp '{name}' is empty")
        pts[name] = v.voxel_centers_world(idx)
    if annulus is not None and np.any(annulus.data > 0):
        ring = annulus.voxel_centers_world(np.argwhere(annulus.data > 0))
        plane = fit_plane(ring, toward=toward_aorta)
    else:
        plane = fit_plane(np.array([p.mean(axis=0) for p in pts.values()]), toward=toward_aorta)
    out = {}
    for name, p in pts.items():
        h = (p - plane.center) @ plane.normal
        low = p[h <= h.min() + 0.5]
        out[CUSP_LABELS.get(name, name)] = tuple(float(x) for x in low.mean(axis=0))
    return out
