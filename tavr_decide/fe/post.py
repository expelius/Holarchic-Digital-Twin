"""Mechanistic indices read from a deployed configuration.

Two quantities feed the decision layer:

* **sealing gap** – the area left between the envelope of the frame's outer surface and
  the deformed inner wall in a band around the annulus. It is the mid-rung of the
  paravalvular-leak ladder (calcium volume -> gap area -> CFD leak volume).
* **wall displacement below the membranous septum** – how far the frame pushes the wall
  outward in the part of the LVOT that lies below the inferior border of the membranous
  septum, where the conduction tissue runs. It is the mechanistic counterpart of dMSID
  and follows the use of implantation-site deformation as a conduction marker
  (Bosi et al. 2020).

Both are geometric, computed from node positions only, so they do not depend on how the
solver reports contact.
"""
from __future__ import annotations

import numpy as np

from .deploy import DeploymentResult


def _envelope(theta: np.ndarray, r: np.ndarray, n_bins: int) -> np.ndarray:
    """Outer envelope r(theta): maximum radius per angular bin, empty bins filled circularly."""
    b = np.floor((theta % (2 * np.pi)) / (2 * np.pi) * n_bins).astype(int) % n_bins
    env = np.full(n_bins, np.nan)
    for k in range(n_bins):
        m = b == k
        if m.any():
            env[k] = r[m].max()
    if np.isnan(env).all():
        raise ValueError("no frame nodes in the requested band")
    idx = np.arange(n_bins)
    good = ~np.isnan(env)
    env[~good] = np.interp(idx[~good], np.r_[idx[good] - n_bins, idx[good], idx[good] + n_bins],
                           np.r_[env[good], env[good], env[good]])
    return env


def sealing_gap(res: DeploymentResult, z_band: tuple[float, float] = (-3.0, 1.0), n_bins: int = 72,
                tol_mm: float = 0.05) -> dict:
    """Gap between the frame envelope and the deformed inner wall, in ``z_band`` (local z)."""
    if res.frame_final is None or res.vessel_final is None:
        raise ValueError("deployment has no final positions")
    f_ids = np.unique(res.frame_mesh.outer_faces)
    fz = res.frame_final[f_ids, 2]
    sel = (fz >= z_band[0]) & (fz <= z_band[1])
    if not sel.any():
        return {"area_mm2": float("nan"), "max_gap_mm": float("nan"), "covered": False}
    fp = res.frame_final[f_ids][sel]
    env = _envelope(np.arctan2(fp[:, 1], fp[:, 0]), np.linalg.norm(fp[:, :2], axis=1), n_bins)
    inner = res.vessel_mesh.junction_center_nodes
    zref = res.vessel_mesh.nodes[inner[:, 0], 2]
    rows = (zref >= z_band[0]) & (zref <= z_band[1])
    if not rows.any():
        return {"area_mm2": float("nan"), "max_gap_mm": float("nan"), "covered": False}
    vp = res.vessel_final[inner[rows].ravel()]
    vth = np.arctan2(vp[:, 1], vp[:, 0]); vr = np.linalg.norm(vp[:, :2], axis=1)
    b = np.floor((vth % (2 * np.pi)) / (2 * np.pi) * n_bins).astype(int) % n_bins
    wall = np.array([vr[b == k].mean() if np.any(b == k) else np.nan for k in range(n_bins)])
    good = ~np.isnan(wall)
    idx = np.arange(n_bins)
    wall[~good] = np.interp(idx[~good], np.r_[idx[good] - n_bins, idx[good], idx[good] + n_bins],
                            np.r_[wall[good], wall[good], wall[good]])
    gap = np.clip(wall - env - tol_mm, 0.0, None)
    dth = 2 * np.pi / n_bins
    area = float(np.sum(gap * (env + gap / 2) * dth))
    return {"area_mm2": area, "max_gap_mm": float(gap.max()), "gap_by_angle_mm": gap, "covered": True}


def wall_displacement(res: DeploymentResult, z_max: float, z_min: float | None = None) -> dict:
    """Outward displacement of the inner wall for reference heights in ``[z_min, z_max]``."""
    if res.vessel_final is None:
        raise ValueError("deployment has no final wall positions")
    inner = res.vessel_mesh.junction_center_nodes
    zref = res.vessel_mesh.nodes[inner[:, 0], 2]
    rows = zref <= z_max + 1e-9
    if z_min is not None:
        rows &= zref >= z_min - 1e-9
    if not rows.any():
        return {"p90_mm": 0.0, "mean_mm": 0.0, "max_mm": 0.0, "n": 0}
    ids = inner[rows].ravel()
    r0 = np.linalg.norm(res.vessel_mesh.nodes[ids, :2], axis=1)
    r1 = np.linalg.norm(res.vessel_final[ids, :2], axis=1)
    d = np.clip(r1 - r0, 0.0, None)
    return {"p90_mm": float(np.percentile(d, 90)), "mean_mm": float(d.mean()), "max_mm": float(d.max()), "n": int(len(d))}
