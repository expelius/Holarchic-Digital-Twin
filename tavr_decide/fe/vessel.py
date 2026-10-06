"""Patient-specific landing zone: from the lumen segmentation to a structured wall mesh.

Local frame: origin at the annulus-plane centre, x = plane.u, y = plane.v, z = plane normal
(toward the aorta). At each axial station the lumen section is sampled by rays from the
section centroid; the wall is extruded outward by a uniform thickness. Elements whose
interior samples exceed the calcium HU threshold are flagged and receive a stiffer
material.

Declared simplifications: uniform wall thickness (CT does not resolve the wall); native
leaflets are not modelled; sections are assumed star-shaped about their centroid, and
rays are clipped so that coronary ostia do not leak into the surface.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry import CALCIUM_HU_THRESHOLD, Plane, Volume
from .lattice import HexMesh, tube_hex_mesh


@dataclass
class PatientVessel:
    mesh: HexMesh
    calcified: np.ndarray        # (E,) bool
    inner_radius: np.ndarray     # (n_z+1, n_theta) about each section centroid
    centers: np.ndarray          # (n_z+1, 2) section centroids, local x-y
    zs: np.ndarray

    def min_radius(self, z0: float, z1: float) -> float:
        """Smallest distance from the local z axis to the inner wall between z0 and z1."""
        sel = (self.zs >= z0 - 1e-9) & (self.zs <= z1 + 1e-9)
        ids = self.mesh.junction_center_nodes[sel].ravel()
        return float(np.linalg.norm(self.mesh.nodes[ids, :2], axis=1).min())

    def mean_radius(self, z0: float, z1: float) -> float:
        sel = (self.zs >= z0 - 1e-9) & (self.zs <= z1 + 1e-9)
        return float(self.inner_radius[sel].mean())


def _local_to_world(plane: Plane, xyz: np.ndarray) -> np.ndarray:
    return plane.center + np.outer(xyz[:, 0], plane.u) + np.outer(xyz[:, 1], plane.v) + np.outer(xyz[:, 2], plane.normal)


def _section(lumen: Volume, plane: Plane, z: float, guess: np.ndarray, n_theta: int,
             r_max: float = 30.0, step: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """Return (centroid_xy, r[theta]) of the lumen section at local height z."""
    from skimage import measure
    a = np.arange(-r_max, r_max + 0.5, 0.5)
    A, B = np.meshgrid(a, a, indexing="ij")
    pts = _local_to_world(plane, np.c_[A.ravel(), B.ravel(), np.full(A.size, z)])
    occ = lumen.sample_linear(pts).reshape(A.shape) >= 0.5
    lab = measure.label(occ, connectivity=1)
    if lab.max() == 0:
        raise ValueError(f"no lumen at z = {z:.1f} mm")
    best, bestd = 0, np.inf
    for k in range(1, lab.max() + 1):                 # component whose centroid is nearest to the guess
        m = lab == k
        if m.sum() < 20:
            continue
        c = np.array([A[m].mean(), B[m].mean()])
        d = np.linalg.norm(c - guess)
        if d < bestd:
            best, bestd, cbest = k, d, c
    if best == 0:
        raise ValueError(f"no usable lumen component at z = {z:.1f} mm")
    th = 2 * np.pi * np.arange(n_theta) / n_theta
    rr = np.arange(0.0, r_max, step)
    X = cbest[0] + np.outer(np.cos(th), rr)
    Y = cbest[1] + np.outer(np.sin(th), rr)
    pts = _local_to_world(plane, np.c_[X.ravel(), Y.ravel(), np.full(X.size, z)])
    o = lumen.sample_linear(pts).reshape(n_theta, len(rr))
    r = np.empty(n_theta)
    for i in range(n_theta):
        below = np.where(o[i] < 0.5)[0]
        if len(below) == 0:
            r[i] = rr[-1]
        elif below[0] == 0:
            r[i] = step
        else:
            j = below[0]                                 # linear interpolation of the 0.5 crossing
            f = (o[i, j - 1] - 0.5) / max(o[i, j - 1] - o[i, j], 1e-9)
            r[i] = rr[j - 1] + f * step
    return cbest, r


def vessel_from_lumen(lumen: Volume, plane: Plane, z_range: tuple[float, float] = (-12.0, 10.0),
                      n_theta: int = 48, n_z: int = 10, n_r: int = 2, thickness_mm: float = 2.0,
                      ct: Volume | None = None, hu_threshold: float = CALCIUM_HU_THRESHOLD,
                      smooth_theta: int = 3, leak_factor: float = 1.6) -> PatientVessel:
    zs = np.linspace(z_range[0], z_range[1], n_z + 1)
    R = np.zeros((n_z + 1, n_theta)); C = np.zeros((n_z + 1, 2))
    guess = np.zeros(2)
    order = np.argsort(np.abs(zs))                     # start at the annulus and walk outward
    done: dict[int, np.ndarray] = {}
    for iz in order:
        near = [k for k in done if abs(k - iz) == 1]
        g = done[near[0]] if near else guess
        c, r = _section(lumen, plane, float(zs[iz]), g, n_theta)
        med = np.median(r)
        r = np.minimum(r, leak_factor * med)           # coronary ostia / side branches
        if smooth_theta > 1:
            k = np.ones(smooth_theta) / smooth_theta
            r = np.convolve(np.r_[r[-smooth_theta:], r, r[:smooth_theta]], k, mode="same")[smooth_theta:-smooth_theta]
        R[iz], C[iz] = r, c
        done[iz] = c
    mesh = tube_hex_mesh(R, C, zs, thickness_mm, n_r)
    calc = np.zeros(len(mesh.elems), dtype=bool)
    if ct is not None:
        P = mesh.nodes[mesh.elems]                     # (E, 8, 3) local
        cen = P.mean(axis=1, keepdims=True)
        samples = np.concatenate([cen, 0.5 * (P + cen)], axis=1)       # centroid + 8 interior points
        hu = ct.sample_nearest(_local_to_world(plane, samples.reshape(-1, 3)), fill=-1000).reshape(len(P), 9)
        calc = (hu >= hu_threshold).sum(axis=1) >= 3
    return PatientVessel(mesh, calc, R, C, zs)
