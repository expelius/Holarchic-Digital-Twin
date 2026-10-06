"""The paravalvular channel: the space between the deployed frame's skirt and the wall.

Paravalvular leak flows in diastole from the aorta back into the ventricle through this
space. Its geometry comes straight from the deployment holon: the skirt is modelled as a
membrane draped over the outer surface of the struts (piecewise-linear between them) and
the wall is the deformed inner surface of the landing zone. Both are expressed as radius
fields over (angle, height) in the annulus-aligned local frame, and the channel height
h(theta, z) = r_wall - r_skirt.

Where the frame presses into the wall (contact penalty allows a little penetration) the
channel is closed; it is kept at a minimum height so the structured mesh stays valid, and
that height carries negligible flow (resistance goes as h^-3).

Declared simplifications: the skirt height is a placeholder fraction of the frame; the
native leaflets and their calcium are not in the fluid domain (they would partly seal the
space above the annulus); the skirt does not deform under pressure.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

SKIRT_ROWS = 1.5          # PLACEHOLDER: skirt covers 1.5 cell rows from the inflow edge
H_MIN_MM = 0.05           # closed channel kept at this height (meshing only)


@dataclass
class GapMap:
    theta: np.ndarray           # (n_theta,) bin centres, radians
    z: np.ndarray               # (n_z,) mm, local frame
    r_skirt: np.ndarray         # (n_z, n_theta) mm
    r_wall: np.ndarray          # (n_z, n_theta) mm

    @property
    def h(self) -> np.ndarray:
        return np.clip(self.r_wall - self.r_skirt, 0.0, None)

    @property
    def open_fraction(self) -> float:
        """Fraction of angles whose channel stays open (> 0.1 mm) over its whole height."""
        return float(np.mean(self.h.min(axis=0) > 0.1))

    def min_section_area_mm2(self) -> float:
        """Area of the narrowest cross-section of each angular strip, summed: the geometric
        regurgitant orifice."""
        dth = 2 * np.pi / len(self.theta)
        h = self.h
        k = np.argmin(h, axis=0)
        hm = h[k, np.arange(len(self.theta))]
        rm = self.r_skirt[k, np.arange(len(self.theta))] + hm / 2
        return float(np.sum(rm * dth * hm))


def _surface_field(pts: np.ndarray, theta_q: np.ndarray, z_q: np.ndarray, r_ref: float) -> np.ndarray:
    """Radius field r(theta, z) from scattered surface points, linear on (arc length, z),
    periodic in theta. Points outside the convex hull take the nearest value."""
    th = np.arctan2(pts[:, 1], pts[:, 0]); r = np.linalg.norm(pts[:, :2], axis=1); z = pts[:, 2]
    TH = np.r_[th - 2 * np.pi, th, th + 2 * np.pi]
    X = np.c_[TH * r_ref, np.r_[z, z, z]]
    V = np.r_[r, r, r]
    G_t, G_z = np.meshgrid(theta_q, z_q)
    Q = np.c_[G_t.ravel() * r_ref, G_z.ravel()]
    lin = griddata(X, V, Q, method="linear")
    bad = np.isnan(lin)
    if bad.any():
        lin[bad] = griddata(X, V, Q[bad], method="nearest")
    return lin.reshape(G_z.shape)


def gap_map(res, z_range: tuple[float, float] | None = None, n_theta: int = 72, n_z: int = 24,
            skirt_rows: float = SKIRT_ROWS) -> GapMap:
    """Channel height field from a deployment result (see ``tavr_decide.fe.deploy``).

    Default height range: from just above the frame's inflow edge to the top of the skirt,
    clipped to the wall segment that was modelled.
    """
    if res.frame_final is None or res.vessel_final is None:
        raise ValueError("deployment has no final positions")
    fm, vm = res.frame_mesh, res.vessel_mesh
    face_ids = np.unique(fm.outer_faces)
    f_pts = res.frame_final[face_ids]
    centroids = res.frame_final[fm.outer_faces].mean(axis=1)
    f_pts = np.vstack([f_pts, centroids])
    w_ids = vm.junction_center_nodes.ravel()
    w_pts = res.vessel_final[w_ids]
    z_in = float(res.frame_final[:, 2].min())
    if z_range is None:
        spec_h = float(np.ptp(res.frame_mesh.nodes[:, 2]))
        rows = max(len(res.frame_mesh.junction_center_nodes) - 1, 1)
        z_top = z_in + skirt_rows * spec_h / rows
        z_range = (z_in + 0.25, min(z_top, float(w_pts[:, 2].max()) - 0.25))
    theta = (np.arange(n_theta) + 0.5) * 2 * np.pi / n_theta
    z = np.linspace(z_range[0], z_range[1], n_z)
    r_ref = float(np.linalg.norm(f_pts[:, :2], axis=1).mean())
    rs = _surface_field(f_pts, theta, z, r_ref)
    rw = _surface_field(w_pts, theta, z, r_ref)
    return GapMap(theta, z, rs, rw)


@dataclass
class ChannelMesh:
    nodes: np.ndarray           # (N, 3) cm
    elems: np.ndarray           # (E, 8) hex8, zero-based
    faces: dict                 # name -> (F, 4) quads, zero-based node ids, outward normals
    face_elems: dict            # name -> (F,) owning element ids


def channel_mesh(g: GapMap, n_h: int = 4, h_min_mm: float = H_MIN_MM, scale: float = 0.1) -> ChannelMesh:
    """Structured hexahedral mesh of the channel, periodic in angle; ``scale`` converts mm to
    the solver units (0.1: centimetres, the usual cgs choice for cardiovascular CFD).

    Faces: 'aortic' (top, flow comes in during diastole), 'ventricular' (bottom, flow leaves),
    'skirt' (inner wall), 'wall' (outer wall).
    """
    nz, nt = g.r_skirt.shape
    h = np.maximum(g.h, h_min_mm)

    def nid(k, i, j):              # k: z level, i: theta, j: layer across the gap
        return (k * nt + (i % nt)) * (n_h + 1) + j

    nodes = np.zeros((nz * nt * (n_h + 1), 3))
    for k in range(nz):
        for i in range(nt):
            for j in range(n_h + 1):
                r = g.r_skirt[k, i] + h[k, i] * j / n_h
                nodes[nid(k, i, j)] = (r * np.cos(g.theta[i]), r * np.sin(g.theta[i]), g.z[k])
    elems, faces, owners = [], {"aortic": [], "ventricular": [], "skirt": [], "wall": []}, \
        {"aortic": [], "ventricular": [], "skirt": [], "wall": []}
    for k in range(nz - 1):
        for i in range(nt):
            for j in range(n_h):
                a = [nid(k, i, j), nid(k, i, j + 1), nid(k, i + 1, j + 1), nid(k, i + 1, j)]
                b = [nid(k + 1, i, j), nid(k + 1, i, j + 1), nid(k + 1, i + 1, j + 1), nid(k + 1, i + 1, j)]
                e = len(elems)
                elems.append(a + b)
                if k == 0:
                    faces["ventricular"].append(a); owners["ventricular"].append(e)
                if k == nz - 2:
                    faces["aortic"].append(b); owners["aortic"].append(e)
                if j == 0:
                    faces["skirt"].append([a[0], a[3], b[3], b[0]]); owners["skirt"].append(e)
                if j == n_h - 1:
                    faces["wall"].append([a[1], b[1], b[2], a[2]]); owners["wall"].append(e)
    nodes *= scale
    E = np.array(elems, dtype=int)
    from ..fe.lattice import hex_signed_volumes
    vol = hex_signed_volumes(nodes, E)
    if (vol < 0).any():
        E[vol < 0] = E[vol < 0][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    out = {}
    centre = nodes[E].mean(axis=1)
    for name, F in faces.items():
        F = np.array(F, dtype=int); own = np.array(owners[name], dtype=int)
        p = nodes[F]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
        inward = np.einsum("ij,ij->i", n, centre[own] - p.mean(axis=1)) > 0
        F[inward] = F[inward][:, [0, 3, 2, 1]]
        out[name] = F
    return ChannelMesh(nodes, E, out, {k: np.array(v, dtype=int) for k, v in owners.items()})
