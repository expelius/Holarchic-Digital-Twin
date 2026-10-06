"""The paravalvular channel: the space between the deployed frame's skirt and the wall.

Paravalvular leak flows in diastole from the aorta back into the ventricle through this
space. Its geometry comes straight from the deployment holon: the skirt is modelled as a
membrane draped over the outer surface of the struts (piecewise-linear between them) and
the wall is the deformed inner surface of the landing zone. Both are expressed as radius
fields over (angle, height) in the annulus-aligned local frame, and the channel height
h(theta, z) = r_wall - r_skirt.

Where the frame presses into the wall the channel is closed. Below ``H_SEAL_MM`` a cell is
treated as contact by both the 0D and the CFD rung: it is not fluid, its faces are walls.

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
H_SEAL_MM = 0.10          # thinner than this is contact, not fluid (used by both 0D and CFD)


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


def fluid_cells(g: GapMap, h_seal_mm: float | None = H_SEAL_MM) -> np.ndarray:
    """Cells (k, i) between heights k..k+1 and angles i..i+1 that are fluid AND belong to a
    component connecting the ventricular (k = 0) to the aortic (k = nz-2) side.
    ``h_seal_mm=None`` makes every cell fluid (uniform gaps, verification)."""
    nz, nt = g.r_skirt.shape
    if h_seal_mm is None:
        return np.ones((nz - 1, nt), dtype=bool)
    hk = g.h
    corner_min = np.minimum.reduce([hk[:-1, :], hk[1:, :], np.roll(hk[:-1, :], -1, axis=1),
                                    np.roll(hk[1:, :], -1, axis=1)])
    fluid = corner_min > h_seal_mm
    cells = list(zip(*np.nonzero(fluid)))
    parent = {c: c for c in cells}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (k, i) in cells:                              # union with upper and next-angle neighbours
        for (kk, ii) in ((k + 1, i), (k, (i + 1) % nt)):
            if kk < nz - 1 and fluid[kk, ii]:
                a, b = root((k, i)), root((kk, ii))
                if a != b:
                    parent[a] = b
    roots = {c: root(c) for c in cells}
    through = {roots[c] for c in cells if c[0] == 0} & {roots[c] for c in cells if c[0] == nz - 2}
    keep = np.zeros_like(fluid)
    for c in cells:
        if roots[c] in through:
            keep[c] = True
    return keep


def channel_mesh(g: GapMap, n_h: int = 4, h_min_mm: float = H_MIN_MM, scale: float = 0.1,
                 h_seal_mm: float | None = H_SEAL_MM) -> ChannelMesh:
    """Structured hexahedral mesh of the channel, periodic in angle; ``scale`` converts mm to
    the solver units (0.1: centimetres, the usual cgs choice for cardiovascular CFD).

    Only the cells returned by :func:`fluid_cells` are meshed: contact cells and fluid pockets
    that do not connect the two sides are left out (the pockets carry no leak and would leave
    the pressure undetermined).

    Faces: 'aortic' (top, flow comes in during diastole), 'ventricular' (bottom, flow leaves),
    'skirt' (inner wall), 'wall' (outer wall), 'seal' (contact boundaries, no slip).
    """
    nz, nt = g.r_skirt.shape
    h = np.maximum(g.h, h_min_mm)
    keep = fluid_cells(g, h_seal_mm)
    if not keep.any():
        raise ValueError("the channel is sealed: no fluid path from the aortic to the ventricular side")

    def nid(k, i, j):              # k: z level, i: theta, j: layer across the gap
        return (k * nt + (i % nt)) * (n_h + 1) + j

    nodes = np.zeros((nz * nt * (n_h + 1), 3))
    for k in range(nz):
        for i in range(nt):
            for j in range(n_h + 1):
                r = g.r_skirt[k, i] + h[k, i] * j / n_h
                nodes[nid(k, i, j)] = (r * np.cos(g.theta[i]), r * np.sin(g.theta[i]), g.z[k])
    names = ("aortic", "ventricular", "skirt", "wall", "seal")
    elems, faces, owners = [], {n: [] for n in names}, {n: [] for n in names}

    def add(name, quad, e):
        faces[name].append(quad); owners[name].append(e)

    for k in range(nz - 1):
        for i in range(nt):
            if not keep[k, i]:
                continue
            for j in range(n_h):
                a = [nid(k, i, j), nid(k, i, j + 1), nid(k, i + 1, j + 1), nid(k, i + 1, j)]
                b = [nid(k + 1, i, j), nid(k + 1, i, j + 1), nid(k + 1, i + 1, j + 1), nid(k + 1, i + 1, j)]
                e = len(elems)
                elems.append(a + b)
                if k == 0:
                    add("ventricular", a, e)
                elif not keep[k - 1, i]:
                    add("seal", a, e)
                if k == nz - 2:
                    add("aortic", b, e)
                elif not keep[k + 1, i]:
                    add("seal", b, e)
                if j == 0:
                    add("skirt", [a[0], a[3], b[3], b[0]], e)
                if j == n_h - 1:
                    add("wall", [a[1], b[1], b[2], a[2]], e)
                if not keep[k, (i - 1) % nt]:
                    add("seal", [a[0], a[1], b[1], b[0]], e)
                if not keep[k, (i + 1) % nt]:
                    add("seal", [a[3], b[3], b[2], a[2]], e)
    E = np.array(elems, dtype=int)
    used = np.unique(E)
    remap = -np.ones(len(nodes), dtype=int)
    remap[used] = np.arange(len(used))
    nodes = nodes[used] * scale
    E = remap[E]
    from ..fe.lattice import hex_signed_volumes
    vol = hex_signed_volumes(nodes, E)
    if (vol < 0).any():
        E[vol < 0] = E[vol < 0][:, [0, 3, 2, 1, 4, 7, 6, 5]]
    out, own_out = {}, {}
    centre = nodes[E].mean(axis=1)
    for name in names:
        if not faces[name]:
            continue
        F = remap[np.array(faces[name], dtype=int)]
        own = np.array(owners[name], dtype=int)
        p = nodes[F]
        n = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
        inward = np.einsum("ij,ij->i", n, centre[own] - p.mean(axis=1)) > 0
        F[inward] = F[inward][:, [0, 3, 2, 1]]
        out[name], own_out[name] = F, own
    return ChannelMesh(nodes, E, out, own_out)
