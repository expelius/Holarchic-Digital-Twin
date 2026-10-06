"""Conforming hexahedral mesh of a diamond-lattice frame.

The lattice is built unrolled in the (arc, axial) plane and wrapped onto the frame's
radial profile. Every junction is a rhombus split into four quads (nine planar nodes);
every strut is a strip two elements wide between the facing edges of two junctions, so
struts share their end nodes with the junctions they meet. Two elements across the strut
width are the minimum for in-plane bending, which is how a diamond cell opens and closes.

Planar construction. With circumferential pitch ``px`` and axial pitch ``pz`` the strut
direction is d = (px/2, pz)/L. A junction rhombus with vertices T = (0, w sx),
R = (w sy, 0), B = -T, L = -R has every edge of length w and each edge perpendicular
to one of the four strut directions only at 45 degrees; elsewhere the strut is a
slightly sheared parallelogram, which is acceptable for a parametric frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frame import FrameSpec

# local junction node slots
T, R, B, L, MTR, MRB, MBL, MLT, C = range(9)


@dataclass
class HexMesh:
    nodes: np.ndarray            # (N, 3)
    elems: np.ndarray            # (E, 8) zero-based, FEBio/VTK hex8 ordering
    outer_faces: np.ndarray      # (F, 4) quads on the outer radial surface, normal pointing outward
    inner_faces: np.ndarray      # (F, 4) quads on the inner radial surface, normal pointing inward
    junction_center_nodes: np.ndarray   # (levels, n_cells) node index of each junction centre (inner layer)

    def min_scaled_jacobian(self) -> float:
        return float(hex_signed_volumes(self.nodes, self.elems).min())


def hex_signed_volumes(nodes: np.ndarray, elems: np.ndarray) -> np.ndarray:
    """Signed volume proxy at the element centre (positive for a valid hex8 ordering)."""
    p = nodes[elems]                                   # (E, 8, 3)
    dxi = (p[:, [1, 2, 6, 5]].mean(1) - p[:, [0, 3, 7, 4]].mean(1))
    deta = (p[:, [3, 2, 6, 7]].mean(1) - p[:, [0, 1, 5, 4]].mean(1))
    dzeta = (p[:, [4, 5, 6, 7]].mean(1) - p[:, [0, 1, 2, 3]].mean(1))
    return np.einsum("ij,ij->i", np.cross(dxi, deta), dzeta)


def lattice_hex_mesh(spec: FrameSpec, n_along: int = 4, n_thick: int = 1) -> HexMesh:
    """Mesh the frame described by ``spec`` (see :mod:`tavr_decide.frame`)."""
    N, M = spec.n_cells_circ, spec.n_rows
    r_ref = float(np.mean([p[1] for p in spec.profile]))
    px = 2 * np.pi * r_ref / N
    pz = spec.height_mm / M
    Ls = np.hypot(px / 2, pz)
    sx, sy = (px / 2) / Ls, pz / Ls
    w, t = spec.strut_width_mm, spec.strut_thickness_mm
    if w * sy * 2 >= px / 2 or w * sx * 2 >= pz:
        raise ValueError("strut width too large for this lattice pitch")
    rel = np.zeros((9, 2))
    rel[T], rel[R], rel[B], rel[L] = (0, w * sx), (w * sy, 0), (0, -w * sx), (-w * sy, 0)
    rel[MTR], rel[MRB], rel[MBL], rel[MLT] = (rel[T] + rel[R]) / 2, (rel[R] + rel[B]) / 2, (rel[B] + rel[L]) / 2, (rel[L] + rel[T]) / 2
    rel[C] = (0, 0)

    planar: list[tuple[float, float]] = []            # (arc x, axial z)

    def add(p) -> int:
        planar.append((float(p[0]), float(p[1])))
        return len(planar) - 1

    centers = np.zeros((M + 1, N, 2))
    jn = np.zeros((M + 1, N, 9), dtype=int)
    for lv in range(M + 1):
        for c in range(N):
            ctr = np.array([(c + 0.5 * (lv % 2)) * px, lv * pz])
            centers[lv, c] = ctr
            for k in range(9):
                jn[lv, c, k] = add(ctr + rel[k])

    quads: list[list[int]] = []
    for lv in range(M + 1):
        for c in range(N):
            j = jn[lv, c]
            quads += [[j[C], j[MRB], j[R], j[MTR]], [j[C], j[MTR], j[T], j[MLT]],
                      [j[C], j[MLT], j[L], j[MBL]], [j[C], j[MBL], j[B], j[MRB]]]

    def strut(a_edge: list[int], b_edge: list[int], offset: np.ndarray) -> None:
        """Strip between edge ``a_edge`` (3 nodes) and ``b_edge`` (3 nodes); ``offset`` is the
        planar vector between corresponding nodes (handles the circumferential wrap)."""
        prev = a_edge
        pa = np.array([planar[i] for i in a_edge])
        for s in range(1, n_along + 1):
            if s == n_along:
                cur = b_edge
            else:
                cur = [add(pa[k] + offset * s / n_along) for k in range(3)]
            for k in range(2):
                quads.append([prev[k], prev[k + 1], cur[k + 1], cur[k]])
            prev = cur

    off_ur = np.array([px / 2 - w * sy, pz - w * sx])
    off_ul = np.array([-px / 2 + w * sy, pz - w * sx])
    for lv in range(M):
        for c in range(N):
            a = jn[lv, c]
            if lv % 2 == 0:
                b_ur, b_ul = jn[lv + 1, c], jn[lv + 1, (c - 1) % N]
            else:
                b_ur, b_ul = jn[lv + 1, (c + 1) % N], jn[lv + 1, c]
            # up-right: A [T, MTR, R] -> B [L, MBL, B]
            strut([a[T], a[MTR], a[R]], [b_ur[L], b_ur[MBL], b_ur[B]], off_ur)
            # up-left:  A [L, MLT, T] -> B [B, MRB, R]
            strut([a[L], a[MLT], a[T]], [b_ul[B], b_ul[MRB], b_ul[R]], off_ul)

    P = np.array(planar)
    theta = P[:, 0] / r_ref
    z = P[:, 1]
    r_mid = spec.radius_at(z / spec.height_mm)
    n_planar = len(P)
    layers = []
    for q in range(n_thick + 1):
        r = r_mid - t / 2 + t * q / n_thick
        layers.append(np.c_[r * np.cos(theta), r * np.sin(theta), z])
    nodes = np.vstack(layers)

    Q = np.array(quads, dtype=int)
    elems = []
    for q in range(n_thick):
        lo, hi = Q + q * n_planar, Q + (q + 1) * n_planar
        elems.append(np.hstack([lo, hi]))
    elems = np.vstack(elems)
    vol = hex_signed_volumes(nodes, elems)
    flip = vol < 0
    if flip.any():                                     # reverse the in-plane orientation of those quads
        e = elems[flip]
        elems[flip] = e[:, [0, 3, 2, 1, 4, 7, 6, 5]]
        Qf = Q.copy()
    # orient surface quads by geometry: outer normal +r, inner normal -r
    def oriented(face_nodes: np.ndarray, outward: bool) -> np.ndarray:
        p = nodes[face_nodes]
        nrm = np.cross(p[:, 1] - p[:, 0], p[:, 3] - p[:, 0])
        ctr = p.mean(1); radial = np.c_[ctr[:, 0], ctr[:, 1], np.zeros(len(ctr))]
        s = np.einsum("ij,ij->i", nrm, radial)
        bad = (s < 0) if outward else (s > 0)
        out = face_nodes.copy()
        out[bad] = out[bad][:, [0, 3, 2, 1]]
        return out

    outer = oriented(Q + n_thick * n_planar, outward=True)
    inner = oriented(Q.copy(), outward=False)
    return HexMesh(nodes, elems, outer, inner, jn[:, :, C].copy())


def tube_hex_mesh(inner_radius: np.ndarray, centers: np.ndarray, zs: np.ndarray, thickness: float,
                  n_r: int = 1) -> HexMesh:
    """Structured thick-walled tube with an arbitrary inner surface.

    ``inner_radius[iz, it]`` is the lumen radius about ``centers[iz]`` (x, y) at height
    ``zs[iz]`` and angle ``2 pi it / n_theta``; the wall is extruded ``thickness`` outward
    along the in-plane radial direction. ``junction_center_nodes[iz, it]`` holds the inner
    surface node indices (used for boundary conditions and post-processing).
    """
    n_zp, n_theta = inner_radius.shape
    n_z = n_zp - 1
    th = 2 * np.pi * np.arange(n_theta) / n_theta

    def nid(ir, it, iz):
        return (ir * (n_z + 1) + iz) * n_theta + (it % n_theta)

    nodes = np.zeros(((n_r + 1) * (n_z + 1) * n_theta, 3))
    for ir in range(n_r + 1):
        for iz in range(n_z + 1):
            rho = inner_radius[iz] + thickness * ir / n_r
            idx = [nid(ir, it, iz) for it in range(n_theta)]
            nodes[idx, 0] = centers[iz, 0] + rho * np.cos(th)
            nodes[idx, 1] = centers[iz, 1] + rho * np.sin(th)
            nodes[idx, 2] = zs[iz]
    elems, inner, outer = [], [], []
    for ir in range(n_r):
        for iz in range(n_z):
            for it in range(n_theta):
                lo = [nid(ir, it, iz), nid(ir, it + 1, iz), nid(ir, it + 1, iz + 1), nid(ir, it, iz + 1)]
                hi = [nid(ir + 1, it, iz), nid(ir + 1, it + 1, iz), nid(ir + 1, it + 1, iz + 1), nid(ir + 1, it, iz + 1)]
                elems.append(lo + hi)
                if ir == 0:
                    inner.append(lo[::-1])             # normal into the lumen
                if ir == n_r - 1:
                    outer.append(hi)                   # normal outward
    elems = np.array(elems, dtype=int)
    vol = hex_signed_volumes(nodes, elems)
    if (vol < 0).any():
        e = elems[vol < 0]
        elems[vol < 0] = e[:, [0, 3, 2, 1, 4, 7, 6, 5]]
    inner_ids = np.array([[nid(0, it, iz) for it in range(n_theta)] for iz in range(n_z + 1)])
    return HexMesh(nodes, elems, np.array(outer, dtype=int), np.array(inner, dtype=int), inner_ids)


def cylinder_hex_mesh(r_inner: float, thickness: float, z0: float, z1: float,
                      n_theta: int = 24, n_z: int = 6, n_r: int = 1) -> HexMesh:
    """Structured thick-walled cylinder (vessel segment or crimping sleeve)."""
    zs = np.linspace(z0, z1, n_z + 1)
    return tube_hex_mesh(np.full((n_z + 1, n_theta), float(r_inner)), np.zeros((n_z + 1, 2)), zs, thickness, n_r)
