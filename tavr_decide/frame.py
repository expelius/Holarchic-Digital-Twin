"""Parametric transcatheter valve frames.

Why parametric: no manufacturer publishes CAD for Evolut, SAPIEN, Navitor or ACURATE
frames, and the legal status of redistributing micro-CT reconstructions is unresolved.
A parametric frame built from published dimensions is sufficient for a decision-logic
benchmark (it is NOT sufficient for clinical validation of frame–anatomy interaction,
and the report says so). Any group with a real device model plugs it in through the same
``FrameMesh`` contract.

Two families:

* ``self_expanding_frame`` — diamond-cell lattice with an inflow / waist / outflow radial
  profile (Evolut-like); nitinol, deployed by releasing a crimped state.
* ``balloon_expandable_frame`` — short cylindrical lattice (SAPIEN-like); cobalt-chromium,
  deployed by balloon inflation (plastic).

The mesh is a set of straight strut segments (centre-line graph) plus a rectangular
cross-section; exporters give (a) a line mesh for visualisation and beam models and
(b) a hexahedral solid mesh for solvers that prefer continuum struts. Units: mm.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FrameSpec:
    name: str
    family: str                      # 'self-expanding' | 'balloon-expandable'
    n_cells_circ: int                # diamond cells around the circumference
    n_rows: int                      # cell rows along the axis
    height_mm: float
    # radial profile as (axial fraction 0..1 from inflow, radius mm) control points
    profile: list[tuple[float, float]]
    strut_width_mm: float = 0.30
    strut_thickness_mm: float = 0.25
    material: str = "nitinol"
    nominal_size_mm: int = 26
    notes: str = ""

    def radius_at(self, s: np.ndarray) -> np.ndarray:
        f = np.array([p[0] for p in self.profile]); r = np.array([p[1] for p in self.profile])
        return np.interp(np.clip(s, 0, 1), f, r)


@dataclass
class FrameMesh:
    nodes: np.ndarray                # (N, 3) mm, expanded configuration
    segments: np.ndarray             # (S, 2) node indices (strut centre-lines)
    spec: FrameSpec
    row_of_node: np.ndarray = field(default_factory=lambda: np.zeros(0, int))

    # ---- contract used by the deployment holon --------------------------------------------
    @property
    def inflow_z(self) -> float:
        return float(self.nodes[:, 2].min())

    @property
    def outflow_z(self) -> float:
        return float(self.nodes[:, 2].max())

    def radius_profile(self, n_bins: int = 12) -> np.ndarray:
        """Mean radius per axial bin (used to compare with FE output and with post-TAVR imaging)."""
        z = self.nodes[:, 2]; r = np.linalg.norm(self.nodes[:, :2], axis=1)
        edges = np.linspace(z.min(), z.max() + 1e-9, n_bins + 1)
        idx = np.clip(np.searchsorted(edges, z, side="right") - 1, 0, n_bins - 1)
        return np.array([r[idx == b].mean() if np.any(idx == b) else np.nan for b in range(n_bins)])

    def crimped(self, radius_mm: float) -> "FrameMesh":
        """Radial mapping to a crimped cylinder of the given radius (axial coordinate kept).
        This is a geometric initial state for the solver's release step, not a mechanical crimp."""
        r = np.linalg.norm(self.nodes[:, :2], axis=1)
        scale = np.where(r > 1e-9, radius_mm / np.maximum(r, 1e-9), 1.0)
        n = self.nodes.copy(); n[:, 0] *= scale; n[:, 1] *= scale
        return FrameMesh(n, self.segments.copy(), self.spec, self.row_of_node.copy())

    def strut_length_total_mm(self) -> float:
        a, b = self.nodes[self.segments[:, 0]], self.nodes[self.segments[:, 1]]
        return float(np.linalg.norm(a - b, axis=1).sum())

    # ---- exporters ----------------------------------------------------------------------
    def to_vtk_lines(self, path: str) -> None:
        """Legacy VTK polydata with lines (opens in ParaView / 3D Slicer)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write("# vtk DataFile Version 3.0\nparametric frame\nASCII\nDATASET POLYDATA\n")
            f.write(f"POINTS {len(self.nodes)} float\n")
            for p in self.nodes:
                f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
            f.write(f"LINES {len(self.segments)} {3 * len(self.segments)}\n")
            for s in self.segments:
                f.write(f"2 {s[0]} {s[1]}\n")

    def to_hex_mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """One hexahedron per strut segment with the spec's rectangular cross-section
        (width tangential, thickness radial). Returns (nodes (8S,3), elements (S,8))."""
        w, t = self.spec.strut_width_mm / 2, self.spec.strut_thickness_mm / 2
        nodes, elems = [], []
        for k, (i, j) in enumerate(self.segments):
            a, b = self.nodes[i], self.nodes[j]
            axis = b - a; L = np.linalg.norm(axis)
            if L < 1e-9:
                continue
            axis = axis / L
            mid = (a + b) / 2
            radial = np.array([mid[0], mid[1], 0.0]); nr = np.linalg.norm(radial)
            radial = radial / nr if nr > 1e-9 else np.array([1.0, 0.0, 0.0])
            radial = radial - (radial @ axis) * axis
            nr = np.linalg.norm(radial); radial = radial / nr if nr > 1e-9 else np.array([1.0, 0.0, 0.0])
            tang = np.cross(axis, radial)
            corners = []
            for p in (a, b):
                for st, sw in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                    corners.append(p + st * t * radial + sw * w * tang)
            base = len(nodes)
            nodes.extend(corners)
            elems.append([base + q for q in range(8)])
        return np.array(nodes), np.array(elems, dtype=int)


def _lattice(spec: FrameSpec, z_shift: float = 0.0) -> FrameMesh:
    """Diamond lattice: nodes at the vertices of a zig-zag ring per row boundary; struts
    connect each ring to the next, alternating circumferential offset (half a cell)."""
    n = spec.n_cells_circ
    levels = spec.n_rows + 1
    nodes, rows = [], []
    for lv in range(levels):
        s = lv / spec.n_rows
        z = s * spec.height_mm + z_shift
        r = float(spec.radius_at(np.array([s]))[0])
        offset = 0.5 if lv % 2 else 0.0
        for c in range(n):
            th = 2 * np.pi * (c + offset) / n
            nodes.append([r * np.cos(th), r * np.sin(th), z]); rows.append(lv)
    nodes = np.array(nodes); rows = np.array(rows)
    segs = []
    for lv in range(levels - 1):
        for c in range(n):
            a = lv * n + c
            if lv % 2 == 0:
                segs.append([a, (lv + 1) * n + c]); segs.append([a, (lv + 1) * n + (c - 1) % n])
            else:
                segs.append([a, (lv + 1) * n + c]); segs.append([a, (lv + 1) * n + (c + 1) % n])
    return FrameMesh(nodes, np.array(segs, dtype=int), spec, rows)


def self_expanding_frame(size_mm: int = 26, n_cells_circ: int = 15, n_rows: int = 5,
                         height_mm: float | None = None) -> FrameMesh:
    """Evolut-like supra-annular self-expanding frame.

    Dimensions are PLACEHOLDERS from publicly described proportions: inflow diameter equals
    the nominal size, a constrained waist (~0.80 of inflow) and a flared outflow (~1.2 of
    inflow) over a height of ~45 mm for 26 mm. Refit against a measured device before
    making any geometric claim.
    """
    h = height_mm or (45.0 * size_mm / 26.0)
    r_in = size_mm / 2
    spec = FrameSpec(name=f"SE-{size_mm}", family="self-expanding", n_cells_circ=n_cells_circ, n_rows=n_rows,
                     height_mm=h, profile=[(0.0, r_in), (0.25, 0.92 * r_in), (0.45, 0.80 * r_in),
                                           (0.75, 0.95 * r_in), (1.0, 1.20 * r_in)],
                     strut_width_mm=0.30, strut_thickness_mm=0.25, material="nitinol", nominal_size_mm=size_mm,
                     notes="placeholder proportions; not a manufacturer geometry")
    return _lattice(spec)


def balloon_expandable_frame(size_mm: int = 26, n_cells_circ: int = 12, n_rows: int = 3,
                             height_mm: float | None = None) -> FrameMesh:
    """SAPIEN-like short cylindrical balloon-expandable frame (placeholder proportions)."""
    h = height_mm or (20.0 * size_mm / 26.0)
    r = size_mm / 2
    spec = FrameSpec(name=f"BE-{size_mm}", family="balloon-expandable", n_cells_circ=n_cells_circ, n_rows=n_rows,
                     height_mm=h, profile=[(0.0, r), (1.0, r)], strut_width_mm=0.40, strut_thickness_mm=0.45,
                     material="cobalt-chromium", nominal_size_mm=size_mm,
                     notes="placeholder proportions; not a manufacturer geometry")
    return _lattice(spec)
