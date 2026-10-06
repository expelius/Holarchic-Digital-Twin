"""One call from a frame specification to a deployed configuration.

This is the works of the deployment holon's high-fidelity rung: build the meshes, write
the FEBio model, run it, read back the deployed frame and wall. The result carries the
numbers the decision layer and the validation against imaging need (radius by axial band,
expansion relative to the free shape, final node positions) together with wall time and
termination status, because the cost of this rung is part of its contract.

The landing zone is either an idealised cylinder (``vessel_radius_mm``) or a
patient-specific wall (:class:`~tavr_decide.fe.vessel.PatientVessel`) with its calcified
elements. ``inflow_z`` places the frame's inflow edge in the vessel's local frame:
``inflow_z = -depth`` for an implantation depth measured below the annulus plane.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..frame import FrameSpec
from .febio import DeploymentParams, assemble, read_node_positions, run_febio, write_deployment
from .lattice import HexMesh, cylinder_hex_mesh, lattice_hex_mesh, tube_hex_mesh


@dataclass
class DeploymentResult:
    normal_termination: bool
    wall_s: float
    n_nodes: int
    n_elems: int
    times: np.ndarray
    mean_radius_t: np.ndarray                 # frame mean radius over time
    band_z: np.ndarray                        # axial band centres (reference z)
    band_radius_free: np.ndarray              # free (stress-free) radius per band
    band_radius_deployed: np.ndarray          # deployed radius per band
    equilibrium_drift_mm: float               # change of mean radius over the last 10 % of the release
    workdir: Path
    params: dict = field(default_factory=dict)
    frame_mesh: HexMesh | None = None
    vessel_mesh: HexMesh | None = None
    frame_final: np.ndarray | None = None     # (N, 3) deployed frame nodes
    vessel_final: np.ndarray | None = None    # (N, 3) deformed wall nodes
    calcified: np.ndarray | None = None

    @property
    def expansion_ratio(self) -> np.ndarray:
        return self.band_radius_deployed / self.band_radius_free

    def summary(self) -> str:
        lines = [f"termination: {'NORMAL' if self.normal_termination else 'ERROR'}   wall: {self.wall_s:.0f} s   "
                 f"nodes: {self.n_nodes}   hexes: {self.n_elems}   equilibrium drift: {self.equilibrium_drift_mm:.4f} mm",
                 "  z (mm)   free r   deployed r   expansion"]
        for z, a, b in zip(self.band_z, self.band_radius_free, self.band_radius_deployed):
            lines.append(f"  {z:6.1f}  {a:7.2f}  {b:10.2f}  {b / a:10.3f}")
        return "\n".join(lines)


def run_deployment(spec: FrameSpec, vessel_radius_mm: float | None, workdir: str | Path,
                   vessel=None, inflow_z: float = 0.0,
                   vessel_z: tuple[float, float] | None = None, vessel_thickness_mm: float = 2.0,
                   n_along: int = 2, n_theta: int = 48, n_z_vessel: int = 6, n_r_vessel: int = 2,
                   gap_mm: float = 0.3, crimp_radius_mm: float | None = None, crimp_margin_mm: float = 1.5, n_bands: int = 6,
                   params: DeploymentParams | None = None, timeout_s: int = 7200, threads: int = 4) -> DeploymentResult:
    """Crimp ``spec`` with a sleeve and release it into the landing zone.

    Give either ``vessel_radius_mm`` (idealised cylinder; ``vessel_z`` defaults to the first
    third of the frame plus 3 mm below it) or ``vessel`` (a ``PatientVessel``).
    """
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    frame = lattice_hex_mesh(spec, n_along=n_along, n_thick=1)
    frame.nodes[:, 2] += inflow_z
    r_free = np.linalg.norm(frame.nodes[:, :2], axis=1)
    r_out_max = float(r_free.max())
    r0 = r_out_max + gap_mm
    H = spec.height_mm
    z_lo, z_hi = inflow_z, inflow_z + H
    calcified = None
    if vessel is not None:
        vmesh = vessel.mesh
        calcified = vessel.calcified
        v_axis_r = np.linalg.norm(vmesh.nodes[vmesh.junction_center_nodes.ravel(), :2], axis=1)
        zv = vmesh.nodes[vmesh.junction_center_nodes.ravel(), 2]
        overlap = (zv >= z_lo - 1e-9) & (zv <= z_hi + 1e-9)
        if not overlap.any():
            raise ValueError("the frame does not overlap the vessel segment axially")
        r_min = float(v_axis_r[overlap].min())
        r_vessel_outer = float(np.linalg.norm(vmesh.nodes[:, :2], axis=1).max())
        landing = float(v_axis_r[overlap].mean())
    else:
        if vessel_radius_mm is None:
            raise ValueError("give vessel_radius_mm or vessel")
        if vessel_z is None:
            vessel_z = (z_lo - 3.0, z_lo + max(H / 3.0, 8.0))
        vmesh = cylinder_hex_mesh(vessel_radius_mm, vessel_thickness_mm, vessel_z[0], vessel_z[1],
                                  n_theta=n_theta, n_z=n_z_vessel, n_r=n_r_vessel)
        r_min = landing = float(vessel_radius_mm)
        r_vessel_outer = vessel_radius_mm + vessel_thickness_mm
    crimp = crimp_radius_mm if crimp_radius_mm is not None else r_min - crimp_margin_mm
    # The sleeve follows the frame's outer profile and is scaled uniformly, so every level
    # is compressed by the same ratio. A cylindrical sleeve crushes the flared outflow much
    # more than the inflow and makes the lattice unstable once the mesh stops locking.
    n_z_sleeve = max(6, int(round((H + 2.0) / 2.0)))
    zs_s = np.linspace(z_lo - 1.0, z_hi + 1.0, n_z_sleeve + 1)
    r_sleeve = spec.radius_at((zs_s - inflow_z) / H) + spec.strut_thickness_mm / 2 + gap_mm
    sleeve = tube_hex_mesh(np.repeat(r_sleeve[:, None], n_theta, axis=1), np.zeros((n_z_sleeve + 1, 2)), zs_s, 0.5, 1)
    # frame outer radius inside the landing zone decides how far to crimp
    zf = frame.nodes[:, 2]
    if vessel is not None:
        zv_lo, zv_hi = float(vmesh.nodes[:, 2].min()), float(vmesh.nodes[:, 2].max())
    else:
        zv_lo, zv_hi = vessel_z
    in_zone = (zf >= zv_lo - 1e-9) & (zf <= zv_hi + 1e-9)
    r_frame_zone = float(r_free[in_zone].max()) if in_zone.any() else r_out_max
    p = params or DeploymentParams()
    p.sleeve_r0, p.sleeve_r_crimp = r0, crimp
    p.sleeve_r_final = max(r_out_max, r_vessel_outer) + 2.0
    p.sleeve_scale_crimp = crimp / (r_frame_zone + gap_mm)
    p.sleeve_scale_final = max(1.15, (r_vessel_outer + 2.0) / float(r_sleeve.min()))
    A = assemble(frame, vmesh, sleeve, calcified)
    feb = write_deployment(workdir / "deploy.feb", A, p)
    run = run_febio(feb, threads=threads, timeout_s=timeout_s)

    fpos = read_node_positions(workdir / "frame_nodes.txt") if (workdir / "frame_nodes.txt").exists() else {}
    vpos = read_node_positions(workdir / "vessel_nodes.txt") if (workdir / "vessel_nodes.txt").exists() else {}
    times = np.array(sorted(fpos))
    mean_r = np.array([np.linalg.norm(fpos[t][:, 1:3], axis=1).mean() for t in times]) if len(times) else np.zeros(0)
    z_ref = frame.nodes[:, 2]
    edges = np.linspace(z_ref.min(), z_ref.max() + 1e-9, n_bands + 1)
    band = np.clip(np.searchsorted(edges, z_ref, side="right") - 1, 0, n_bands - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    free = np.array([r_free[band == b].mean() for b in range(n_bands)])
    frame_final = vessel_final = None
    if len(times):
        frame_final = fpos[times[-1]][:, 1:4]
        r_dep = np.linalg.norm(frame_final[:, :2], axis=1)
        dep = np.array([r_dep[band == b].mean() for b in range(n_bands)])
        tail = mean_r[times >= times[-1] - 0.1] if times[-1] >= 1.8 else mean_r[-1:]
        drift = float(np.ptp(tail))
        if vpos:
            vessel_final = vpos[max(vpos)][:, 1:4]
    else:
        dep, drift = np.full(n_bands, np.nan), float("nan")
    result = DeploymentResult(run["normal"], run["wall_s"], len(A.nodes), int(sum(len(e) for e in A.parts.values())),
                            times, mean_r, centres, free, dep, drift, workdir,
                            {"landing_radius_mm": landing, "inflow_z": inflow_z, "crimp_radius_mm": crimp,
                             "frame_E_MPa": p.frame_E_MPa, "vessel_E_MPa": p.vessel_E_MPa,
                             "calcium_E_MPa": p.calcium_E_MPa, "n_calcified": int(calcified.sum()) if calcified is not None else 0,
                             "contact_penalty": p.contact_penalty, "vessel_penalty": p.vessel_penalty},
                            frame, vmesh, frame_final, vessel_final, calcified)
    import pickle
    with open(workdir / "result.pkl", "wb") as fh:          # so that downstream holons (CFD) can reload it
        pickle.dump(result, fh)
    return result
