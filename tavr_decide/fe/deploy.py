"""One call from a frame specification to a deployed configuration.

This is the works of the deployment holon's high-fidelity rung: build the meshes, write
the FEBio model, run it, read back the deployed frame. The result carries the numbers
the decision layer and the validation against imaging need (radius by axial band,
expansion relative to the free shape) together with wall time and termination status,
because the cost of this rung is part of its contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..frame import FrameSpec
from .febio import DeploymentParams, assemble, read_node_positions, run_febio, write_deployment
from .lattice import cylinder_hex_mesh, lattice_hex_mesh


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
    equilibrium_drift_mm: float               # change of mean radius over the last 20 % of the release
    workdir: Path
    params: dict = field(default_factory=dict)

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


def run_deployment(spec: FrameSpec, vessel_radius_mm: float, workdir: str | Path,
                   vessel_z: tuple[float, float] | None = None, vessel_thickness_mm: float = 2.0,
                   n_along: int = 2, n_theta: int = 48, n_z_vessel: int = 6, n_r_vessel: int = 2,
                   gap_mm: float = 0.3, crimp_radius_mm: float | None = None, n_bands: int = 6,
                   params: DeploymentParams | None = None, timeout_s: int = 7200, threads: int = 4) -> DeploymentResult:
    """Crimp ``spec`` with a sleeve and release it into a cylindrical vessel segment.

    ``vessel_z`` is the axial extent of the vessel segment in the frame's coordinates
    (inflow at z = 0); by default it spans the first third of the frame plus 3 mm below it,
    i.e. the annulus / LVOT landing zone. The rest of the frame deploys free.
    """
    workdir = Path(workdir); workdir.mkdir(parents=True, exist_ok=True)
    frame = lattice_hex_mesh(spec, n_along=n_along, n_thick=1)
    r_free = np.linalg.norm(frame.nodes[:, :2], axis=1)
    r_out_max = float(r_free.max())
    r0 = r_out_max + gap_mm
    H = spec.height_mm
    if vessel_z is None:
        vessel_z = (-3.0, max(H / 3.0, 8.0))
    crimp = crimp_radius_mm if crimp_radius_mm is not None else vessel_radius_mm - 1.5
    vessel = cylinder_hex_mesh(vessel_radius_mm, vessel_thickness_mm, vessel_z[0], vessel_z[1],
                               n_theta=n_theta, n_z=n_z_vessel, n_r=n_r_vessel)
    n_z_sleeve = max(4, int(round((H + 2.0) / 2.5)))
    sleeve = cylinder_hex_mesh(r0, 0.5, -1.0, H + 1.0, n_theta=n_theta, n_z=n_z_sleeve, n_r=1)
    p = params or DeploymentParams()
    p.sleeve_r0, p.sleeve_r_crimp = r0, crimp
    p.sleeve_r_final = max(r_out_max, vessel_radius_mm + vessel_thickness_mm) + 2.0
    A = assemble(frame, vessel, sleeve)
    feb = write_deployment(workdir / "deploy.feb", A, p)
    run = run_febio(feb, threads=threads, timeout_s=timeout_s)

    pos = read_node_positions(workdir / "frame_nodes.txt") if (workdir / "frame_nodes.txt").exists() else {}
    times = np.array(sorted(pos))
    mean_r = np.array([np.linalg.norm(pos[t][:, 1:3], axis=1).mean() for t in times]) if len(times) else np.zeros(0)
    z_ref = frame.nodes[:, 2]
    edges = np.linspace(z_ref.min(), z_ref.max() + 1e-9, n_bands + 1)
    band = np.clip(np.searchsorted(edges, z_ref, side="right") - 1, 0, n_bands - 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    free = np.array([r_free[band == b].mean() for b in range(n_bands)])
    if len(times):
        last = pos[times[-1]]
        r_dep = np.linalg.norm(last[:, 1:3], axis=1)
        dep = np.array([r_dep[band == b].mean() for b in range(n_bands)])
        tail = mean_r[times >= times[-1] - 0.2] if times[-1] >= 1.8 else mean_r[-1:]
        drift = float(np.ptp(tail))
    else:
        dep, drift = np.full(n_bands, np.nan), float("nan")
    return DeploymentResult(run["normal"], run["wall_s"], len(A.nodes), int(sum(len(e) for e in A.parts.values())),
                            times, mean_r, centres, free, dep, drift, workdir,
                            {"vessel_radius_mm": vessel_radius_mm, "vessel_z": vessel_z, "crimp_radius_mm": crimp,
                             "frame_E_MPa": p.frame_E_MPa, "vessel_E_MPa": p.vessel_E_MPa,
                             "contact_penalty": p.contact_penalty, "vessel_penalty": p.vessel_penalty})
