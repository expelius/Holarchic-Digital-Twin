"""svMultiPhysics input writer, runner and reader for the leak holon's CFD rung.

Unsteady incompressible Navier-Stokes (stabilised finite elements, generalised-alpha in
time) on the paravalvular channel, driven by the diastolic pressure difference between the
aortic and the ventricular faces, no slip on the skirt and the wall. The run is integrated
until the flux through the ventricular face is steady; that flux is the regurgitant flow.

Units are cgs (cm, g, s, dyn), the convention of the SimVascular tools.
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .channel import ChannelMesh
from .lumped import MMHG, MU, RHO, T_DIASTOLE_S, grade_of

SVMP_WSL = "/opt/src/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics"


# --- VTK XML writers (ASCII; what svMultiPhysics' VTK reader expects) ----------------------------
def _arr(name: str, dtype: str, values: np.ndarray, ncomp: int = 1) -> str:
    flat = " ".join(str(int(v)) if dtype.startswith("Int") else f"{v:.9g}" for v in np.asarray(values).ravel())
    nc = f' NumberOfComponents="{ncomp}"' if ncomp > 1 else ""
    return f'<DataArray type="{dtype}" Name="{name}"{nc} format="ascii">{flat}</DataArray>'


def write_vtu(path: Path, nodes: np.ndarray, hexes: np.ndarray) -> None:
    n, e = len(nodes), len(hexes)
    xml = ['<?xml version="1.0"?>', '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
           "<UnstructuredGrid>", f'<Piece NumberOfPoints="{n}" NumberOfCells="{e}">',
           "<PointData>", _arr("GlobalNodeID", "Int32", np.arange(1, n + 1)), "</PointData>",
           "<CellData>", _arr("GlobalElementID", "Int32", np.arange(1, e + 1)), "</CellData>",
           "<Points>", _arr("Points", "Float64", nodes, 3), "</Points>",
           "<Cells>", _arr("connectivity", "Int64", hexes), _arr("offsets", "Int64", 8 * np.arange(1, e + 1)),
           _arr("types", "UInt8", np.full(e, 12)), "</Cells>", "</Piece>", "</UnstructuredGrid>", "</VTKFile>"]
    path.write_text("\n".join(xml), encoding="ascii")


def write_vtp(path: Path, nodes: np.ndarray, quads: np.ndarray, owners: np.ndarray) -> None:
    gids = np.unique(quads)
    local = {g: i for i, g in enumerate(gids)}
    conn = np.vectorize(local.get)(quads)
    n, f = len(gids), len(quads)
    xml = ['<?xml version="1.0"?>', '<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">',
           "<PolyData>", f'<Piece NumberOfPoints="{n}" NumberOfVerts="0" NumberOfLines="0" NumberOfStrips="0" NumberOfPolys="{f}">',
           "<PointData>", _arr("GlobalNodeID", "Int32", gids + 1), "</PointData>",
           "<CellData>", _arr("GlobalElementID", "Int32", owners + 1), "</CellData>",
           "<Points>", _arr("Points", "Float64", nodes[gids], 3), "</Points>",
           "<Polys>", _arr("connectivity", "Int64", conn), _arr("offsets", "Int64", 4 * np.arange(1, f + 1)), "</Polys>",
           "</Piece>", "</PolyData>", "</VTKFile>"]
    path.write_text("\n".join(xml), encoding="ascii")


# --- solver input ----------------------------------------------------------------------------
@dataclass
class CFDParams:
    dp_mmhg: float = 60.0
    rho: float = RHO
    mu: float = MU
    dt_s: float = 1e-4
    n_steps: int = 300
    save_every: int = 100
    backflow: float = 0.2


def write_case(workdir: str | Path, m: ChannelMesh, p: CFDParams,
               inlet: str = "aortic", outlet: str = "ventricular") -> Path:
    wd = Path(workdir); (wd / "mesh" / "mesh-surfaces").mkdir(parents=True, exist_ok=True)
    write_vtu(wd / "mesh" / "mesh-complete.mesh.vtu", m.nodes, m.elems)
    for name, F in m.faces.items():
        write_vtp(wd / "mesh" / "mesh-surfaces" / f"{name}.vtp", m.nodes, F, m.face_elems[name])
    faces = "\n".join(f'  <Add_face name="{n}"><Face_file_path> mesh/mesh-surfaces/{n}.vtp </Face_file_path></Add_face>'
                      for n in m.faces)
    walls = "\n".join(f'   <Add_BC name="{n}"><Type> Dir </Type><Time_dependence> Steady </Time_dependence><Value> 0.0 </Value></Add_BC>'
                      for n in m.faces if n not in (inlet, outlet))
    xml = f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">
<GeneralSimulationParameters>
  <Continue_previous_simulation> false </Continue_previous_simulation>
  <Number_of_spatial_dimensions> 3 </Number_of_spatial_dimensions>
  <Number_of_time_steps> {p.n_steps} </Number_of_time_steps>
  <Time_step_size> {p.dt_s} </Time_step_size>
  <Spectral_radius_of_infinite_time_step> 0.50 </Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop> STOP_SIM </Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format> 1 </Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files> result </Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files> {p.save_every} </Increment_in_saving_VTK_files>
  <Start_saving_after_time_step> 1 </Start_saving_after_time_step>
  <Increment_in_saving_restart_files> {p.n_steps} </Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format> 0 </Convert_BIN_to_VTK_format>
  <Verbose> 1 </Verbose>
  <Warning> 0 </Warning>
  <Debug> 0 </Debug>
</GeneralSimulationParameters>
<Add_mesh name="msh">
  <Mesh_file_path> mesh/mesh-complete.mesh.vtu </Mesh_file_path>
{faces}
</Add_mesh>
<Add_equation type="fluid">
   <Coupled> true </Coupled>
   <Min_iterations> 3 </Min_iterations>
   <Max_iterations> 8 </Max_iterations>
   <Tolerance> 1e-6 </Tolerance>
   <Backflow_stabilization_coefficient> {p.backflow} </Backflow_stabilization_coefficient>
   <Density> {p.rho} </Density>
   <Viscosity model="Constant"><Value> {p.mu} </Value></Viscosity>
   <Output type="Spatial"><Velocity> true </Velocity><Pressure> true </Pressure></Output>
   <Output type="B_INT"><Pressure> true </Pressure><Velocity> true </Velocity></Output>
   <LS type="NS">
      <Linear_algebra type="fsils"><Preconditioner> fsils </Preconditioner></Linear_algebra>
      <Max_iterations> 15 </Max_iterations>
      <NS_GM_max_iterations> 10 </NS_GM_max_iterations>
      <NS_CG_max_iterations> 300 </NS_CG_max_iterations>
      <Tolerance> 1e-3 </Tolerance>
      <NS_GM_tolerance> 1e-3 </NS_GM_tolerance>
      <NS_CG_tolerance> 1e-3 </NS_CG_tolerance>
      <Absolute_tolerance> 1e-17 </Absolute_tolerance>
      <Krylov_space_dimension> 250 </Krylov_space_dimension>
   </LS>
   <Add_BC name="{inlet}"><Type> Neu </Type><Time_dependence> Steady </Time_dependence><Value> {p.dp_mmhg * MMHG:.6g} </Value></Add_BC>
   <Add_BC name="{outlet}"><Type> Neu </Type><Time_dependence> Steady </Time_dependence><Value> 0.0 </Value></Add_BC>
{walls}
</Add_equation>
</svMultiPhysicsFile>
"""
    f = wd / "solver.xml"
    f.write_text(xml, encoding="utf-8")
    return f


def _to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/mnt/{m.group(1).lower()}/{m.group(2)}" if m else s


def run_case(workdir: str | Path, nproc: int = 4, timeout_s: int = 7200, distro: str = "Ubuntu-24.04") -> dict:
    wd = Path(workdir)
    t0 = time.time()
    if os.name == "nt":
        cmd = ["wsl", "-d", distro, "-u", "root", "--", "bash", "-c",
               f"cd '{_to_wsl(wd)}' && mpirun --allow-run-as-root --oversubscribe -np {nproc} {SVMP_WSL} solver.xml > svmp.log 2>&1"]
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout_s)
    else:
        with open(wd / "svmp.log", "w") as fh:
            proc = subprocess.run(["mpirun", "-np", str(nproc), os.environ.get("SVMP", "svmultiphysics"), "solver.xml"],
                                  cwd=wd, stdout=fh, stderr=subprocess.STDOUT, timeout=timeout_s)
    return {"returncode": proc.returncode, "wall_s": time.time() - t0}


def read_face_flux(workdir: str | Path, face: str) -> np.ndarray:
    """Flux through ``face`` per time step from the boundary-integral output (cm^3/s).

    svMultiPhysics writes one file per quantity with one column per face; the file is
    located by pattern because its name carries the processor-count folder."""
    wd = Path(workdir)
    cands = sorted(wd.rglob("B_*Velocity_flux.txt"))
    if not cands:
        raise FileNotFoundError("no B_*Velocity_flux.txt found; did the run finish?")
    lines = [l.split() for l in cands[-1].read_text().splitlines() if l.strip()]
    header = lines[0]
    if face not in header:
        raise KeyError(f"face '{face}' not in {header}")
    j = header.index(face)
    return np.array([float(l[j]) for l in lines[1:]])


@dataclass
class PVLCFD:
    flow_ml_s: float
    rvol_ml: float
    grade: str
    flux_history: np.ndarray
    steady_rel_change: float
    wall_s: float
    model: str = "3D Navier-Stokes (svMultiPhysics)"


def pvl_cfd(m: ChannelMesh, workdir: str | Path, p: CFDParams | None = None, nproc: int = 4,
            t_diastole_s: float = T_DIASTOLE_S, timeout_s: int = 7200) -> PVLCFD:
    p = p or CFDParams()
    write_case(workdir, m, p)
    r = run_case(workdir, nproc=nproc, timeout_s=timeout_s)
    q = np.abs(read_face_flux(workdir, "ventricular"))
    tail = q[-max(5, len(q) // 10):]
    Q = float(tail.mean())
    rel = float(np.ptp(tail) / max(Q, 1e-12))
    rvol = Q * t_diastole_s
    return PVLCFD(Q, rvol, grade_of(rvol), q, rel, r["wall_s"])
