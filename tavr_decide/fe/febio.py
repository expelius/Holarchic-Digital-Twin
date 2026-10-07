"""FEBio 4 input writer and runner for the deployment holon's high-fidelity rung.

Model (units mm, MPa, quasi-static, frictionless):

* **frame** – conforming hex lattice in its stress-free expanded shape;
* **vessel** – thick-walled elastic cylinder (annulus / LVOT segment) smaller than the frame;
* **sleeve** – a crimping sleeve whose nodes are all displacement-controlled.

Step 1 (t 0→1): the sleeve shrinks and crimps the frame by contact; the vessel is not in
contact with anything. Step 2 (t 1→2): the sleeve expands past the vessel; the frame
follows it elastically until it meets the vessel, contact frame–vessel takes over and the
sleeve walks away. Everything is displacement-controlled, which is what makes the release
solvable by an implicit static solver.

Declared simplifications (the report must carry them): nitinol is a neo-Hookean solid
with a placeholder effective modulus (no superelastic plateau, no hysteresis); the vessel
is homogeneous and has no calcium; the crimp stops just below the vessel radius rather
than at catheter size (the final equilibrium of a path-independent elastic model does not
depend on it).
"""
from __future__ import annotations

import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .lattice import HexMesh


@dataclass
class DeploymentParams:
    frame_E_MPa: float = 5000.0        # PLACEHOLDER effective modulus (see module docstring)
    frame_nu: float = 0.30
    vessel_E_MPa: float = 1.0
    vessel_nu: float = 0.45
    calcium_E_MPa: float = 20.0        # PLACEHOLDER: calcified tissue, literature spans ~10-60 MPa
    calcium_nu: float = 0.30
    sleeve_r0: float = 13.2            # initial inner radius of the sleeve (just outside the frame)
    sleeve_r_crimp: float = 9.5        # inner radius at the end of step 1
    sleeve_r_final: float = 15.0       # inner radius at the end of step 2 (beyond the vessel)
    sleeve_scale_crimp: float | None = None   # uniform in-plane scaling of the sleeve at the end of step 1
    sleeve_scale_final: float | None = None   # ... and at the end of step 2 (None: derived from the radii above)
    steps_per_stage: int = 20
    contact_penalty: float = 0.01       # sleeve-frame; 1.0 diverges at first touch (stiff tiny frame elements)
    vessel_penalty: float = 0.001       # frame-vessel; auto-penalty is scaled by the stiff frame, the wall is soft
    search_radius: float = 1.0


@dataclass
class Assembly:
    nodes: np.ndarray
    parts: dict[str, np.ndarray]            # name -> (E, 8) global zero-based
    node_ranges: dict[str, tuple[int, int]]
    surfaces: dict[str, np.ndarray]         # name -> (F, 4) global zero-based
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)


def assemble(frame: HexMesh, vessel: HexMesh, sleeve: HexMesh, calcified: np.ndarray | None = None) -> Assembly:
    """Concatenate the three meshes. ``calcified`` (bool per vessel element) splits the wall
    into a soft part and a calcium part that share nodes."""
    nodes, parts, ranges, surfaces = [], {}, {}, {}
    off = 0
    for name, m in (("frame", frame), ("vessel", vessel), ("sleeve", sleeve)):
        nodes.append(m.nodes)
        E = m.elems + off
        if name == "vessel" and calcified is not None and calcified.any():
            if (~calcified).any():
                parts["vessel"] = E[~calcified]
            parts["calcium"] = E[calcified]
        else:
            parts[name] = E
        ranges[name] = (off, off + len(m.nodes))
        surfaces[f"{name}_outer"] = m.outer_faces + off
        surfaces[f"{name}_inner"] = m.inner_faces + off
        off += len(m.nodes)
    A = Assembly(np.vstack(nodes), parts, ranges, surfaces)
    X = A.nodes

    def at_angle(idx: np.ndarray, deg: float, tol_deg: float = 1.0) -> np.ndarray:
        th = np.degrees(np.arctan2(X[idx, 1], X[idx, 0])) % 360
        d = np.abs((th - deg + 180) % 360 - 180)
        return idx[d < tol_deg]

    f0, f1 = ranges["frame"]
    jc = frame.junction_center_nodes[0] + f0            # inflow ring of junction centres
    A.node_sets["frame_a0"] = at_angle(jc, 0.0)
    A.node_sets["frame_a90"] = at_angle(jc, 90.0)
    A.node_sets["frame_a180"] = at_angle(jc, 180.0)
    for k in ("frame_a0", "frame_a90", "frame_a180"):
        if len(A.node_sets[k]) == 0:
            raise ValueError("frame needs junction centres at 0/90/180 degrees: use n_cells_circ divisible by 4")
    # The delivery system also holds the far end against rotation: restrain the last even
    # (non-offset) junction ring in-plane at 0/180 (y) and 90 (x), leaving radial motion free.
    # This removes the torsional bifurcation of the lattice under deep crimp.
    top = len(frame.junction_center_nodes) - 1
    top = top if top % 2 == 0 else top - 1
    jt = frame.junction_center_nodes[top] + f0
    A.node_sets["frame_top_a0"] = np.concatenate([at_angle(jt, 0.0), at_angle(jt, 180.0)])
    A.node_sets["frame_top_a90"] = at_angle(jt, 90.0)
    v0, v1 = ranges["vessel"]
    vid = np.arange(v0, v1)
    z = X[vid, 2]
    A.node_sets["vessel_ends"] = vid[(np.isclose(z, z.min())) | (np.isclose(z, z.max()))]
    # Rigid-body restraint by structured index (works for non-circular, off-axis sections):
    # inner-surface columns at 0 and 180 degrees hold y, columns at 90 and 270 hold x.
    inner = vessel.junction_center_nodes + v0            # (n_z+1, n_theta)
    nt = inner.shape[1]
    if nt % 4:
        raise ValueError("vessel n_theta must be divisible by 4")
    A.node_sets["vessel_a0"] = np.concatenate([inner[:, 0], inner[:, nt // 2]])
    A.node_sets["vessel_a90"] = np.concatenate([inner[:, nt // 4], inner[:, 3 * nt // 4]])
    A.node_sets["vessel_all"] = vid
    s0, s1 = ranges["sleeve"]
    A.node_sets["sleeve_all"] = np.arange(s0, s1)
    A.node_sets["frame_all"] = np.arange(f0, f1)
    return A


def _fmt_ids(ids) -> str:
    return ",".join(str(int(i) + 1) for i in ids)


def write_deployment(path: str | Path, A: Assembly, p: DeploymentParams) -> Path:
    X = A.nodes
    L: list[str] = ['<?xml version="1.0" encoding="ISO-8859-1"?>', '<febio_spec version="4.0">',
                    '  <Module type="solid"/>', "  <Material>"]
    all_mats = {"frame": (p.frame_E_MPa, p.frame_nu), "vessel": (p.vessel_E_MPa, p.vessel_nu),
                "calcium": (p.calcium_E_MPa, p.calcium_nu), "sleeve": (1000.0, 0.3)}
    mats = {k: all_mats[k] for k in A.parts}
    for i, (name, (E, nu)) in enumerate(mats.items(), 1):
        L.append(f'    <material id="{i}" name="{name}_mat" type="neo-Hookean"><density>1</density><E>{E}</E><v>{nu}</v></material>')
    L += ["  </Material>", "  <Mesh>", '    <Nodes name="all">']
    L += [f'      <node id="{i + 1}">{x:.6f},{y:.6f},{z:.6f}</node>' for i, (x, y, z) in enumerate(X)]
    L.append("    </Nodes>")
    eid = 1
    for name, E in A.parts.items():
        L.append(f'    <Elements type="hex8" name="{name}">')
        for e in E:
            L.append(f'      <elem id="{eid}">{_fmt_ids(e)}</elem>'); eid += 1
        L.append("    </Elements>")
    for name, ids in A.node_sets.items():
        L.append(f'    <NodeSet name="{name}">{_fmt_ids(ids)}</NodeSet>')
    for name in ("frame_outer", "vessel_inner", "sleeve_inner"):
        L.append(f'    <Surface name="{name}">')
        L += [f'      <quad4 id="{k + 1}">{_fmt_ids(f)}</quad4>' for k, f in enumerate(A.surfaces[name])]
        L.append("    </Surface>")
    L.append('    <SurfacePair name="frame_sleeve"><primary>frame_outer</primary><secondary>sleeve_inner</secondary></SurfacePair>')
    L.append('    <SurfacePair name="frame_vessel"><primary>frame_outer</primary><secondary>vessel_inner</secondary></SurfacePair>')
    L += ["  </Mesh>", "  <MeshDomains>"]
    for name in A.parts:
        L.append(f'    <SolidDomain name="{name}" mat="{name}_mat"/>')
    L += ["  </MeshDomains>", "  <MeshData>"]
    sl = A.node_sets["sleeve_all"]
    # displacement = (s(t) - 1) * (x, y): uniform in-plane scaling about the axis, so a sleeve
    # that follows the frame's profile compresses every level by the same ratio
    for comp, vals in (("x", X[sl, 0]), ("y", X[sl, 1])):
        L.append(f'    <NodeData name="sleeve_u{comp}" node_set="sleeve_all">')
        L += [f'      <node lid="{k + 1}">{v:.8f}</node>' for k, v in enumerate(vals)]
        L.append("    </NodeData>")
    L += ["  </MeshData>", "  <Boundary>"]

    def zero(name: str, ns: str, dofs: str) -> None:
        d = "".join(f"<{c}_dof>{1 if c in dofs else 0}</{c}_dof>" for c in "xyz")
        L.append(f'    <bc name="{name}" node_set="{ns}" type="zero displacement">{d}</bc>')

    zero("frame_a0", "frame_a0", "yz"); zero("frame_a90", "frame_a90", "xz"); zero("frame_a180", "frame_a180", "y")
    if len(A.node_sets.get("frame_top_a0", [])) and len(A.node_sets.get("frame_top_a90", [])):
        zero("frame_top_a0", "frame_top_a0", "y"); zero("frame_top_a90", "frame_top_a90", "x")
    zero("vessel_ends", "vessel_ends", "z"); zero("vessel_a0", "vessel_a0", "y"); zero("vessel_a90", "vessel_a90", "x")
    zero("sleeve_z", "sleeve_all", "z")
    for comp in "xy":
        L.append(f'    <bc name="sleeve_{comp}" node_set="sleeve_all" type="prescribed displacement">'
                 f'<dof>{comp}</dof><value lc="1" type="map">sleeve_u{comp}</value><relative>0</relative></bc>')
    L.append("  </Boundary>")

    def contact(name: str, pair: str, penalty: float | None = None) -> list[str]:
        pen = p.contact_penalty if penalty is None else penalty
        return [f'    <contact name="{name}" surface_pair="{pair}" type="sliding-elastic">',
                f"      <laugon>0</laugon><tolerance>0.2</tolerance><penalty>{pen}</penalty>",
                "      <auto_penalty>1</auto_penalty><two_pass>0</two_pass><search_tol>0.01</search_tol>",
                f"      <search_radius>{p.search_radius}</search_radius><symmetric_stiffness>1</symmetric_stiffness>",
                "      <fric_coeff>0</fric_coeff>", "    </contact>"]

    L += ["  <Contact>"] + contact("sleeve_contact", "frame_sleeve") + ["  </Contact>"]

    def control() -> list[str]:
        dt = 1.0 / p.steps_per_stage
        return ["      <Control>", "        <analysis>STATIC</analysis>",
                f"        <time_steps>{p.steps_per_stage}</time_steps><step_size>{dt}</step_size>",
                '        <solver type="solid"><symmetric_stiffness>symmetric</symmetric_stiffness>'
                "<max_refs>25</max_refs><dtol>0.001</dtol><etol>0.01</etol><rtol>0</rtol><lstol>0.9</lstol>"
                "<min_residual>1e-8</min_residual>"
                '<qn_method type="BFGS"><max_ups>10</max_ups></qn_method></solver>',
                f'        <time_stepper type="default"><dtmin>{dt / 200}</dtmin><dtmax>{dt}</dtmax>'
                "<max_retries>12</max_retries><opt_iter>12</opt_iter><aggressiveness>1</aggressiveness></time_stepper>",
                "      </Control>"]

    sc = p.sleeve_scale_crimp if p.sleeve_scale_crimp is not None else p.sleeve_r_crimp / p.sleeve_r0
    sf = p.sleeve_scale_final if p.sleeve_scale_final is not None else p.sleeve_r_final / p.sleeve_r0
    L += ["  <Step>", '    <step id="1" name="crimp">'] + control() + ["    </step>",
          '    <step id="2" name="release">'] + control() + ["      <Contact>"]
    L += ["  " + s for s in contact("vessel_contact", "frame_vessel", p.vessel_penalty)]
    L += ["      </Contact>", "    </step>", "  </Step>", "  <LoadData>",
          '    <load_controller id="1" name="sleeve_radius" type="loadcurve"><interpolate>LINEAR</interpolate>'
          "<extend>CONSTANT</extend><points>"
          f"<pt>0,0</pt><pt>1,{sc - 1.0}</pt><pt>2,{sf - 1.0}</pt>"
          "</points></load_controller>", "  </LoadData>", "  <Output>",
          '    <plotfile type="febio"><var type="displacement"/><var type="stress"/><var type="contact pressure"/></plotfile>',
          '    <logfile><node_data data="x;y;z" file="frame_nodes.txt" node_set="frame_all"/>'
          '<node_data data="x;y;z" file="vessel_nodes.txt" node_set="vessel_all"/></logfile>',
          "  </Output>", "</febio_spec>"]
    path = Path(path)
    path.write_text("\n".join(L), encoding="ISO-8859-1")
    return path


# --- running ---------------------------------------------------------------------------------
FEBIO_WSL = "/opt/src/FEBio/build/bin/febio4"
MKL_LIBS = os.environ.get("TAVR_MKL_LIBS", "/opt/intel/oneapi/mkl/2024.2/lib:/opt/intel/oneapi/compiler/2024.2/lib")
WSL_DISTRO = os.environ.get("TAVR_WSL_DISTRO", "Ubuntu-24.04")


def _to_wsl(p: Path) -> str:
    s = str(p.resolve()).replace("\\", "/")
    m = re.match(r"^([A-Za-z]):/(.*)$", s)
    return f"/mnt/{m.group(1).lower()}/{m.group(2)}" if m else s


def run_febio(feb: str | Path, threads: int = 4, timeout_s: int = 7200, distro: str = WSL_DISTRO) -> dict:
    """Run FEBio on ``feb``. On Windows it goes through WSL; on Linux it calls ``febio4``
    from PATH (or ``FEBIO`` env var). Returns termination status, wall time and log tail."""
    feb = Path(feb)
    t0 = time.time()
    if os.name == "nt":
        cmd = ["wsl", "-d", distro, "-u", "root", "--", "bash", "-c",
               f"cd '{_to_wsl(feb.parent)}' && export LD_LIBRARY_PATH={MKL_LIBS} && OMP_NUM_THREADS={threads} {FEBIO_WSL} -i '{feb.name}' -silent"]
    else:
        cmd = [os.environ.get("FEBIO", "febio4"), "-i", feb.name, "-silent"]
    proc = subprocess.run(cmd, cwd=None if os.name == "nt" else feb.parent, capture_output=True, timeout=timeout_s)
    wall = time.time() - t0
    log = feb.with_suffix(".log")
    txt = log.read_text(encoding="latin-1", errors="replace") if log.exists() else ""
    return {"normal": "N O R M A L   T E R M I N A T I O N" in txt, "wall_s": wall, "returncode": proc.returncode,
            "tail": "\n".join(txt.splitlines()[-25:]), "stdout": proc.stdout.decode(errors="replace")[-800:]}


def read_node_positions(path: str | Path) -> dict[float, np.ndarray]:
    """Parse a FEBio node_data log file into {time: (n, 4) [id, x, y, z]}."""
    out: dict[float, list] = {}
    t = None
    for line in Path(path).read_text(encoding="latin-1").splitlines():
        s = line.strip()
        if s.startswith("*Time"):
            t = float(s.split("=")[1]); out[t] = []
        elif s and not s.startswith("*") and t is not None:
            out[t].append([float(v) for v in s.replace(",", " ").split()])
    return {k: np.array(v) for k, v in out.items() if v}
