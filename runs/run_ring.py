"""Ring deployment experiments: python runs/run_ring.py OUT [key=value ...]
keys: cells rows along ntheta nz E penalty steps crimp gap laugon"""
import sys
import numpy as np
from tavr_decide.frame import FrameSpec
from tavr_decide.fe.lattice import lattice_hex_mesh, cylinder_hex_mesh
from tavr_decide.fe.febio import DeploymentParams, assemble, write_deployment, run_febio, read_node_positions

out = sys.argv[1]
kw = dict(cells=12, rows=2, along=4, ntheta=48, nz=6, E=5000.0, penalty=0.01, vpenalty=0.001, nr=2, Ev=1.0, steps=20, crimp=9.5, gap=0.3, timeout=3600)
for a in sys.argv[2:]:
    k, v = a.split("="); kw[k] = type(kw[k])(float(v)) if not isinstance(kw[k], float) else float(v)
spec = FrameSpec("ring", "self-expanding", kw["cells"], kw["rows"], 5.0 * kw["rows"], [(0, 13.0), (1, 13.0)], 0.30, 0.25)
frame = lattice_hex_mesh(spec, n_along=kw["along"], n_thick=1)
H = spec.height_mm
r_out = 13.0 + 0.125
r0 = r_out + kw["gap"]
vessel = cylinder_hex_mesh(11.0, 2.0, -3.0, H + 3.0, n_theta=kw["ntheta"], n_z=kw["nz"] + 2, n_r=kw["nr"])
sleeve = cylinder_hex_mesh(r0, 0.5, -1.0, H + 1.0, n_theta=kw["ntheta"], n_z=kw["nz"], n_r=1)
p = DeploymentParams(frame_E_MPa=kw["E"], sleeve_r0=r0, sleeve_r_crimp=kw["crimp"], sleeve_r_final=15.0,
                     steps_per_stage=kw["steps"], contact_penalty=kw["penalty"], vessel_penalty=kw["vpenalty"], vessel_E_MPa=kw["Ev"])
A = assemble(frame, vessel, sleeve)
feb = write_deployment(f"{out}/deploy.feb", A, p)
print("nodes", len(A.nodes), "elems", sum(len(e) for e in A.parts.values()), kw)
r = run_febio(feb, timeout_s=kw["timeout"])
print("normal termination:", r["normal"], "| wall s:", round(r["wall_s"], 1))
try:
    pos = read_node_positions(f"{out}/frame_nodes.txt")
    ts = sorted(pos)
    for t in ts[::max(1, len(ts) // 10)] + [ts[-1]]:
        rad = np.linalg.norm(pos[t][:, 1:3], axis=1)
        print(f"t={t:6.3f}  frame radius mean {rad.mean():6.3f}  min {rad.min():6.3f}  max {rad.max():6.3f}")
except Exception as e:
    print("no node data:", e)
