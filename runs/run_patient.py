"""Deploy into the phantom's patient-specific wall (with its calcium nodule) at a given depth."""
import sys
import numpy as np
from tests.test_geometry import phantom, landmarks, Z_ANN
from tavr_decide.geometry import fit_plane
from tavr_decide.frame import self_expanding_frame
from tavr_decide.fe.vessel import vessel_from_lumen
from tavr_decide.fe.deploy import run_deployment
from tavr_decide.fe.post import sealing_gap, wall_displacement

out, depth = sys.argv[1], float(sys.argv[2])
ct, lumen, _, (cx, cy) = phantom()
lm = landmarks(cx, cy)
plane = fit_plane(np.array(lm.nadirs), toward=np.array([cx, cy, Z_ANN + 20])).with_x_toward(np.array([cx + 50, cy, Z_ANN]))
pv = vessel_from_lumen(lumen, plane, z_range=(-12, 10), n_theta=48, n_z=11, n_r=2, ct=ct)
spec = self_expanding_frame(26, n_cells_circ=12, n_rows=5).spec
res = run_deployment(spec, None, out, vessel=pv, inflow_z=-depth, n_along=2, timeout_s=3000)
print(res.summary()); print(res.params)
if res.frame_final is not None and res.vessel_final is not None:
    g = sealing_gap(res, z_band=(-3, 1)); w = wall_displacement(res, z_max=-lm.ms_length_mm)
    print("sealing gap: area %.3f mm2, max %.3f mm" % (g["area_mm2"], g["max_gap_mm"]))
    print("gap by angle (deg -> mm):", {int(i * 5): round(float(v), 2) for i, v in enumerate(g["gap_by_angle_mm"]) if v > 0.02})
    print("wall displacement below MS (z <= %.1f): p90 %.3f mm, mean %.3f, max %.3f, n %d" % (-lm.ms_length_mm, w["p90_mm"], w["mean_mm"], w["max_mm"], w["n"]))
