import sys, pickle, numpy as np
from tavr_decide.frame import self_expanding_frame
from tavr_decide.fe.vessel import vessel_from_lumen
from tavr_decide.fe.deploy import run_deployment
from tavr_decide.fe.post import sealing_gap, wall_displacement
cid, size, depth, out = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
along = int(sys.argv[5]) if len(sys.argv) > 5 else 2
case = pickle.load(open(f"data/tavrp_pl/case_ct{cid}.pkl", "rb"))
import nibabel as nib
from tavr_decide.geometry import Volume
img = nib.load(f"data/tavrp_pl/taviHeartData/imagesTs/ct{cid}.nii.gz"); lab = nib.load(f"data/tavrp_pl/taviHeartData/labelsTs/label{cid}.nii.gz")
ct = Volume(np.asarray(img.dataobj, dtype=np.float32), np.asarray(img.affine, dtype=float)); L = np.asarray(lab.dataobj).astype(np.int16)
lumen = Volume(np.isin(L, [1, 2, 3, 4, 5]).astype(np.float32), ct.affine)
pv = vessel_from_lumen(lumen, case["plane"], z_range=(-8, 8), n_theta=48, n_z=8, n_r=2, ct=ct)
spec = self_expanding_frame(size, n_cells_circ=12, n_rows=5).spec
from tavr_decide.fe.febio import DeploymentParams
res = run_deployment(spec, None, out, vessel=pv, inflow_z=-depth, n_along=along, crimp_margin_mm=0.5,
                     params=DeploymentParams(steps_per_stage=40), timeout_s=5400)
print(res.summary()); print({k: v for k, v in res.params.items() if k in ("landing_radius_mm", "crimp_radius_mm")})
if res.normal_termination and res.vessel_final is not None:
    g = sealing_gap(res, z_band=(-3, 1)); w = wall_displacement(res, z_max=-case["anatomy"].ms_length_mm.mean)
    print("sealing gap area %.2f mm2, max gap %.2f mm" % (g["area_mm2"], g["max_gap_mm"]))
    print("gap by angle (deg: mm):", {int(i * 5): round(float(v), 2) for i, v in enumerate(g["gap_by_angle_mm"]) if v > 0.05})
    print("wall displacement below MS (z <= %.1f): p90 %.2f mm, max %.2f mm" % (-case["anatomy"].ms_length_mm.mean, w["p90_mm"], w["max_mm"]))
    rf = np.linalg.norm(res.frame_final[:, :2], axis=1); zf = res.frame_final[:, 2]
    th = np.degrees(np.arctan2(res.frame_final[:, 1], res.frame_final[:, 0]))
    sel = (zf > -4.5) & (zf < -2.5)
    print("inflow ring deployed radius by angle:", {int(a): round(float(rf[sel][(np.abs(((th[sel] - a + 180) % 360) - 180) < 15)].mean()), 1) for a in range(0, 360, 30)})
