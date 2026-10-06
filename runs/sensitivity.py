"""Mesh and contact-penalty sensitivity of the full-frame deployment. One process, modules imported once."""
import csv, sys, time
from tavr_decide.frame import self_expanding_frame
from tavr_decide.fe.deploy import run_deployment
from tavr_decide.fe.febio import DeploymentParams

spec = self_expanding_frame(26, n_cells_circ=12, n_rows=5).spec
cases = [  # name, n_along, n_theta, n_r, sleeve_penalty, vessel_penalty
    ("baseline",      2, 48, 2, 0.01, 1e-3),
    ("along3",        3, 48, 2, 0.01, 1e-3),
    ("along4",        4, 48, 2, 0.01, 1e-3),
    ("vp1e-2",        2, 48, 2, 0.01, 1e-2),
    ("vp1e-4",        2, 48, 2, 0.01, 1e-4),
    ("ntheta72",      2, 72, 2, 0.01, 1e-3),
    ("nr3",           2, 48, 3, 0.01, 1e-3),
    ("sp0.003",       2, 48, 2, 0.003, 1e-3),
]
out = sys.argv[1]
with open(f"{out}/sensitivity.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["case", "n_along", "n_theta", "n_r", "sleeve_pen", "vessel_pen", "normal", "wall_s",
                                   "nodes", "exp_inflow", "exp_band2", "exp_waist", "exp_outflow", "r_inflow", "drift"]); f.flush()
    for name, na, nt, nr, sp, vp in cases:
        p = DeploymentParams(contact_penalty=sp, vessel_penalty=vp)
        try:
            r = run_deployment(spec, 11.0, f"{out}/{name}", n_along=na, n_theta=nt, n_r_vessel=nr, params=p, timeout_s=5400)
            b = r.band_radius_deployed; e = r.expansion_ratio
            w.writerow([name, na, nt, nr, sp, vp, r.normal_termination, round(r.wall_s), r.n_nodes,
                        round(e[0], 4), round(e[1], 4), round(e[2], 4), round(e[5], 4), round(b[0], 4), round(r.equilibrium_drift_mm, 5)])
        except Exception as e:
            w.writerow([name, na, nt, nr, sp, vp, f"EXC {type(e).__name__}", "", "", "", "", "", "", ""])
        f.flush(); print(name, "done", flush=True)
