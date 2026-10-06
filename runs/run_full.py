"""Full parametric self-expanding frame (26 mm, 12 cells x 5 rows) into a 22 mm landing zone."""
import sys
from tavr_decide.frame import self_expanding_frame
from tavr_decide.fe.deploy import run_deployment
out = sys.argv[1]
along = int(sys.argv[2]) if len(sys.argv) > 2 else 2
spec = self_expanding_frame(26, n_cells_circ=12, n_rows=5).spec
res = run_deployment(spec, vessel_radius_mm=11.0, workdir=out, n_along=along, n_theta=48, n_z_vessel=6, n_r_vessel=2,
                     n_bands=6, timeout_s=6000)
print(res.summary()); print(res.params)
