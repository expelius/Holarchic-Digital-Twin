"""CFD of the paravalvular channel of the real-case deployment, compared with the 0D rung."""
import sys, numpy as np
sys.path.insert(0, ".")
from runs.rebuild_result import rebuild
from tavr_decide.cfd.channel import gap_map, channel_mesh
from tavr_decide.cfd.lumped import pvl_lumped
from tavr_decide.cfd.svmp import pvl_cfd, CFDParams
out = sys.argv[1]
res = rebuild("0091", 29, 4.0, "runs/real_0091_s29_d4b")
g = gap_map(res, n_theta=72, n_z=24)
m = channel_mesh(g, n_h=6)
h0 = pvl_lumped(g)
c = pvl_cfd(m, out, CFDParams(dp_mmhg=60.0, dt_s=2e-4, n_steps=200, save_every=100), nproc=6, timeout_s=7000)
print(f"elements {len(m.elems)}  wall {c.wall_s:.0f} s  steady change over last 10% {c.steady_rel_change:.3f}")
print(f"0D : Q {h0.flow_ml_s:.2f} mL/s  RVol {h0.rvol_ml:.2f} mL  grade {h0.grade}")
print(f"CFD: Q {c.flow_ml_s:.2f} mL/s  RVol {c.rvol_ml:.2f} mL  grade {c.grade}")
print("flux history (every 40 steps):", np.round(c.flux_history[::20], 3))
