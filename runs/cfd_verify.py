"""Verification of the CFD rung: annular gap of uniform height, low Reynolds number.
Analytic (narrow-gap Poiseuille): Q = 2 pi r_m h^3 dP / (12 mu L)."""
import sys, numpy as np
from tavr_decide.cfd.channel import GapMap, channel_mesh
from tavr_decide.cfd.lumped import pvl_lumped, MMHG, MU
from tavr_decide.cfd.svmp import pvl_cfd, CFDParams
out = sys.argv[1]
n_h = int(sys.argv[2]) if len(sys.argv) > 2 else 6
h, r, L, dp = 0.3, 12.0, 10.0, 0.5            # mm, mm, mm, mmHg
nt, nz = 48, 11
th = (np.arange(nt) + 0.5) * 2 * np.pi / nt; z = np.linspace(0, L, nz)
g = GapMap(th, z, np.full((nz, nt), r), np.full((nz, nt), r + h))
m = channel_mesh(g, n_h=n_h, h_seal_mm=None)
rm = (r + h / 2) / 10
q_an = 2 * np.pi * rm * (h / 10) ** 3 * dp * MMHG / (12 * MU * L / 10)
q_0d = pvl_lumped(g, dp_mmhg=dp).flow_ml_s
res = pvl_cfd(m, out, CFDParams(dp_mmhg=dp, dt_s=2e-3, n_steps=int(sys.argv[3]) if len(sys.argv) > 3 else 150, save_every=500), nproc=4, timeout_s=3600)
print(f"elements {len(m.elems)}  wall {res.wall_s:.0f} s  steady change {res.steady_rel_change:.2e}")
print(f"analytic {q_an:.5f} mL/s   0D {q_0d:.5f}   CFD {res.flow_ml_s:.5f}   CFD/analytic {res.flow_ml_s / q_an:.4f}   1-1/N^2 = {1 - 1 / n_h ** 2:.4f}")
