"""End to end with physics: CT -> geometry holon -> patient wall -> FE rungs -> delegated decision.

Uses the test phantom (cylindrical lumen with a calcium nodule) so that it runs without
patient data. Needs FEBio (see README). Run from the repository root:

    python -m examples.fe_patient [workdir]
"""
import json
import sys
from dataclasses import replace

import numpy as np

from tests.test_geometry import phantom, landmarks, Z_ANN
from tavr_decide import Uncertain, Utility, evolut_like_grammar, evaluate, holarchic_select
from tavr_decide.geometry import build_anatomy, fit_plane
from tavr_decide.fe.vessel import vessel_from_lumen
from tavr_decide.fe.rungs import FEEngine

workdir = sys.argv[1] if len(sys.argv) > 1 else "runs/fe_patient"

# 1. geometry holon: CT + lumen + landmarks -> anatomy with uncertainty
ct, lumen, _, (cx, cy) = phantom()
lm = landmarks(cx, cy)
anat, prov = build_anatomy(ct, lumen, lm, label="phantom")
anat = replace(anat, observed_depth_mm=Uncertain(5.0, 0.8))      # pre-release cine: 5 mm below the NCC
print(f"anatomy: annulus {anat.annulus_diameter_mm.mean:.1f} mm, MS {anat.ms_length_mm.mean:.1f} mm, "
      f"upper-LVOT calcium {anat.upper_lvot_calcium_mm3.mean:.0f} mm3, observed depth {anat.observed_depth_mm.mean:.1f} mm")

# 2. patient-specific landing zone (local frame: 0 degrees along +x of the scan)
plane = fit_plane(np.array(lm.nadirs), toward=np.array([cx, cy, Z_ANN + 20])).with_x_toward(np.array([cx + 50, cy, Z_ANN]))
vessel = vessel_from_lumen(lumen, plane, z_range=(-12, 10), n_theta=48, n_z=11, n_r=2, ct=ct)
print(f"wall mesh: {len(vessel.mesh.elems)} hexes, {int(vessel.calcified.sum())} calcified")

# 3. holons whose high rung is a finite-element deployment
engine = FEEngine(anat, vessel, workdir, fe_kwargs=dict(n_along=2, timeout_s=3000))
holons = engine.holons()
grammar = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
utility = Utility()

base = evaluate(anat, grammar, "assess", holons, utility, n=3000, seed=0)
print(f"\nproxies only     : recommend {base.best.key}  stability {base.pea:.3f}  margin {base.margin:.3f}")

res, trace = holarchic_select(anat, grammar, "assess", holons, utility, eta=0.95, n=3000, seed=0)
print(f"delegated        : recommend {res.best.key}  stability {res.pea:.3f}  FE solves so far: {len(engine.runs)}")
for t in trace:
    print("  round", t["round"], t.get("reason") or {k: round(v, 3) for k, v in t["contracts"].items()},
          "| rungs:", {o: n["rung"] for o, n in t["narratives"].items()})

# 4. open the physics regardless, to show what the rung returns
for h in holons.values():
    h.active = 1
forced = evaluate(anat, grammar, "assess", holons, utility, n=3000, seed=0)
print(f"FE rungs open    : recommend {forced.best.key}  stability {forced.pea:.3f}  margin {forced.margin:.3f}")
print("\nfinite-element solves:")
for r in engine.runs:
    print(" ", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k != "workdir"}))
print(f"mean wall time per action: {engine.measured_cost_s:.0f} s")
