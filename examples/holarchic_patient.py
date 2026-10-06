"""Delegated (holarchic) selection on a synthetic patient.

The orchestrator does not pick fidelities. It derives one tolerance contract per
outcome from the decision margin and hands it down; each holon meets it on its own,
recursively through its parts, and narrates what it did.

Run:  python examples/holarchic_patient.py
"""
import json

from tavr_decide import (Anatomy, Uncertain, evolut_like_grammar, ConductionProxy, PVLProxy,
                         Utility, Holon, Rung, leaf, evaluate, holarchic_select, render_markdown)

CREAON = ("ms_length_mm", "annulus_diameter_mm", "upper_lvot_calcium_mm3")

# Conduction as a composite holon: frame mechanics + tissue response are its parts.
# (Stand-ins share the proxy physics with smaller declared error; in the benchmark the
# high rungs are FE callables.)
frame = leaf("frame mechanics", "conduction", CREAON, [
    ("statistical frame", "low", 0.001, 0.06, ConductionProxy().predict),
    ("FE frame (stand-in)", "high", 2 * 3600, 0.01, ConductionProxy().predict),
])
tissue = leaf("septal tissue", "conduction", CREAON, [
    ("population stiffness", "low", 0.001, 0.05, ConductionProxy().predict),
    ("calibrated stiffness (stand-in)", "high", 1 * 3600, 0.01, ConductionProxy().predict),
])
conduction = Holon("conduction", "conduction", CREAON, [Rung("assembly", "composite", 1.0, 0.0)],
                   parts=[frame, tissue], combine=lambda rs: 0.5 * (rs[0] + rs[1]))

pvl = leaf("paravalvular leak", "pvl", CREAON, [
    ("upper-LVOT calcium proxy", "low", 0.001, 0.10, PVLProxy().predict),
    ("FE frame-annulus gap (stand-in)", "mid", 2 * 3600, 0.03, PVLProxy().predict),
    ("CFD leak volume (stand-in)", "high", 6 * 3600, 0.01, PVLProxy().predict),
])

holons = {"conduction": conduction, "pvl": pvl}
grammar = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0, 4.0))
utility = Utility()

for label, pat in {
    "easy (deep valve, short septum)": Anatomy(Uncertain(22.4, 0.6), Uncertain(3.0, 1.0), Uncertain(15.0, 6.0),
                                                observed_depth_mm=Uncertain(7.5, 0.8), label="easy"),
    "boundary (valve near target)": Anatomy(Uncertain(22.4, 0.6), Uncertain(4.0, 1.0), Uncertain(26.0, 6.0),
                                            observed_depth_mm=Uncertain(4.0, 0.8), label="boundary"),
}.items():
    base = evaluate(pat, grammar, "assess", holons, utility, n=3000, seed=0)
    res, trace = holarchic_select(pat, grammar, "assess", holons, utility, eta=0.95, n=3000, seed=0)
    print(f"\n=== {label} ===")
    print(f"stability low-fidelity only: {base.pea:.3f}  ->  after delegation: {res.pea:.3f}   "
          f"recommended: {res.best.key}   active cost: {res.active_cost_s:.0f} s")
    for t in trace:
        print(f" round {t['round']}: contracts {{" + ", ".join(f"{k}: {v:.3f}" for k, v in t['contracts'].items()) + "}"
              + (f"  UNMET: {t['unmet']}" if t["unmet"] else ""))
        print("  narratives:", json.dumps(t["narratives"], indent=None))
