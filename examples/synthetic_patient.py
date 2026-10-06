"""End-to-end example on a synthetic patient at the 80 % checkpoint of a self-expanding valve.

Run:  python examples/synthetic_patient.py
It prints a markdown report and writes it to reports/synthetic_patient.md.
"""
from pathlib import Path

from tavr_decide import (Anatomy, Uncertain, evolut_like_grammar, ConductionProxy, PVLProxy,
                         PluggableModule, Utility, evaluate, margin_certificate, select_modules,
                         render_markdown)

# 1. Anatomy with declared uncertainty (segmentation + measurement).
patient = Anatomy(
    annulus_diameter_mm=Uncertain(22.4, 0.6),
    ms_length_mm=Uncertain(3.5, 1.0),          # short membranous septum: conduction-sensitive
    upper_lvot_calcium_mm3=Uncertain(26.0, 6.0),  # just above the published 21 mm^3 cut-off
    lcc_coronary_height_mm=Uncertain(13.0, 1.0),
    rcc_coronary_height_mm=Uncertain(15.0, 1.0),
    observed_depth_mm=Uncertain(4.6, 0.8),       # fluoroscopic depth at ~80 % deployment
    label="synthetic-01",
)

# 2. Reversibility grammar of the device (self-expanding, recapturable), at 'assess':
#    a 26 mm valve is in; continue (release where it is) or recapture and re-target.
grammar = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0, 4.0, 5.0))

# 3. Low-fidelity rung: published 0D proxies (milliseconds).
low = {"conduction": ConductionProxy(), "pvl": PVLProxy()}

# 4. Higher rungs. Here they are stand-ins with the same physics and a smaller declared
#    error, so the example runs in seconds; in the benchmark they are FE/CFD callables.
high = {
    "conduction": PluggableModule(fn=ConductionProxy().predict, name="FE septal contact (stand-in)",
                                  outcome="conduction", fidelity="high", cost_s=3 * 3600, error_sd=0.01),
    "pvl": PluggableModule(fn=PVLProxy().predict, name="FE frame-annulus gap (stand-in)",
                           outcome="pvl", fidelity="high", cost_s=3 * 3600, error_sd=0.01),
}

utility = Utility(w_conduction=1.0, w_pvl=1.0)

# 5. Decide with the cheap rung only.
base = evaluate(patient, grammar, "assess", low, utility, n=4000, seed=0, eta=0.95)

# 6. Margin certificates in physical units.
certs = [
    margin_certificate(patient, grammar, "assess", low, utility, "observed_depth_mm", max_shift=3.0, step=0.1, n=2000),
    margin_certificate(patient, grammar, "assess", low, utility, "ms_length_mm", max_shift=4.0, step=0.25, n=2000),
    margin_certificate(patient, grammar, "assess", low, utility, "upper_lvot_calcium_mm3", max_shift=40.0, step=2.0, n=2000),
]

# 7. Activate richer physics only if it pays (lam prices seconds of compute in utility units).
final, trace = select_modules(patient, grammar, "assess", low, high, utility, eta=0.95, lam=1e-7, n=4000, seed=0)

md = render_markdown(final, certificates=certs, trace=trace,
                     title=f"TAVR-Decide report — {patient.label} — state '{final.state}'")
md = md.replace("## Recommendation", f"Low-fidelity-only stability was {base.pea:.3f}.\n\n## Recommendation", 1)
print(md)
out = Path(__file__).resolve().parents[1] / "reports"
out.mkdir(exist_ok=True)
(out / "synthetic_patient.md").write_text(md, encoding="utf-8")
