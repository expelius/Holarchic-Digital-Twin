# Holarchic Digital Twin for TAVR

**An open platform that unites structural and physical simulation with a decision layer,
organised as a holarchy. Package: `tavr_decide`.**

The twin applies hierarchy theory to a multiphysics decision problem. Each physical
stage is a **holon** in Koestler's sense, with Patten's portals as read by Allen &
Giampietro (2014): a *creaon* (what it accepts), a *genon* (what it emits), a coded half
(its fidelity ladder, calibration and contract) that constrains the works (the solver
that runs), optional *parts* (sub-holons), and a *narrative* to the level above. The
meaning of a holon lives one level up: how accurate it must be is not its own business
but a **tolerance contract** handed down by the decision that uses it. The orchestrator
never looks inside a holon.

What this is meant to beat, measurably: a two-day closed-service turnaround, a single
geometry without uncertainty, no notion of procedural time or reversible actions, device
models locked behind manufacturer agreements, and a structural accuracy bar of ±2 mm in
≥95 % of cases on 89 patients (the only FDA-authorised TAVR simulation). The first five we
already do differently; the last one is the number to beat on a public cohort, and it is
not beaten yet.

Commercial TAVR planning tools predict a configuration from a pre-procedural CT and return
a report hours or days later. They do not know which actions are still available, how
far the recommendation is from flipping, or whether running more physics would change
anything. TAVR-Decide is the missing layer, built to be used and extended by anyone:

| Layer | What it does | Why it is different |
|---|---|---|
| **Reversibility grammar** | Procedural states, reversible vs irreversible transitions, actions per state; one instance per device family | The orchestrator never knows which valve it is working with |
| **Uncertainty-aware anatomy** | Every measurement carries a distribution | Planning software has been shown to mis-measure the root systematically; a number hides that |
| **Modules with switchable fidelity** | A published 0D proxy answers in milliseconds; richer models (FE, CFD, learned surrogates) are plugged in and activated only when they could change the decision | Instead of always running all the physics |
| **Decision layer** | Recommended action, **action-stability probability**, **margin certificate in millimetres**, **value of computation** per module, explicit *"need more information"* state, provenance | None of this exists in any product |

Status: **v0.1, pre-alpha, benchmarking only.** The proxies' coefficients are declared
placeholders anchored to published cut-offs (see `tavr_decide/calibration.py`). Nothing
here is a clinical claim.

## Install and run

```bash
pip install -e ".[dev]"
python -m pytest -q
python examples/synthetic_patient.py
python examples/holarchic_patient.py
```

The example evaluates a synthetic patient at the ~80 % checkpoint of a self-expanding
valve (continue vs recapture, two sizes, two target depths), prints the ranking, the
stability probability, the margin certificates and the module-activation trace, and
writes `reports/synthetic_patient.md`.

## The ideas, in five lines

* **Stability, not accuracy.** The question is not "how precise is the model" but "would a
  better model change the action". The action-stability probability is Felli & Hazen's
  decision sensitivity (1998) computed over the uncertainty the active modules leave open.
* **Margin certificate.** For the recommended action, the smallest shift of one measurement
  (membranous-septum length, calcium volume, annulus diameter...) that flips the
  recommendation, in its own units. A clinician can read "flips if the final depth is
  1.2 mm deeper" where a probability means little.
* **Value of computation.** The expected value of partial perfect information on a module's
  error (Strong, Oakley & Brennan 2014, binned-regression estimator), scaled by the
  irreversibility of the action under consideration (Russell & Wefald 1991, with solvers
  as computations), minus the cost of running it under the procedure's time budget.
* **Minimal activation.** Greedy selection of the cheapest set of modules and fidelities
  that reaches the stability threshold. Prior work selects one model from a single fidelity
  ladder by accuracy; selecting a *subset of modules by decision stability* is the claim
  this repository exists to test.
* **Irreversibility first.** Compute is worth most right before an irreversible edge. In the
  reversible window, observing the patient competes with simulating the patient.
* **Delegation, not central control.** `holarchic_select` derives one tolerance contract per
  outcome from the decision margin (`s_tol = m / (w · z_η · √(2(1−ρ)))`) and hands it to
  the holon, which meets it with its cheapest sufficient rung and delegates to its parts
  the same way. A holon that cannot meet its contract says so: the honest answer is then
  to observe, not to compute. This is benchmark condition C6 against the centrally
  selected C5.

## Layout

```
tavr_decide/
  grammar.py      reversibility grammar; Evolut-like and balloon-expandable instances
  anatomy.py      Uncertain measurements, sampling, shifted copies for certificates
  modules.py      Module protocol; ConductionProxy (dMSID), PVLProxy (upper-LVOT calcium), PluggableModule
  holon.py        Holon (creaon, genon, rungs, parts, meet, narrate), Rung, leaf
  decision.py     evaluate, margin_certificate, evppi_of_module, value_of_computation, select_modules,
                  tolerance_contracts, holarchic_select
  report.py       markdown report
  calibration.py  every placeholder coefficient with its provenance
examples/         runnable end-to-end example
tests/            pytest suite
```

## Plugging in real physics

Any callable `(samples, action) -> risk_per_sample` becomes a module:

```python
from tavr_decide import PluggableModule
fe = PluggableModule(fn=my_febio_contact_risk, name="FEBio septal contact",
                     outcome="conduction", fidelity="high", cost_s=4*3600, error_sd=0.02)
```

`samples` is a dict of NumPy arrays (one draw per Monte-Carlo world) with the anatomy
fields plus `depth_noise_mm`; `action` carries size, target depth, manoeuvre and
inflation delta. The decision layer never looks inside a module.

## Roadmap

1. **Benchmark harness** (next): synthetic cohort, parametric self-expanding and
   balloon-expandable frames, an open FE oracle (FEBio via SlicerHeart), and the five
   pre-registered conditions C1–C5 (always-high, always-low, random, accuracy-driven
   selection, decision-driven selection) with hypotheses H1–H8 and thresholds.
2. 3D Slicer / SlicerHeart module for interactive use.
3. Procedural replay from real fluoroscopy sequences (requires a clinical partner).

## Citing

See `CITATION.cff`. A theory paper (computation versus observation under staged
irreversible interventions; margin certificates; a fidelity-invariance bound) and a
benchmark paper are in preparation.

## License

Apache License 2.0. Contributions welcome; open an issue first for anything larger than
a fix so the benchmark's pre-registration is not invalidated by silent changes.
