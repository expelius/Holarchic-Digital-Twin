# Numerical sensitivity of the deployment (2026-10-06)

Case: parametric self-expanding frame, 26 mm, 12 cells × 5 rows, released into a 22 mm
cylindrical landing zone that covers the inflow third. FEBio 4.13 with Pardiso, 4 threads.
Quantity: expansion ratio (deployed radius / free radius) by axial band. All eight runs
ended in normal termination with zero equilibrium drift. Raw table:
`sensitivity_2026-10-06.csv`; script: `runs/sensitivity.py`.

| What varies | Values | Inflow expansion | Change vs baseline |
|---|---|---|---|
| Elements along each strut | 2 → 3 → 4 | 0.8696 → 0.8659 → 0.8639 | −0.0037, −0.0057 |
| Frame–vessel contact penalty | 1e-2 / **1e-3** / 1e-4 | 0.8689 / 0.8696 / 0.8737 | −0.0007 / — / +0.0041 |
| Vessel circumferential divisions | 48 → 72 | 0.8696 → 0.8704 | +0.0008 |
| Vessel elements through the wall | 2 → 3 | 0.8696 → 0.8697 | +0.0001 |
| Sleeve–frame contact penalty | 0.01 → 0.003 | 0.8696 → 0.8696 | 0 |

Reading.

* The strut discretisation dominates: the inflow expansion falls monotonically with
  refinement and the step shrinks (0.0037, then 0.0020), which is the signature of a
  converging sequence. The coarsest mesh over-predicts inflow expansion by about 0.006,
  i.e. roughly 0.1–0.17 mm in radius. Coarse linear hexahedra are too stiff in bending.
* The frame–vessel penalty is converged at 1e-3 (1e-2 changes the result by 0.0007). At 1e-4
  the frame visibly sinks into the wall; do not use it.
* Wall mesh and sleeve penalty do not matter at this resolution.

Consequence for the decision layer. The wall-displacement index differs by about 0.08 mm
between implantation depths of 3 and 5 mm on the phantom, which is the same order as the
discretisation error in absolute radius at two elements per strut. The error is largely
common to both actions (same mesh), but any comparison between actions should be confirmed
at three elements per strut before it is trusted.

Wall times ranged from 38 to 324 s; they were measured while other jobs shared the machine
and are indicative only.

Not covered here: material model (neo-Hookean nitinol with a placeholder modulus), wall
thickness, calcium stiffness, friction, and the parametric frame's proportions. Those are
modelling uncertainties, not numerical ones, and are larger.
