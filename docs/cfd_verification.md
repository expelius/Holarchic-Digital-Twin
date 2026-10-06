# Verification of the CFD rung (2026-10-07)

Annular gap of uniform height h = 0.3 mm, mean radius 12.15 mm, length 10 mm, pressure
difference 0.5 mmHg (Reynolds number ~ 5, so the narrow-gap Poiseuille solution applies):
Q = 2 pi r_m h^3 dP / (12 mu L) = 0.32715 mL/s. svMultiPhysics 2026 (built from source),
stabilised Navier-Stokes, hexahedral channel mesh, 48 x 10 elements in angle and height.

| Elements across the gap | CFD flux (mL/s) | CFD / analytic | Steady (last 10 steps) |
|---|---|---|---|
| 3 | 0.30598 | 0.9353 | change 0 |
| 6 | 0.31996 | 0.9780 | change 0 |
| Richardson (second order) | 0.32462 | 0.9923 | — |

Reading. The error falls with refinement and the extrapolated flux is within 0.8 % of the
analytic value; the remaining difference is consistent with the Neumann condition imposing a
traction (the averaged inlet pressure is 663.4 against 666.6 dyn/cm^2 imposed, 0.5 %).

A prediction made before the 3-element run failed and is recorded as such: the error of a
plain Galerkin solution with linear elements would be 1 - 1/N^2 (0.889 for N = 3); the
stabilised formulation is markedly more accurate than that (0.935). The convergence holds;
the error law was wrong.

Consequence for the patient channel: six elements across the gap carry about 2 % flux
error in viscous-dominated strips. The 0D model reproduces the analytic value to 0.1 % in
this regime, so differences between 0D and CFD on a real channel are model-form (geometry,
circumferential flow, jets), not discretisation.
