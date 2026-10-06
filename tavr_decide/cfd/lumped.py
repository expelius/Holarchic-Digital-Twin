"""Zero-dimensional hydraulics of the paravalvular channel (the leak holon's middle rung).

Each angular strip of the channel is a conduit of width w = r dtheta and height h(z). The
pressure drop for a flow Q combines

* a viscous term from lubrication theory, parallel-plate Poiseuille integrated along the
  strip:  dP_v = Q * integral( 12 mu / (w h^3) dz ),
* an inertial term for the jet that forms at the narrowest section and dissipates
  downstream:  dP_i = K rho Q^2 / (2 A_min^2),  with A_min = w * min(h),

so  dP = R Q + a Q^2  is solved for Q on each strip and summed. The viscous term rules thin
gaps, the inertial term rules wide ones; between the two (Reynolds numbers of hundreds to
thousands, which is where clinically relevant leaks live) the strips also exchange flow
circumferentially and the jet is not a simple orifice. That is the model-form error the
CFD rung exists to remove.

Grading follows the quantitative criteria for prosthetic aortic regurgitation used by
VARC-3 (regurgitant volume per beat: mild < 30 mL, moderate 30-59 mL, severe >= 60 mL;
effective regurgitant orifice area: mild < 0.10 cm^2, moderate 0.10-0.29, severe >= 0.30).
Hemodynamic inputs (diastolic pressure difference, diastolic duration) are placeholders
until patient values are supplied.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .channel import GapMap

MMHG = 1333.22                  # dyn/cm^2
RHO = 1.06                      # g/cm^3
MU = 0.035                      # poise (dyn s / cm^2)
DP_DIASTOLE_MMHG = 60.0         # PLACEHOLDER: mean aortic minus LV pressure during diastole
T_DIASTOLE_S = 0.50             # PLACEHOLDER: diastolic duration (~70 bpm)
K_JET = 1.5                     # entrance + sudden-expansion loss coefficient
RVOL_GRADES_ML = (30.0, 60.0)   # VARC-3 / ASE prosthetic AR: moderate, severe thresholds
EROA_GRADES_CM2 = (0.10, 0.30)


@dataclass
class PVLHydraulics:
    flow_ml_s: float
    rvol_ml: float
    eroa_cm2: float
    geometric_orifice_cm2: float
    strip_flow_ml_s: np.ndarray
    max_reynolds: float
    grade: str
    dp_mmhg: float
    model: str = "0D lubrication + orifice"


def grade_of(rvol_ml: float) -> str:
    lo, hi = RVOL_GRADES_ML
    if rvol_ml < 1.0:
        return "none/trace"
    return "mild" if rvol_ml < lo else ("moderate" if rvol_ml < hi else "severe")


def pvl_lumped(g: GapMap, dp_mmhg: float = DP_DIASTOLE_MMHG, t_diastole_s: float = T_DIASTOLE_S,
               rho: float = RHO, mu: float = MU, k_jet: float = K_JET, h_closed_mm: float = 0.02) -> PVLHydraulics:
    dP = dp_mmhg * MMHG
    nz, nt = g.r_skirt.shape
    dth = 2 * np.pi / nt
    h = g.h / 10.0                                          # cm
    r = (g.r_skirt + g.h / 2) / 10.0
    w = r * dth                                             # cm, per (z, theta)
    zc = g.z / 10.0                                         # cm
    dz = np.gradient(zc); dz[0] /= 2.0; dz[-1] /= 2.0         # trapezoid weights: they sum to the length
    q = np.zeros(nt)
    re_max = 0.0
    for i in range(nt):
        hi = h[:, i]
        if hi.min() <= h_closed_mm / 10.0:
            continue                                        # strip sealed somewhere along its length
        R = float(np.sum(12.0 * mu * dz / (w[:, i] * hi ** 3)))
        A = float(w[np.argmin(hi), i] * hi.min())
        a = k_jet * rho / (2.0 * A * A)
        qi = (-R + np.sqrt(R * R + 4.0 * a * dP)) / (2.0 * a)
        q[i] = qi
        v = qi / A
        re_max = max(re_max, rho * v * 2.0 * hi.min() / mu)      # hydraulic diameter of a slot ~ 2h
    Q = float(q.sum())
    v_jet = np.sqrt(2.0 * dP / rho)
    rvol = Q * t_diastole_s
    return PVLHydraulics(Q, rvol, Q / v_jet, g.min_section_area_mm2() / 100.0, q, re_max, grade_of(rvol), dp_mmhg)
