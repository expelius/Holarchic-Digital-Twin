"""Declared coefficients for the low-fidelity proxies.

Every number here is a PLACEHOLDER anchored to a published cut-off or discriminative
statistic. The published sources give a threshold and an AUC, not a calibrated risk
curve; the logistic slopes and baseline risks below are chosen so that the proxy
reproduces the published cut-off as the point of steepest change, and must be refitted
on a cohort before any use beyond benchmarking the decision logic.

Provenance
----------
* dMSID (membranous-septum length minus implantation depth): optimal cut-off < 0 mm,
  AUC 0.896 for conduction disturbance; "the strongest and only modifiable predictor"
  (PMC9258482). MIDAS strategy (pre-release depth < MS length) reduced new permanent
  pacemaker from 9.7 % to 3.0 % (Jilaihawi et al., JACC Cardiovasc Interv 2019,
  doi:10.1016/j.jcin.2019.05.056).
* Upper-LVOT calcium volume >= 21 mm^3: AUC 0.80 for significant paravalvular
  regurgitation vs 0.60 for total LVOT calcium (TCTMD coverage; PMC11951446).
* Resheathing: a single resheath carries no measurable cost; two or more reduced device
  success from 89.9 % to 80 % and doubled one-year mortality (HR 2.06) in 1,026
  patients (Bernardi et al., JAHA 2021, PMC8649510).
"""

# --- Conduction proxy (dMSID) ---------------------------------------------------------
DMSID_CUTOFF_MM = 0.0          # published optimal cut-off
DMSID_RISK_AT_CUTOFF = 0.20    # PLACEHOLDER: risk of conduction disturbance when dMSID == 0
DMSID_SLOPE_PER_MM = 0.9       # PLACEHOLDER: logit change per mm of dMSID (risk falls as dMSID grows)
CONDUCTION_PROXY_ERROR_SD = 0.08   # PLACEHOLDER: model-form error of the proxy, in risk units

# --- Paravalvular leak proxy (upper-LVOT calcium) -----------------------------------
CALCIUM_CUTOFF_MM3 = 21.0      # published cut-off
PVL_RISK_AT_CUTOFF = 0.25      # PLACEHOLDER
PVL_SLOPE_PER_MM3 = 0.08       # PLACEHOLDER: logit change per mm^3 of calcium
PVL_OVERSIZING_LOGIT_PER_PCT = -0.12        # PLACEHOLDER: more oversizing -> less leak
CONDUCTION_OVERSIZING_LOGIT_PER_PCT = 0.06  # PLACEHOLDER: more oversizing -> more conduction risk
PVL_PROXY_ERROR_SD = 0.10      # PLACEHOLDER

# --- Deployment behaviour --------------------------------------------------------------
# Deviation between the targeted implantation depth and the depth actually achieved.
# No study quantifies the pre-release-to-final change; this is declared uncertainty,
# not a measured value (see proposal v2.0, section 2.2).
DEPTH_ACHIEVEMENT_SD_MM = 1.0  # PLACEHOLDER
# Utility penalty by cumulative number of recaptures (Bernardi 2021: >= 2 is costly).
RECAPTURE_PENALTY = {0: 0.0, 1: 0.0, 2: 0.10, 3: 0.20}

# --- Self-expanding sizing table (nominal annulus diameter ranges, mm) -----------------
# Approximate ranges for an Evolut-like platform; used only to compute percent oversizing.
SE_SIZING_MM = {23: (18.0, 20.0), 26: (20.0, 23.0), 29: (23.0, 26.0), 34: (26.0, 30.0)}

# --- Links from finite-element indices to risk (high-fidelity rungs) --------------------
# The mechanistic direction is established (more wall displacement below the membranous
# septum -> more conduction risk; more sealing gap -> more leak). The magnitudes below are
# PLACEHOLDERS: they have not been fitted to any cohort and exist so that the decision
# logic can be exercised end to end.
FE_COND_W0_MM = 0.5            # wall displacement (p90) at which conduction risk equals DMSID_RISK_AT_CUTOFF
FE_COND_SLOPE_PER_MM = 2.0     # logit change per mm of wall displacement
FE_PVL_G0_MM2 = 3.0            # sealing-gap area at which leak risk equals PVL_RISK_AT_CUTOFF
FE_PVL_SLOPE_PER_MM2 = 0.3     # logit change per mm^2 of gap area
# Declared error of the FE-corrected rung. It is deliberately LARGER than the proxies' until the
# links above are fitted: the first end-to-end run (phantom, depth 5 vs 3 mm) showed that with
# unfitted links the rung reduces a risk difference of ~0.3 given by the published dMSID proxy
# to ~0.02 and turns a stable decision (0.955) into a coin flip (0.51). The mechanics are solved;
# the outcome link is not. With this value no tolerance contract selects the rung for accuracy,
# and its indices are reported as mechanics, not as risk, until calibration against a cohort.
FE_RUNG_ERROR_SD = 0.15
FE_RUNG_COST_S = 300.0         # declared cost per action before any run has been timed

# --- Leak holon: hydraulic rungs ----------------------------------------------------------
# The middle and high rungs predict a regurgitant volume from the deployed geometry and grade
# it with the published VARC-3 threshold (moderate or worse: >= 30 mL/beat). What is declared
# here is only the model-form error of that prediction, as a standard deviation of log(RVol):
# PLACEHOLDERS until compared against echocardiographic grading in a cohort.
PVL_0D_SIGMA_LOG = 1.0          # 0D strips ignore circumferential flow and jet geometry; raised from 0.7
                                # after the first CFD comparison under-predicted by x2.75 (docs/cfd_patient_ct0091.md)
PVL_CFD_SIGMA_LOG = 0.4         # 3D CFD: remaining error is the skirt and leaflet modelling
PVL_RVOL_FLOOR_ML = 0.05        # below this the prediction is "no leak"

# Declared errors of the leak rungs in risk units, for the tolerance contracts. PLACEHOLDERS.
# Their ORDER reflects physical content (the CFD resolves the circumferential detours the 0D
# strips cannot), not validation: neither has been compared with echocardiography. A fixed
# error in risk units is crude: the real error is in log(RVol) and matters only near the
# 30 mL threshold, where a factor 2.75 can move the grade; far from it, it does not.
PVL_0D_RUNG_ERROR_SD = 0.08
PVL_CFD_RUNG_ERROR_SD = 0.04
CFD_RUNG_COST_S = 1300.0        # measured: 4,376 hexes, 200 steps, 6 cores (docs/cfd_patient_ct0091.md)
