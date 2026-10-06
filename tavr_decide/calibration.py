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
