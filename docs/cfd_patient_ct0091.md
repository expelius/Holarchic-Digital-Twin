# First CFD on a deployed channel: 0D versus 3D (2026-10-07)

Case: TAVRP-PL ct0091 (public CT; routine, 1.5 mm, not a TAVR-planning scan: see the
caveat below), 29 mm parametric self-expanding frame deployed by FEBio at 4 mm below the
annulus plane. Paravalvular channel between the skirt (draped over the struts, 1.5 rows)
and the deformed wall, z from -3.9 to +7.8 mm. Cells thinner than 0.1 mm are contact.
Diastolic pressure difference 60 mmHg and diastolic time 0.5 s are placeholders.

| Rung | Model | Flow (mL/s) | RVol (mL/beat) | VARC-3 grade | Cost |
|---|---|---|---|---|---|
| Middle | 0D strips (lubrication + orifice), 3 strips open end to end | 2.21 | 1.1 | mild | milliseconds |
| High | 3D Navier-Stokes, svMultiPhysics, 4,376 hexes, 4 layers across the gap | 6.08 | 3.0 | mild | 21 min on 6 cores |

Steady to 0.1 % over the last 10 % of the run; mass conserved between the aortic and
ventricular faces to 1e-10; no linear-solver warnings.

Reading.

* The 0D rung underestimates the leak by a factor 2.75. The direction was predicted before
  the run: 66 % of the fluid cells connect both sides only once flow can detour around the
  contact points, and the strip model forbids circumferential flow. This is model-form
  error, not discretisation: the verification case shows both models agree with the
  analytic solution where the geometry is a uniform gap, and the CFD at 4 layers, if
  anything, under-predicts by a few percent.
* The error did not change the decision. Both rungs grade the leak as mild, an order of
  magnitude below the 30 mL moderate threshold. In this case the expensive rung would not
  have been worth running: this is the decision-sufficiency argument on a real geometry,
  and it is also a warning that the cheap rung's declared error was too small.
* One comparison is not a calibration. The declared 0D model-form error was raised from
  sigma_log 0.7 to 1.0 (a factor 2.75 is 1.4 sigma at 0.7); it should be fitted on a set of
  CFD comparisons and, ultimately, on echocardiographic grading.

Caveat that applies to every number above: the CT is not from an aortic-stenosis patient,
has no valve calcium, and its pseudo-label lumen merges the LV cavity with the LVOT and
includes the native leaflets. The pipeline is demonstrated; the leak value is not
clinically meaningful.
