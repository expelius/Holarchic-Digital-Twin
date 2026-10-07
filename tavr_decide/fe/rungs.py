"""The finite-element rung of the conduction and leak holons.

A deployment is one deterministic solve per action; the decision layer needs a risk per
Monte-Carlo world. The two are joined by a multifidelity additive correction: the
finite-element result fixes the level of the cheap proxy at the nominal anatomy, and the
proxy carries the variation across the sampled anatomies.

    risk_high(world, action) = proxy(world, action) + [ link(FE index(action)) - proxy(nominal, action) ]

The correction is cached per (size, depth). One solve feeds both holons, so opening the
conduction holon makes the leak holon's high rung free for the same actions: the cost of
the works is shared, which is what a composite holon is for.

What the high rung is and is not: the mechanics are solved; the links from mechanical
index to risk are placeholders (see ``calibration.py``). It corrects the proxy with
physics, it does not yet predict outcomes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .. import calibration as C
from ..anatomy import Anatomy
from ..frame import FrameSpec, self_expanding_frame
from ..grammar import Action
from ..holon import Holon, Rung
from ..modules import ConductionProxy, PVLProxy, _logit, _sigmoid
from .deploy import DeploymentResult, run_deployment
from .post import sealing_gap, wall_displacement
from .vessel import PatientVessel


def conduction_link(wall_p90_mm: float) -> float:
    return float(_sigmoid(_logit(C.DMSID_RISK_AT_CUTOFF) + C.FE_COND_SLOPE_PER_MM * (wall_p90_mm - C.FE_COND_W0_MM)))


def pvl_link(gap_area_mm2: float) -> float:
    """Superseded as a decision input: a band-wise gap is not a leak path (see cfd.channel).
    Kept for reporting and backward comparison."""
    return float(_sigmoid(_logit(C.PVL_RISK_AT_CUTOFF) + C.FE_PVL_SLOPE_PER_MM2 * (gap_area_mm2 - C.FE_PVL_G0_MM2)))


def pvl_risk_from_rvol(rvol_ml: float, sigma_log: float) -> float:
    """P(regurgitant volume >= 30 mL/beat, VARC-3 moderate or worse) under a lognormal
    model-form error of the hydraulic prediction."""
    from scipy.special import ndtr
    from ..cfd.lumped import RVOL_GRADES_ML
    r = max(float(rvol_ml), C.PVL_RVOL_FLOOR_ML)
    return float(ndtr((np.log(r) - np.log(RVOL_GRADES_ML[0])) / sigma_log))


@dataclass
class FEEngine:
    """Runs and caches deployments for one patient; exposes high-fidelity rungs."""
    anatomy: Anatomy
    vessel: PatientVessel
    workdir: Path
    frame_factory: Callable[[int], FrameSpec] = lambda size: self_expanding_frame(size, n_cells_circ=12, n_rows=5).spec
    fe_kwargs: dict = field(default_factory=dict)
    gap_band: tuple[float, float] = (-3.0, 1.0)
    cache: dict = field(default_factory=dict)
    runs: list = field(default_factory=list)
    hemo_kwargs: dict = field(default_factory=dict)       # dp_mmhg, t_diastole_s: patient values when known
    gap_maps: dict = field(default_factory=dict)
    cfd_kwargs: dict = field(default_factory=dict)        # n_h, nproc, CFDParams fields
    cfd_cache: dict = field(default_factory=dict)
    cfd_runs: list = field(default_factory=list)

    # ---- the works -----------------------------------------------------------------------
    def depth_of(self, action: Action) -> float:
        if action.target_depth_mm is not None:
            return float(action.target_depth_mm)
        if self.anatomy.observed_depth_mm is None:
            raise ValueError("action releases at the current depth but the anatomy has no observed depth")
        return float(self.anatomy.observed_depth_mm.mean)

    def solve(self, action: Action) -> dict:
        key = (int(action.size_mm), round(self.depth_of(action), 2))
        if key in self.cache:
            return self.cache[key]
        depth = key[1]
        wd = Path(self.workdir) / f"size{key[0]}_depth{depth:g}"
        res: DeploymentResult = run_deployment(self.frame_factory(key[0]), None, wd, vessel=self.vessel,
                                               inflow_z=-depth, **self.fe_kwargs)
        out = {"key": key, "normal": res.normal_termination, "wall_s": res.wall_s, "workdir": str(wd)}
        if res.normal_termination and res.vessel_final is not None:
            w = wall_displacement(res, z_max=-float(self.anatomy.ms_length_mm.mean))
            g = sealing_gap(res, z_band=self.gap_band)
            out.update(wall_p90_mm=w["p90_mm"], wall_max_mm=w["max_mm"], gap_area_mm2=g["area_mm2"],
                       gap_max_mm=g["max_gap_mm"], inflow_expansion=float(res.expansion_ratio[0]))
            from ..cfd.channel import gap_map
            from ..cfd.lumped import pvl_lumped
            gm = gap_map(res)
            h0 = pvl_lumped(gm, **self.hemo_kwargs)
            out.update(rvol_0d_ml=h0.rvol_ml, eroa_0d_cm2=h0.eroa_cm2, grade_0d=h0.grade,
                       channel_open_fraction=gm.open_fraction, max_reynolds_0d=h0.max_reynolds)
            self.gap_maps[key] = gm
        self.cache[key] = out
        self.runs.append(out)
        return out

    def cfd_solve(self, action: Action) -> dict:
        """CFD of the paravalvular channel of this action's deployment (runs the deployment
        first if needed). Shares the deployment with the conduction holon and the 0D rung."""
        fe = self.solve(action)
        key = fe["key"]
        if key in self.cfd_cache:
            return self.cfd_cache[key]
        out = {"key": key, "normal": False, "wall_s": 0.0}
        if fe["normal"] and key in self.gap_maps:
            from ..cfd.channel import channel_mesh
            from ..cfd.svmp import CFDParams, pvl_cfd
            kw = dict(self.cfd_kwargs)
            n_h, nproc = kw.pop("n_h", 4), kw.pop("nproc", 6)
            params = CFDParams(**{"dt_s": 2e-4, "n_steps": 200, "save_every": 100, **kw})
            if self.hemo_kwargs.get("dp_mmhg") is not None:
                params.dp_mmhg = self.hemo_kwargs["dp_mmhg"]
            try:
                mesh = channel_mesh(self.gap_maps[key], n_h=n_h)
            except ValueError:                         # sealed: no fluid path, no leak
                out.update(normal=True, rvol_cfd_ml=0.0, flow_cfd_ml_s=0.0, sealed=True)
            else:
                wd = Path(fe["workdir"]) / "cfd"
                kwargs = {"t_diastole_s": self.hemo_kwargs["t_diastole_s"]} if "t_diastole_s" in self.hemo_kwargs else {}
                c = pvl_cfd(mesh, wd, params, nproc=nproc, **kwargs)
                out.update(normal=c.steady_rel_change < 0.02, wall_s=c.wall_s, rvol_cfd_ml=c.rvol_ml,
                           flow_cfd_ml_s=c.flow_ml_s, grade_cfd=c.grade, steady_rel_change=c.steady_rel_change,
                           n_elems=int(len(mesh.elems)))
        self.cfd_cache[key] = out
        self.cfd_runs.append(out)
        return out

    @property
    def measured_cost_s(self) -> float:
        done = [r["wall_s"] for r in self.runs]
        return float(np.mean(done)) if done else C.FE_RUNG_COST_S

    @property
    def measured_cfd_cost_s(self) -> float:
        done = [r["wall_s"] for r in self.cfd_runs if r.get("wall_s")]
        return float(np.mean(done)) if done else C.CFD_RUNG_COST_S

    # ---- the genon: risk per sampled world ------------------------------------------------
    def _nominal(self) -> dict[str, np.ndarray]:
        a = self.anatomy
        s = {"annulus_diameter_mm": a.annulus_diameter_mm.mean, "ms_length_mm": a.ms_length_mm.mean,
             "upper_lvot_calcium_mm3": a.upper_lvot_calcium_mm3.mean,
             "lcc_coronary_height_mm": a.lcc_coronary_height_mm.mean, "rcc_coronary_height_mm": a.rcc_coronary_height_mm.mean,
             "depth_noise_mm": 0.0}
        if a.observed_depth_mm is not None:
            s["observed_depth_mm"] = a.observed_depth_mm.mean
        return {k: np.array([float(v)]) for k, v in s.items()}

    def _corrected(self, proxy, link, index_key: str, solver=None):
        solver = solver or self.solve

        def fn(samples: dict[str, np.ndarray], action: Action) -> np.ndarray:
            r = solver(action)
            lo = proxy.predict(samples, action)
            if not r["normal"] or index_key not in r or not np.isfinite(r[index_key]):
                return lo                                  # the works failed: fall back to the proxy, visibly
            hi_nom = link(r[index_key])
            lo_nom = float(proxy.predict(self._nominal(), action)[0])
            return np.clip(lo + (hi_nom - lo_nom), 0.0, 1.0)
        return fn

    def holons(self, creaon: tuple[str, ...] = ("ms_length_mm", "annulus_diameter_mm", "upper_lvot_calcium_mm3"),
               with_cfd: bool = True) -> dict[str, Holon]:
        cp, pp = ConductionProxy(), PVLProxy()
        cost = self.measured_cost_s
        cond = Holon("conduction", "conduction", creaon, [
            Rung("dMSID proxy", "low", 0.001, cp.error_sd, cp.predict),
            Rung("FE wall displacement below MS", "high", cost, C.FE_RUNG_ERROR_SD,
                 self._corrected(cp, conduction_link, "wall_p90_mm"))])
        rungs = [Rung("upper-LVOT calcium proxy", "low", 0.001, pp.error_sd, pp.predict),
                 Rung("FE channel + 0D hydraulics", "mid", cost, C.PVL_0D_RUNG_ERROR_SD,
                      self._corrected(pp, lambda rv: pvl_risk_from_rvol(rv, C.PVL_0D_SIGMA_LOG), "rvol_0d_ml"))]
        if with_cfd:
            rungs.append(Rung("FE channel + 3D CFD", "high", cost + self.measured_cfd_cost_s, C.PVL_CFD_RUNG_ERROR_SD,
                              self._corrected(pp, lambda rv: pvl_risk_from_rvol(rv, C.PVL_CFD_SIGMA_LOG),
                                              "rvol_cfd_ml", solver=self.cfd_solve)))
        pvl = Holon("paravalvular leak", "pvl", creaon, rungs)
        return {"conduction": cond, "pvl": pvl}
