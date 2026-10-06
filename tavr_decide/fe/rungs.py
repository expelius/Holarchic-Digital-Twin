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
    return float(_sigmoid(_logit(C.PVL_RISK_AT_CUTOFF) + C.FE_PVL_SLOPE_PER_MM2 * (gap_area_mm2 - C.FE_PVL_G0_MM2)))


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
        self.cache[key] = out
        self.runs.append(out)
        return out

    @property
    def measured_cost_s(self) -> float:
        done = [r["wall_s"] for r in self.runs]
        return float(np.mean(done)) if done else C.FE_RUNG_COST_S

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

    def _corrected(self, proxy, link, index_key: str):
        def fn(samples: dict[str, np.ndarray], action: Action) -> np.ndarray:
            r = self.solve(action)
            lo = proxy.predict(samples, action)
            if not r["normal"] or index_key not in r or not np.isfinite(r[index_key]):
                return lo                                  # the works failed: fall back to the proxy, visibly
            hi_nom = link(r[index_key])
            lo_nom = float(proxy.predict(self._nominal(), action)[0])
            return np.clip(lo + (hi_nom - lo_nom), 0.0, 1.0)
        return fn

    def holons(self, creaon: tuple[str, ...] = ("ms_length_mm", "annulus_diameter_mm", "upper_lvot_calcium_mm3")) -> dict[str, Holon]:
        cp, pp = ConductionProxy(), PVLProxy()
        cost = self.measured_cost_s
        cond = Holon("conduction", "conduction", creaon, [
            Rung("dMSID proxy", "low", 0.001, cp.error_sd, cp.predict),
            Rung("FE wall displacement below MS", "high", cost, C.FE_RUNG_ERROR_SD,
                 self._corrected(cp, conduction_link, "wall_p90_mm"))])
        pvl = Holon("paravalvular leak", "pvl", creaon, [
            Rung("upper-LVOT calcium proxy", "low", 0.001, pp.error_sd, pp.predict),
            Rung("FE sealing gap", "high", cost, C.FE_RUNG_ERROR_SD,
                 self._corrected(pp, pvl_link, "gap_area_mm2"))])
        return {"conduction": cond, "pvl": pvl}
