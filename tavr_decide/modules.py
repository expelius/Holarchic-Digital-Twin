"""Physics modules with switchable fidelity.

A module maps sampled anatomy plus a candidate action to a risk in [0, 1] for one
outcome ("conduction", "pvl", ...). Each module declares its fidelity, its cost in
seconds, and its model-form error (``error_sd``, in risk units). The decision layer
draws one error realisation per module per Monte-Carlo sample, shared across actions,
so that the value of resolving that error (by activating a richer module) can be computed.

Two published zero-dimensional proxies are provided as the low-fidelity rung of the
conduction and paravalvular-leak ladders. Higher rungs (finite-element contact, CFD) are
plugged in through :class:`PluggableModule` with a user-supplied callable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from . import calibration as C
from .grammar import Action


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def oversizing_pct(size_mm: float, annulus_diameter_mm: np.ndarray) -> np.ndarray:
    """Percent oversizing of the valve inflow relative to the measured annulus diameter."""
    return 100.0 * (size_mm - annulus_diameter_mm) / annulus_diameter_mm


def achieved_depth_mm(action: Action, samples: dict[str, np.ndarray]) -> np.ndarray:
    """Depth actually achieved, including declared achievement noise.

    ``target_depth_mm is None`` means "release where the valve is now": the observed
    depth at the checkpoint (``samples['observed_depth_mm']``) plus the declared
    pre-release-to-final deviation.
    """
    if action.target_depth_mm is None:
        if "observed_depth_mm" not in samples:
            raise ValueError("action releases at the current depth but no observed_depth_mm was given")
        base = samples["observed_depth_mm"]
    else:
        base = np.full_like(samples["ms_length_mm"], float(action.target_depth_mm))
    noise = samples.get("depth_noise_mm")
    return base if noise is None else base + noise


@runtime_checkable
class Module(Protocol):
    name: str
    outcome: str
    fidelity: str
    cost_s: float
    error_sd: float

    def predict(self, samples: dict[str, np.ndarray], action: Action) -> np.ndarray: ...


@dataclass
class ConductionProxy:
    """Conduction-disturbance risk from dMSID = membranous-septum length - implantation depth.

    Zero-dimensional, milliseconds. Published cut-off < 0 mm (AUC 0.896); slope and
    baseline are placeholders (see ``calibration.py``).
    """
    name: str = "conduction/dMSID"
    outcome: str = "conduction"
    fidelity: str = "low"
    cost_s: float = 0.001
    error_sd: float = C.CONDUCTION_PROXY_ERROR_SD

    def predict(self, samples: dict[str, np.ndarray], action: Action) -> np.ndarray:
        depth = achieved_depth_mm(action, samples)
        dmsid = samples["ms_length_mm"] - depth
        ov = oversizing_pct(action.size_mm, samples["annulus_diameter_mm"])
        logit = (_logit(C.DMSID_RISK_AT_CUTOFF)
                 - C.DMSID_SLOPE_PER_MM * (dmsid - C.DMSID_CUTOFF_MM)
                 + C.CONDUCTION_OVERSIZING_LOGIT_PER_PCT * ov)
        return _sigmoid(logit)


@dataclass
class PVLProxy:
    """Paravalvular-leak risk from upper-LVOT calcium volume and oversizing.

    Zero-dimensional. Published cut-off >= 21 mm^3 (AUC 0.80); slopes are placeholders.
    """
    name: str = "pvl/upper-LVOT-calcium"
    outcome: str = "pvl"
    fidelity: str = "low"
    cost_s: float = 0.001
    error_sd: float = C.PVL_PROXY_ERROR_SD

    def predict(self, samples: dict[str, np.ndarray], action: Action) -> np.ndarray:
        ca = samples["upper_lvot_calcium_mm3"]
        ov = oversizing_pct(action.size_mm, samples["annulus_diameter_mm"])
        logit = (_logit(C.PVL_RISK_AT_CUTOFF)
                 + C.PVL_SLOPE_PER_MM3 * (ca - C.CALCIUM_CUTOFF_MM3)
                 + C.PVL_OVERSIZING_LOGIT_PER_PCT * ov)
        return _sigmoid(logit)


@dataclass
class PluggableModule:
    """Wrap any callable ``(samples, action) -> risk`` as a module of declared fidelity.

    Use this to attach a finite-element contact model, a CFD leak model, a learned
    surrogate, or an oracle in a benchmark. ``error_sd`` is the model-form error you are
    willing to declare for it; the decision layer trusts that number.
    """
    fn: Callable[[dict[str, np.ndarray], Action], np.ndarray]
    name: str
    outcome: str
    fidelity: str = "high"
    cost_s: float = 3600.0
    error_sd: float = 0.02

    def predict(self, samples: dict[str, np.ndarray], action: Action) -> np.ndarray:
        return np.asarray(self.fn(samples, action), dtype=float)
