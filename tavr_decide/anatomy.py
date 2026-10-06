"""Uncertainty-aware anatomy."""
from __future__ import annotations

from dataclasses import dataclass, field, fields

import numpy as np


@dataclass(frozen=True)
class Uncertain:
    """A measurement with a declared standard deviation (Gaussian)."""
    mean: float
    sd: float = 0.0

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        if self.sd <= 0:
            return np.full(n, float(self.mean))
        return rng.normal(self.mean, self.sd, n)

    def shifted(self, delta: float) -> "Uncertain":
        return Uncertain(self.mean + delta, self.sd)


@dataclass(frozen=True)
class Anatomy:
    """Pre-procedural anatomy. Every field carries its own uncertainty.

    Units: mm for lengths, mm^3 for calcium volume.
    """
    annulus_diameter_mm: Uncertain
    ms_length_mm: Uncertain                 # membranous septum length
    upper_lvot_calcium_mm3: Uncertain
    lcc_coronary_height_mm: Uncertain = field(default_factory=lambda: Uncertain(14.0, 1.0))
    rcc_coronary_height_mm: Uncertain = field(default_factory=lambda: Uncertain(15.0, 1.0))
    # Intraprocedural observation at the checkpoint (e.g. depth below the non-coronary
    # cusp on fluoroscopy at ~80 % deployment). None before deployment starts.
    observed_depth_mm: Uncertain | None = None
    label: str = "patient"

    def sample(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, Uncertain):
                out[f.name] = v.sample(rng, n)
        return out

    def shifted(self, name: str, delta: float) -> "Anatomy":
        """Return a copy with one measurement's mean shifted by ``delta``."""
        kw = {f.name: getattr(self, f.name) for f in fields(self)}
        kw[name] = kw[name].shifted(delta)
        return Anatomy(**kw)
