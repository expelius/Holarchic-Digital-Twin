"""Holons: the hierarchy-theory unit of the twin.

Vocabulary follows Koestler (1967) for the part/whole duality and Patten & Auble (1980)
as read by Allen & Giampietro (Ecol. Model. 2014) for the portals:

* **creaon** – the input portal: which fields of the sampled world the holon accepts;
* **genon** – the output portal: the one outcome risk the holon emits, together with
  its declared error and its cost;
* the **coded half** – rate-independent: the fidelity ladder (``rungs``), the
  calibration and the contract. It constrains the
* **works** – rate-dependent: the solver call that actually runs (the rung's ``fn``);
* **parts** – sub-holons; a composite holon is a whole to its parts and a part to the
  orchestrator;
* **narrate** – the holon's story to the level above: which rung is active, what error
  it declares, what it cost, and the same for its parts.

The meaning of a holon lives one level up (Allen & Giampietro's reading of the final
cause): the fidelity a holon must deliver is not its own business but a **tolerance
contract** handed down by the decision that uses it. ``meet`` selects the cheapest rung
whose declared error satisfies the contract and delegates to the parts the same way.
The orchestrator never looks inside.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .grammar import Action

RiskFn = Callable[[dict[str, np.ndarray], Action], np.ndarray]


@dataclass
class Rung:
    """One rung of a holon's fidelity ladder."""
    name: str
    fidelity: str
    cost_s: float
    error_sd: float
    fn: RiskFn | None = None     # None for composite holons (risk comes from the parts)


@dataclass
class Holon:
    name: str
    outcome: str                              # genon: the risk it emits
    creaon: tuple[str, ...]                   # input fields it accepts
    rungs: list[Rung]                         # cheapest first
    parts: list["Holon"] = field(default_factory=list)
    combine: Callable[[list[np.ndarray]], np.ndarray] | None = None   # composite: parts' risks -> risk
    active: int = 0
    error_corr: float = 0.5

    # ---- Module protocol (so a Holon drops into evaluate/select_modules unchanged) ----
    @property
    def rung(self) -> Rung:
        return self.rungs[self.active]

    @property
    def fidelity(self) -> str:
        return self.rung.fidelity

    @property
    def cost_s(self) -> float:
        own = self.rung.cost_s
        return own + sum(p.cost_s for p in self.parts)

    @property
    def error_sd(self) -> float:
        """Declared error of the active configuration. For a composite holon the parts'
        errors are combined in quadrature (independent errors) on top of the rung's own."""
        own = self.rung.error_sd
        if not self.parts:
            return own
        return float(np.sqrt(own ** 2 + sum(p.error_sd ** 2 for p in self.parts)))

    def predict(self, samples: dict[str, np.ndarray], action: Action) -> np.ndarray:
        missing = [k for k in self.creaon if k not in samples]
        if missing:
            raise ValueError(f"holon '{self.name}' creaon lacks {missing}")
        if self.parts:
            if self.combine is None:
                raise ValueError(f"composite holon '{self.name}' needs a combine function")
            return np.asarray(self.combine([p.predict(samples, action) for p in self.parts]), dtype=float)
        if self.rung.fn is None:
            raise ValueError(f"holon '{self.name}' rung '{self.rung.name}' has no works (fn)")
        return np.asarray(self.rung.fn(samples, action), dtype=float)

    # ---- The coded half: contracts ----------------------------------------------------
    def meet(self, tolerance: float) -> bool:
        """Select the cheapest configuration whose declared error is <= ``tolerance``.

        Leaf holon: walk the ladder from the cheapest rung. Composite holon: give each
        part an equal share of the error budget in quadrature (tolerance / sqrt(k)),
        then pick the cheapest own rung that fits in what remains. Returns False when no
        configuration can meet the contract (the orchestrator then needs observation,
        not computation).
        """
        if self.parts:
            k = len(self.parts)
            own_min = min(r.error_sd for r in self.rungs)
            share = tolerance / np.sqrt(k + (1 if own_min > 0 else 0))   # a share for the own rung only if it carries error
            ok = all(p.meet(share) for p in self.parts)
            parts_err2 = sum(p.error_sd ** 2 for p in self.parts)
            remaining2 = tolerance ** 2 - parts_err2
            if not ok or remaining2 <= 0:
                self.active = len(self.rungs) - 1
                return False
            for i, r in enumerate(sorted(range(len(self.rungs)), key=lambda i: self.rungs[i].cost_s)):
                if self.rungs[r].error_sd ** 2 <= remaining2:
                    self.active = r
                    return True
            self.active = len(self.rungs) - 1
            return False
        for r in sorted(range(len(self.rungs)), key=lambda i: self.rungs[i].cost_s):
            if self.rungs[r].error_sd <= tolerance:
                self.active = r
                return True
        self.active = len(self.rungs) - 1
        return False

    def reset(self) -> None:
        self.active = 0
        for p in self.parts:
            p.reset()

    def narrate(self) -> dict:
        d = {"holon": self.name, "outcome": self.outcome, "rung": self.rung.name,
             "fidelity": self.fidelity, "error_sd": round(self.error_sd, 4), "cost_s": self.cost_s}
        if self.parts:
            d["parts"] = [p.narrate() for p in self.parts]
        return d


def leaf(name: str, outcome: str, creaon: tuple[str, ...], ladder: list[tuple[str, str, float, float, RiskFn]],
         error_corr: float = 0.5) -> Holon:
    """Convenience: build a leaf holon from (name, fidelity, cost_s, error_sd, fn) tuples."""
    return Holon(name, outcome, creaon, [Rung(*t) for t in ladder], error_corr=error_corr)
