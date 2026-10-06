"""The decision layer.

Given uncertainty-aware anatomy, a reversibility grammar, a set of active physics modules
and a utility, this module computes:

* expected utility per feasible action (Monte Carlo over anatomy, depth achievement and
  module error),
* the recommended action and its **action-stability probability** (PEA): the fraction
  of Monte-Carlo worlds in which the recommended action is also the best one. This is
  Felli & Hazen's decision sensitivity (Med Decis Making 1998) computed with respect to
  the uncertainty the active modules leave unresolved,
* a **margin certificate**: the smallest shift of one anatomical measurement, in its own
  physical units, that would change the recommendation,
* the **value of computation** of activating a richer module for one outcome: the
  expected value of partial perfect information on that module's error (estimated with
  the binned-regression approach of Strong, Oakley & Brennan, Med Decis Making 2014),
  scaled by the irreversibility of the action under consideration (Russell & Wefald,
  Artif Intell 1991, with solvers as computations), minus its cost,
* a greedy **minimal module selection**: activate the richer module with the highest
  value of computation until the stability threshold is met or no activation pays.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import calibration as C
from .anatomy import Anatomy
from .grammar import Action, ReversibilityGrammar
from .modules import Module


@dataclass
class Utility:
    """Additive utility over outcome risks, with a hard coronary constraint and a
    recapture penalty. Weights are declared before running, never fitted to results."""
    w_conduction: float = 1.0
    w_pvl: float = 1.0
    coronary_min_height_mm: float = 10.0        # PLACEHOLDER hard constraint
    infeasible_penalty: float = 10.0
    recapture_penalty: dict[int, float] = field(default_factory=lambda: dict(C.RECAPTURE_PENALTY))

    def of(self, risks: dict[str, np.ndarray], action: Action, samples: dict[str, np.ndarray]) -> np.ndarray:
        u = np.zeros_like(next(iter(risks.values())))
        if "conduction" in risks:
            u = u - self.w_conduction * risks["conduction"]
        if "pvl" in risks:
            u = u - self.w_pvl * risks["pvl"]
        k = min(action.recaptures_so_far, max(self.recapture_penalty))
        u = u - self.recapture_penalty.get(k, 0.0)
        h = np.minimum(samples["lcc_coronary_height_mm"], samples["rcc_coronary_height_mm"])
        u = u - self.infeasible_penalty * (h < self.coronary_min_height_mm)
        return u


@dataclass
class DecisionResult:
    state: str
    actions: list[Action]
    expected_utility: dict[str, float]
    sample_utility: dict[str, np.ndarray]
    best: Action
    second: Action | None
    margin: float
    pea: float
    eta: float
    modules: dict[str, Module]
    n: int
    eps: dict[str, np.ndarray]
    samples: dict[str, np.ndarray]

    @property
    def needs_more_information(self) -> bool:
        return self.pea < self.eta

    @property
    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self.expected_utility.items(), key=lambda kv: kv[1], reverse=True)

    @property
    def active_cost_s(self) -> float:
        return float(sum(m.cost_s for m in self.modules.values())) * len(self.actions)


def _draw(anatomy: Anatomy, modules: dict[str, Module], n: int, n_actions: int, seed: int):
    """Draw anatomy, depth-achievement noise and one model-form error per module.

    The error of a module is drawn per action (shape ``n x A``) with a common component
    shared across actions and an action-specific component, mixed by the module's
    ``error_corr`` (default 0.5). An error identical across all actions could never
    change which action is best, so it would be decision-irrelevant by construction;
    the action-specific part is what a richer module can resolve.
    """
    rng = np.random.default_rng(seed)
    samples = anatomy.sample(rng, n)
    samples["depth_noise_mm"] = rng.normal(0.0, C.DEPTH_ACHIEVEMENT_SD_MM, n)
    eps: dict[str, np.ndarray] = {}
    for o, m in modules.items():
        if m.error_sd <= 0:
            eps[o] = np.zeros((n, n_actions))
            continue
        rho = float(getattr(m, "error_corr", 0.5))
        common = rng.normal(0.0, 1.0, (n, 1))
        specific = rng.normal(0.0, 1.0, (n, n_actions))
        eps[o] = m.error_sd * (np.sqrt(rho) * common + np.sqrt(1.0 - rho) * specific)
    return samples, eps


def evaluate(anatomy: Anatomy, grammar: ReversibilityGrammar, state: str,
             modules: dict[str, Module], utility: Utility,
             n: int = 4000, seed: int = 0, eta: float = 0.95) -> DecisionResult:
    """Evaluate every feasible action in ``state`` under the active modules."""
    actions = grammar.actions(state)
    if not actions:
        raise ValueError(f"grammar '{grammar.name}' defines no actions in state '{state}'")
    samples, eps = _draw(anatomy, modules, n, len(actions), seed)
    sample_u: dict[str, np.ndarray] = {}
    for j, a in enumerate(actions):
        risks = {o: np.clip(m.predict(samples, a) + eps[o][:, j], 0.0, 1.0) for o, m in modules.items()}
        sample_u[a.key] = utility.of(risks, a, samples)
    keys = [a.key for a in actions]
    U = np.stack([sample_u[k] for k in keys], axis=1)          # n x A
    eu = U.mean(axis=0)
    order = np.argsort(-eu)
    best_i = int(order[0])
    second_i = int(order[1]) if len(order) > 1 else None
    pea = float(np.mean(np.argmax(U, axis=1) == best_i))
    margin = float(eu[best_i] - eu[second_i]) if second_i is not None else float("inf")
    return DecisionResult(
        state=state, actions=actions,
        expected_utility={k: float(v) for k, v in zip(keys, eu)},
        sample_utility=sample_u, best=actions[best_i],
        second=actions[second_i] if second_i is not None else None,
        margin=margin, pea=pea, eta=eta, modules=dict(modules), n=n, eps=eps, samples=samples,
    )


def margin_certificate(anatomy: Anatomy, grammar: ReversibilityGrammar, state: str,
                       modules: dict[str, Module], utility: Utility, variable: str,
                       max_shift: float, step: float, n: int = 4000, seed: int = 0) -> dict | None:
    """Smallest shift of ``variable``'s mean (in its own units) that changes the recommendation.

    Common random numbers (same seed) are used so that the flip reflects the shift, not
    Monte-Carlo noise. Returns ``None`` if no flip occurs within ``±max_shift``.
    """
    base = evaluate(anatomy, grammar, state, modules, utility, n=n, seed=seed)
    best_key = base.best.key
    found = None
    for k in range(1, int(round(max_shift / step)) + 1):
        for sign in (+1.0, -1.0):
            delta = sign * k * step
            res = evaluate(anatomy.shifted(variable, delta), grammar, state, modules, utility, n=n, seed=seed)
            if res.best.key != best_key:
                cand = {"variable": variable, "delta": delta, "from": best_key, "to": res.best.key}
                if found is None or abs(delta) < abs(found["delta"]):
                    found = cand
        if found is not None:
            return found
    return None


def evppi_of_module(result: DecisionResult, outcome: str, bins: int = 20) -> float:
    """Expected value of partial perfect information on one module's error term.

    Binned nonparametric regression of each action's sampled utility on the module's
    error draw (Strong, Oakley & Brennan 2014): E[max_a E[U_a | eps]] - max_a E[U_a].
    """
    E = result.eps[outcome]                       # n x A
    if np.allclose(E, 0.0):
        return 0.0
    keys = [a.key for a in result.actions]
    U = np.stack([result.sample_utility[k] for k in keys], axis=1)   # n x A
    n, A = U.shape
    # Conditional mean of each action's utility given that action's own error draw,
    # by univariate binned regression (each U_a depends on eps only through eps_a).
    cond = np.empty_like(U)
    for j in range(A):
        e = E[:, j]
        edges = np.quantile(e, np.linspace(0, 1, bins + 1))
        idx = np.clip(np.searchsorted(edges, e, side="right") - 1, 0, bins - 1)
        means = np.array([U[idx == b, j].mean() if np.any(idx == b) else U[:, j].mean()
                          for b in range(bins)])
        cond[:, j] = means[idx]
    inner = cond.max(axis=1).mean()
    outer = U.mean(axis=0).max()
    return float(max(inner - outer, 0.0))


def value_of_computation(result: DecisionResult, outcome: str, candidate: Module,
                         grammar: ReversibilityGrammar, lam: float = 1e-5) -> float:
    """kappa(best action) * EVPPI(module error) - lam * cost(candidate) * |actions|.

    ``lam`` converts seconds of compute into utility units; it is the procedural
    time budget expressed as a price, declared per state (seconds matter differently
    before and after the point of no return).
    """
    kappa = grammar.kappa(result.best)
    gain = kappa * evppi_of_module(result, outcome)
    cost = lam * candidate.cost_s * len(result.actions)
    return float(gain - cost)


def select_modules(anatomy: Anatomy, grammar: ReversibilityGrammar, state: str,
                   low: dict[str, Module], high: dict[str, Module], utility: Utility,
                   eta: float = 0.95, lam: float = 1e-5, n: int = 4000, seed: int = 0):
    """Greedy minimal activation: start from the low-fidelity set, activate the richer
    module with the highest positive value of computation until PEA >= eta or nothing pays.

    Returns ``(result, trace)`` where ``trace`` lists each activation with its VoC.
    """
    active: dict[str, Module] = dict(low)
    trace: list[dict] = []
    remaining = dict(high)
    while True:
        res = evaluate(anatomy, grammar, state, active, utility, n=n, seed=seed, eta=eta)
        if res.pea >= eta or not remaining:
            break
        vocs = {o: value_of_computation(res, o, m, grammar, lam) for o, m in remaining.items()
                if o in active}
        if not vocs:
            break
        o_best = max(vocs, key=vocs.get)
        if vocs[o_best] <= 0.0:
            trace.append({"activated": None, "reason": "no activation pays", "pea": res.pea, "voc": vocs})
            break
        trace.append({"activated": remaining[o_best].name, "outcome": o_best,
                      "voc": vocs[o_best], "pea_before": res.pea})
        active[o_best] = remaining.pop(o_best)
    return res, trace
