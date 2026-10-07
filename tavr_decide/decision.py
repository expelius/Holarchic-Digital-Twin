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

    def weight(self, outcome: str) -> float:
        return {"conduction": self.w_conduction, "pvl": self.w_pvl}.get(outcome, 1.0)

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


# ---------------------------------------------------------------------------------------
# Holarchic selection: tolerance contracts handed down to holons
# ---------------------------------------------------------------------------------------

def tolerance_contracts(result: DecisionResult, utility: Utility, eta: float | None = None,
                        var_floor: float = 0.0, split: bool = False) -> dict[str, float]:
    """How much declared error each outcome's holon may carry before the decision flips.

    For an additive utility, an action-specific error e in outcome ``o`` moves that
    action's utility by w_o * e. The recommendation flips when the runner-up gains the
    expected-utility margin m. Requiring P(flip) <= 1 - eta bounds the standard deviation
    of the best-minus-runner-up utility difference by m / z_eta. Part of that budget is
    already spent by what no model can remove (anatomy, depth achievement): ``var_floor``,
    the variance of the difference with every module error set to zero. What remains,
    B = (m / z_eta)^2 - var_floor, is what the holons may spend; with ``split`` it is shared
    equally by the k outcomes. An error of standard deviation s and cross-action
    correlation rho contributes w^2 s^2 2 (1 - rho), so

        s_tol(o) = sqrt( B / k ) / ( w_o * sqrt(2 (1 - rho)) ).

    With var_floor = 0 and k = 1 this is the margin-only contract m / (w z sqrt(2(1-rho))).
    A non-positive B means no fidelity suffices: the contract is 0 and every holon reports
    it unmet. This is the formal statement that a holon's required fidelity is defined one
    level up, by the decision, not by the holon itself.
    """
    from math import sqrt
    eta = result.eta if eta is None else eta
    z = _z(eta)
    out: dict[str, float] = {}
    if not np.isfinite(result.margin):
        return {o: float("inf") for o in result.modules}
    budget = (result.margin / z) ** 2 - float(var_floor) if z > 0 else float("inf")
    k = len(result.modules) if split else 1
    for o, m in result.modules.items():
        rho = float(getattr(m, "error_corr", 0.5))
        w = utility.weight(o)
        if budget <= 0:
            out[o] = 0.0
            continue
        out[o] = float(sqrt(budget / k) / (w * sqrt(max(2.0 * (1.0 - rho), 1e-9))))
    return out


def _z(p: float) -> float:
    """Standard-normal quantile without scipy (Acklam's rational approximation)."""
    if p <= 0.5:
        return 0.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    if p > 1 - plow:
        q = np.sqrt(-2 * np.log(1 - p))
        return float(-(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1))
    q = p - 0.5
    r = q * q
    return float((((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1))


def holarchic_select(anatomy: Anatomy, grammar: ReversibilityGrammar, state: str,
                     holons: dict[str, "Module"], utility: Utility, eta: float = 0.95,
                     n: int = 4000, seed: int = 0, max_rounds: int = 3):
    """Delegated selection (benchmark condition C6).

    Round: evaluate with the holons' current rungs -> derive one tolerance contract per
    outcome from the decision margin -> each holon meets its contract on its own,
    recursively through its parts -> re-evaluate. Stops when the stability threshold is
    met, when nothing changed, or when a holon reports it cannot meet its contract
    (then the honest output is "observe, do not compute").

    Returns ``(result, trace)``; each trace entry carries the contracts and the holons'
    narratives for that round.
    """
    trace: list[dict] = []
    for h in holons.values():
        if hasattr(h, "reset"):
            h.reset()
    res = evaluate(anatomy, grammar, state, holons, utility, n=n, seed=seed, eta=eta)
    var_floor = 0.0
    if res.pea < eta:
        # Ceiling reachable by computation alone: stability if every module error were
        # resolved. If even that is below eta, the instability is irreducible by any
        # physics (anatomy, depth achievement) and the honest move is to observe.
        ceil_res = _without_module_error(anatomy, grammar, state, holons, utility, n, seed, eta)
        if ceil_res.pea < eta:
            trace.append({"round": 0, "pea_before": res.pea, "pea_ceiling_by_computation": ceil_res.pea,
                          "contracts": {}, "unmet": list(holons), "reason": "irreducible by computation: observe",
                          "narratives": {o: (h.narrate() if hasattr(h, "narrate") else h.name) for o, h in holons.items()}})
            return res, trace
        # variance of the best-minus-runner-up difference that no model can remove
        if res.second is not None:
            d = ceil_res.sample_utility[res.best.key] - ceil_res.sample_utility[res.second.key]
            var_floor = float(np.var(d))
    for _round in range(max_rounds):
        if res.pea >= eta:
            break
        contracts = tolerance_contracts(res, utility, eta, var_floor=var_floor, split=True)
        before = {o: getattr(h, "active", None) for o, h in holons.items()}
        unmet = []
        for o, h in holons.items():
            if hasattr(h, "meet") and not h.meet(contracts[o]):
                unmet.append(o)
        after = {o: getattr(h, "active", None) for o, h in holons.items()}
        trace.append({"round": _round + 1, "pea_before": res.pea, "contracts": contracts,
                      "unmet": unmet,
                      "narratives": {o: (h.narrate() if hasattr(h, "narrate") else h.name) for o, h in holons.items()}})
        if after == before:
            break
        res = evaluate(anatomy, grammar, state, holons, utility, n=n, seed=seed, eta=eta)
        if unmet:
            break
    return res, trace


class _ZeroErrorView:
    """Module view with the same physics and zero declared error (for the computation ceiling)."""
    def __init__(self, m):
        self._m = m
        self.name, self.outcome, self.fidelity, self.cost_s = m.name, m.outcome, m.fidelity, m.cost_s
        self.error_sd = 0.0
        self.error_corr = getattr(m, "error_corr", 0.5)

    def predict(self, samples, action):
        return self._m.predict(samples, action)


def _without_module_error(anatomy, grammar, state, modules, utility, n, seed, eta) -> DecisionResult:
    views = {o: _ZeroErrorView(m) for o, m in modules.items()}
    return evaluate(anatomy, grammar, state, views, utility, n=n, seed=seed, eta=eta)


def _pea_without_module_error(anatomy, grammar, state, modules, utility, n, seed, eta) -> float:
    return _without_module_error(anatomy, grammar, state, modules, utility, n, seed, eta).pea
