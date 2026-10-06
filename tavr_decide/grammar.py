"""Reversibility grammar: procedural states, reversible/irreversible transitions, actions.

The orchestrator never knows which valve is in use; it only reads the grammar supplied by
the device module. Two instances are provided: a self-expanding recapturable platform
(Evolut/Navitor/Portico-like) and a balloon-expandable platform (SAPIEN/Myval-like).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Action:
    size_mm: int
    target_depth_mm: float | None    # None = release at the currently observed depth
    maneuver: str = "continue"       # 'continue' | 'recapture' | 'inflate' | 'position'
    inflation_delta_ml: float = 0.0  # balloon-expandable only
    recaptures_so_far: int = 0

    @property
    def key(self) -> str:
        depth = "current" if self.target_depth_mm is None else f"depth{self.target_depth_mm:g}"
        parts = [self.maneuver, f"size{self.size_mm}", depth]
        if self.inflation_delta_ml:
            parts.append(f"vol{self.inflation_delta_ml:+g}mL")
        return "/".join(parts)


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    reversible: bool


@dataclass
class ReversibilityGrammar:
    name: str
    states: list[str]
    edges: list[Edge]
    actions_by_state: dict[str, list[Action]] = field(default_factory=dict)
    # Irreversibility factor kappa(e) >= 1 applied to the value of computation for actions
    # that cross an irreversible edge (proposal v2.0, section 5.4). Declared, not derived.
    kappa_irreversible: float = 2.0
    crossing: dict[str, bool] = field(default_factory=dict)

    def actions(self, state: str) -> list[Action]:
        return list(self.actions_by_state.get(state, []))

    def crosses_irreversible(self, action: Action) -> bool:
        return self.crossing.get(action.maneuver, False)

    def kappa(self, action: Action) -> float:
        return self.kappa_irreversible if self.crosses_irreversible(action) else 1.0

    def is_reversible(self, src: str, dst: str) -> bool:
        for e in self.edges:
            if e.src == src and e.dst == dst:
                return e.reversible
        raise KeyError(f"no edge {src}->{dst}")


def evolut_like_grammar(sizes=(26, 29), depths=(3.0, 5.0), size_in_situ: int | None = None,
                        retarget_depths=(3.0, 5.0), recaptures_so_far: int = 0,
                        kappa_irreversible: float = 2.0) -> ReversibilityGrammar:
    """Self-expanding recapturable platform.

    * 'position' state (planning): choose size and target depth.
    * 'assess' state (about 80 % deployment, before the capsule marker reaches the
      spindle): the valve ``size_in_situ`` (default: first of ``sizes``) is already in.
      The operator may **continue** and release at the depth currently observed
      (irreversible), or **recapture** and re-target one of ``retarget_depths``
      (reversible, but each recapture counts). Changing size at this point means a
      new device and is not modelled as an 'assess' action.
    """
    in_situ = sizes[0] if size_in_situ is None else size_in_situ
    states = ["access", "cross", "position", "partial_deploy", "assess", "release", "post_assess"]
    edges = [
        Edge("access", "cross", True), Edge("cross", "position", True),
        Edge("position", "partial_deploy", True), Edge("partial_deploy", "assess", True),
        Edge("assess", "partial_deploy", True),   # recapture / reposition
        Edge("assess", "release", False),         # point of no return
        Edge("release", "post_assess", True),
    ]
    acts: list[Action] = [Action(in_situ, None, "continue", recaptures_so_far=recaptures_so_far)]
    for d in retarget_depths:
        acts.append(Action(in_situ, d, "recapture", recaptures_so_far=recaptures_so_far + 1))
    position_acts = [Action(s, d, "position") for s in sizes for d in depths]
    return ReversibilityGrammar(
        "self-expanding recapturable", states, edges,
        {"assess": acts, "position": position_acts},
        kappa_irreversible,
        crossing={"continue": True, "recapture": False, "position": False},
    )


def balloon_expandable_grammar(sizes=(23, 26), depths=(3.0,), inflation_deltas=(0.0, 2.0),
                               kappa_irreversible: float = 2.0) -> ReversibilityGrammar:
    """Balloon-expandable platform: positioning is reversible, inflation is not."""
    states = ["access", "cross", "position", "inflate", "post_assess"]
    edges = [Edge("access", "cross", True), Edge("cross", "position", True),
             Edge("position", "inflate", False), Edge("inflate", "post_assess", True)]
    acts = [Action(s, d, "inflate", inflation_delta_ml=v)
            for s in sizes for d in depths for v in inflation_deltas]
    return ReversibilityGrammar(
        "balloon-expandable", states, edges, {"position": acts}, kappa_irreversible,
        crossing={"inflate": True, "position": False},
    )
