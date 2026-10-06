"""TAVR-Decide: an open decision layer for staged transcatheter aortic valve deployment.

The package separates four things that commercial planning tools fuse:

* a *reversibility grammar* (which actions exist in which procedural state, and which
  transitions can be undone),
* *uncertainty-aware anatomy* (every measurement carries a distribution, not a number),
* *physics modules with switchable fidelity* (a cheap published proxy answers first; a
  richer model is activated only when it could change the decision),
* a *decision layer* that returns the recommended action together with its
  action-stability probability, a margin certificate in physical units, the value of
  activating each richer module, and an explicit "need more information" state.

Nothing here is a clinical claim. Coefficients of the low-fidelity proxies are declared
placeholders anchored to published cut-offs and must be calibrated before any use beyond
benchmarking (see ``calibration.py``).
"""
from .anatomy import Anatomy, Uncertain
from .grammar import (Action, Edge, ReversibilityGrammar, evolut_like_grammar,
                      balloon_expandable_grammar)
from .modules import Module, ConductionProxy, PVLProxy, PluggableModule
from .decision import (Utility, DecisionResult, evaluate, margin_certificate,
                       value_of_computation, select_modules, tolerance_contracts,
                       holarchic_select)
from .holon import Holon, Rung, leaf
from .report import render_markdown

__version__ = "0.1.0"
__all__ = [
    "Anatomy", "Uncertain", "Action", "Edge", "ReversibilityGrammar",
    "evolut_like_grammar", "balloon_expandable_grammar",
    "Module", "ConductionProxy", "PVLProxy", "PluggableModule",
    "Utility", "DecisionResult", "evaluate", "margin_certificate",
    "value_of_computation", "select_modules", "render_markdown",
    "tolerance_contracts", "holarchic_select", "Holon", "Rung", "leaf",
]
