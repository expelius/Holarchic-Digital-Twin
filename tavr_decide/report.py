"""Human-readable report of a decision evaluation."""
from __future__ import annotations

from .decision import DecisionResult


def render_markdown(result: DecisionResult, certificates: list[dict | None] | None = None,
                    trace: list[dict] | None = None, title: str = "TAVR-Decide report") -> str:
    lines = [f"# {title}", ""]
    lines.append(f"Procedural state: **{result.state}**  ")
    lines.append(f"Monte-Carlo worlds: {result.n}  ")
    lines.append(f"Active modules: " + ", ".join(f"{o} ← {m.name} [{m.fidelity}]" for o, m in result.modules.items()))
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    verdict = "STABLE" if not result.needs_more_information else "NEEDS MORE INFORMATION"
    lines.append(f"- Recommended action: **{result.best.key}**")
    lines.append(f"- Action-stability probability: **{result.pea:.3f}** (threshold {result.eta:.2f}) → {verdict}")
    if result.second is not None:
        lines.append(f"- Runner-up: {result.second.key} (expected-utility margin {result.margin:.4f})")
    lines.append("")
    lines.append("## Ranking")
    lines.append("")
    lines.append("| action | expected utility |")
    lines.append("|---|---|")
    for k, v in result.ranking:
        lines.append(f"| {k} | {v:.4f} |")
    lines.append("")
    if certificates:
        lines.append("## Margin certificates")
        lines.append("")
        lines.append("Smallest change of one measurement that would change the recommendation.")
        lines.append("")
        lines.append("| measurement | shift | new recommendation |")
        lines.append("|---|---|---|")
        for c in certificates:
            if c is None:
                continue
            lines.append(f"| {c['variable']} | {c['delta']:+.2f} | {c['to']} |")
        lines.append("")
    if trace is not None:
        lines.append("## Module activation trace")
        lines.append("")
        if not trace:
            lines.append("No richer module was activated: the low-fidelity set was decision-sufficient.")
        for t in trace:
            if t.get("activated"):
                lines.append(f"- Activated **{t['activated']}** for `{t['outcome']}` "
                             f"(value of computation {t['voc']:.4f}; stability before {t['pea_before']:.3f})")
            elif "contracts" in t:                      # holarchic (delegated) round
                pea = t.get("pea_before", float("nan"))
                if t.get("reason"):
                    lines.append(f"- Round {t['round']}: {t['reason']} (stability {pea:.3f}; "
                                 f"ceiling by computation {t.get('pea_ceiling_by_computation', float('nan')):.3f})")
                else:
                    c = ", ".join(f"{k}: {v:.3f}" for k, v in t["contracts"].items())
                    unmet = f"; unmet: {', '.join(t['unmet'])}" if t.get("unmet") else ""
                    lines.append(f"- Round {t['round']}: tolerance contracts {{{c}}} (stability before {pea:.3f}){unmet}")
                    for o, nar in t.get("narratives", {}).items():
                        if isinstance(nar, dict):
                            lines.append(f"  - `{o}` ← {nar['holon']}: rung *{nar['rung']}* [{nar['fidelity']}], "
                                         f"error {nar['error_sd']}, cost {nar['cost_s']:.0f} s")
            else:
                lines.append(f"- Stopped: {t.get('reason')} (stability {t.get('pea', float('nan')):.3f})")
        lines.append("")
    lines.append("---")
    lines.append("*Coefficients of the low-fidelity proxies are declared placeholders anchored to "
                 "published cut-offs; this report benchmarks decision logic and makes no clinical claim.*")
    return "\n".join(lines)
