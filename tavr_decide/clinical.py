"""Clinical wording of a decision: what to do, how firm it is, and what would change it.

The technical report (action keys, expected utilities, rung narratives) stays for audit;
this module writes the paragraph an operator reads first, in Spanish or English.
"""
from __future__ import annotations

from .anatomy import Anatomy
from .decision import DecisionResult
from .grammar import Action

T = {
    "es": {
        "title": "Resumen clínico",
        "release": "Liberar en la posición actual (≈{obs:.1f} mm bajo la cúspide no coronaria)",
        "release_noobs": "Liberar en la posición actual",
        "recapture": "Recapturar y reposicionar a {d:g} mm bajo la cúspide no coronaria (sería la recaptura n.º {k})",
        "plan": "Válvula de {s} mm con profundidad objetivo de {d:g} mm",
        "inflate": "Inflar con volumen nominal{v}",
        "recommend": "**Recomendación:** {a}.",
        "stable": "La recomendación es **estable**: en el {p:.0%} de los escenarios plausibles sigue siendo la mejor (umbral {e:.0%}).",
        "unstable": "La recomendación **no es estable**: sólo en el {p:.0%} de los escenarios plausibles sigue siendo la mejor (umbral {e:.0%}). La alternativa más cercana es: {r}.",
        "observe": "Calcular más física no lo resolvería: la incertidumbre está en los datos del paciente, no en el modelo. Conviene **observar mejor** antes de actuar{hint}.",
        "computed": "Para estabilizarla se activaron modelos de mayor fidelidad: {m}.",
        "review": "Modo revisión: se ejecutó la física de mayor fidelidad para todas las acciones ({m}).",
        "uncalibrated": "La física de mayor fidelidad **todavía no está calibrada** contra desenlaces: declara más error que los modelos rápidos, y por eso no aumenta la estabilidad. Sus números (desplazamiento de pared, volumen de fuga) se reportan como mecánica, no como riesgo.",
        "rungs": {"FE wall displacement below MS": "elementos finitos", "FE channel + 0D hydraulics": "elementos finitos + hidráulica 0D",
                  "FE channel + 3D CFD": "elementos finitos + CFD 3D"},
        "needs_physics": "Esta incertidumbre **sí se reduciría con física de mayor fidelidad** ({o}): los modelos rápidos tienen más error del que la decisión tolera. Ejecute el despliegue por elementos finitos y el CFD (sección «Física»).",
        "flip_intro": "Qué cambiaría la recomendación:",
        "flip": "- si {v} fuera **{d:.1f} {u} {dir}**, la recomendación pasaría a: {to}",
        "none_flip": "- ninguna medida dentro del rango explorado cambia la recomendación",
        "more": "mayor", "less": "menor", "higher_valve": "más superficial (la válvula más alta)", "deeper_valve": "más profunda",
        "hint_depth": " (por ejemplo, confirmar la profundidad en una segunda proyección)",
        "outcomes": {"conduction": "conducción", "pvl": "fuga paravalvular"},
        "vars": {"ms_length_mm": ("la longitud del septo membranoso", "mm"),
                 "upper_lvot_calcium_mm3": ("el calcio del tracto de salida superior", "mm³"),
                 "annulus_diameter_mm": ("el diámetro del anillo", "mm"),
                 "observed_depth_mm": ("la profundidad observada", "mm")},
        "disclaimer": "_Software de investigación; los coeficientes de los modelos rápidos son provisionales y no sustituyen el juicio clínico._",
    },
    "en": {
        "title": "Clinical summary",
        "release": "Release at the current position (≈{obs:.1f} mm below the non-coronary cusp)",
        "release_noobs": "Release at the current position",
        "recapture": "Recapture and reposition to {d:g} mm below the non-coronary cusp (recapture number {k})",
        "plan": "{s} mm valve with a target depth of {d:g} mm",
        "inflate": "Inflate with nominal volume{v}",
        "recommend": "**Recommendation:** {a}.",
        "stable": "The recommendation is **stable**: it stays best in {p:.0%} of plausible scenarios (threshold {e:.0%}).",
        "unstable": "The recommendation is **not stable**: it stays best in only {p:.0%} of plausible scenarios (threshold {e:.0%}). The closest alternative is: {r}.",
        "observe": "More physics would not settle it: the uncertainty is in the patient data, not in the model. **Observe better** before acting{hint}.",
        "computed": "Higher-fidelity models were activated to stabilise it: {m}.",
        "review": "Review mode: the higher-fidelity physics was run for every action ({m}).",
        "uncalibrated": "The higher-fidelity physics is **not yet calibrated** against outcomes: it declares more error than the fast models, so it does not raise the stability. Its numbers (wall displacement, leak volume) are reported as mechanics, not as risk.",
        "rungs": {},
        "needs_physics": "This uncertainty **would shrink with higher-fidelity physics** ({o}): the fast models carry more error than the decision tolerates. Run the finite-element deployment and the CFD (\"Physics\" section).",
        "flip_intro": "What would change the recommendation:",
        "flip": "- if {v} were **{d:.1f} {u} {dir}**, the recommendation would become: {to}",
        "none_flip": "- no measurement within the explored range changes the recommendation",
        "more": "larger", "less": "smaller", "higher_valve": "shallower (valve higher)", "deeper_valve": "deeper",
        "hint_depth": " (for example, confirm the depth in a second projection)",
        "outcomes": {"conduction": "conduction", "pvl": "paravalvular leak"},
        "vars": {"ms_length_mm": ("the membranous septum length", "mm"),
                 "upper_lvot_calcium_mm3": ("the upper-LVOT calcium", "mm³"),
                 "annulus_diameter_mm": ("the annulus diameter", "mm"),
                 "observed_depth_mm": ("the observed depth", "mm")},
        "disclaimer": "_Research software; fast-model coefficients are provisional and do not replace clinical judgement._",
    },
}


def describe_action(a: Action, anatomy: Anatomy | None = None, lang: str = "es") -> str:
    t = T[lang]
    if a.maneuver == "continue":
        if a.target_depth_mm is None and anatomy is not None and anatomy.observed_depth_mm is not None:
            return t["release"].format(obs=anatomy.observed_depth_mm.mean)
        return t["release_noobs"]
    if a.maneuver == "recapture":
        return t["recapture"].format(d=a.target_depth_mm, k=a.recaptures_so_far)
    if a.maneuver == "inflate":
        return t["inflate"].format(v=f" {a.inflation_delta_ml:+g} mL" if a.inflation_delta_ml else "")
    return t["plan"].format(s=a.size_mm, d=a.target_depth_mm)


def _action_by_key(res: DecisionResult, key: str) -> Action | None:
    return next((a for a in res.actions if a.key == key), None)


def clinical_summary(res: DecisionResult, anatomy: Anatomy, certificates=None, trace=None, lang: str = "es") -> str:
    t = T[lang]
    lines = [f"## {t['title']}", "", t["recommend"].format(a=describe_action(res.best, anatomy, lang)), ""]
    if res.pea >= res.eta:
        lines.append(t["stable"].format(p=res.pea, e=res.eta))
    else:
        runner = describe_action(res.second, anatomy, lang) if res.second is not None else "—"
        lines.append(t["unstable"].format(p=res.pea, e=res.eta, r=runner))
        irreducible = any(str(x.get("reason", "")).startswith("irreducible") for x in (trace or []))
        if irreducible:
            hint = t["hint_depth"] if anatomy.observed_depth_mm is not None else ""
            lines.append("")
            lines.append(t["observe"].format(hint=hint))
        else:
            unmet = sorted({o for x in (trace or []) for o in x.get("unmet", [])})
            if unmet:
                lines.append("")
                lines.append(t["needs_physics"].format(o=", ".join(t["outcomes"].get(o, o) for o in unmet)))
    opened = []
    for x in trace or []:
        for o, nar in (x.get("narratives") or {}).items():
            if isinstance(nar, dict) and nar.get("fidelity") not in (None, "low"):
                opened.append(f"{t['outcomes'].get(o, nar['holon'])} ({t['rungs'].get(nar['rung'], nar['rung'])})")
    if opened:
        forced = any(x.get("round") == "forced" for x in trace or [])
        lines += ["", t["review" if forced else "computed"].format(m=", ".join(sorted(set(opened))))]
        if res.pea < res.eta:
            lines += ["", t["uncalibrated"]]
    certs = [c for c in (certificates or []) if c]
    lines += ["", t["flip_intro"]]
    if not certs:
        lines.append(t["none_flip"])
    for c in sorted(certs, key=lambda c: abs(c["delta"])):
        name, unit = t["vars"].get(c["variable"], (c["variable"], ""))
        if c["variable"] == "observed_depth_mm":
            direction = t["deeper_valve"] if c["delta"] > 0 else t["higher_valve"]
        else:
            direction = t["more"] if c["delta"] > 0 else t["less"]
        to = _action_by_key(res, c["to"])
        scenario = anatomy.shifted(c["variable"], c["delta"])        # describe the target in its own scenario
        lines.append(t["flip"].format(v=name, d=abs(c["delta"]), u=unit, dir=direction,
                                      to=describe_action(to, scenario, lang) if to else c["to"]))
    lines += ["", t["disclaimer"], ""]
    return "\n".join(lines)
