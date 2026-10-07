from tavr_decide import Anatomy, Uncertain, evolut_like_grammar, evaluate, holarchic_select, margin_certificate, Utility
from tavr_decide.cli import default_holons
from tavr_decide.clinical import clinical_summary, describe_action
from tavr_decide.grammar import Action


def anat(obs=4.6):
    return Anatomy(Uncertain(22.4, 0.6), Uncertain(3.5, 1.0), Uncertain(26.0, 6.0), observed_depth_mm=Uncertain(obs, 0.8))


def test_actions_read_as_clinical_sentences_in_both_languages():
    a = anat()
    assert describe_action(Action(26, None, "continue"), a, "es").startswith("Liberar en la posición actual (≈4.6 mm")
    assert describe_action(Action(26, 3.0, "recapture", recaptures_so_far=1), a, "es") == \
        "Recapturar y reposicionar a 3 mm bajo la cúspide no coronaria (sería la recaptura n.º 1)"
    assert describe_action(Action(29, 5.0, "position"), a, "en") == "29 mm valve with a target depth of 5 mm"
    assert "nominal volume +2 mL" in describe_action(Action(23, 3.0, "inflate", inflation_delta_ml=2.0), a, "en")


def test_summary_states_recommendation_stability_and_what_would_flip_it():
    a = anat()
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0, 4.0, 5.0))
    hol, u = default_holons(), Utility()
    res, trace = holarchic_select(a, g, "assess", hol, u, eta=0.95, n=1500, seed=0)
    cert = margin_certificate(a, g, "assess", hol, u, "observed_depth_mm", 3.0, 0.1, n=1500)
    es = clinical_summary(res, a, [cert], trace, "es")
    assert "**Recomendación:**" in es and "Recapturar" in es
    assert ("no es estable" in es) == (res.pea < 0.95)
    if any(str(x.get("reason", "")).startswith("irreducible") for x in trace):
        assert "observar mejor" in es and "segunda proyección" in es
    assert "la profundidad observada" in es and "más superficial" in es
    en = clinical_summary(res, a, [cert], trace, "en")
    assert "**Recommendation:**" in en and "Research software" in en


def test_summary_says_when_physics_would_help_and_describes_flips_in_their_scenario():
    a = anat()
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0, 4.0, 5.0))
    hol, u = default_holons(), Utility()
    res, trace = holarchic_select(a, g, "assess", hol, u, eta=0.95, n=3000, seed=0)
    cert = margin_certificate(a, g, "assess", hol, u, "observed_depth_mm", 3.0, 0.1, n=1500)
    es = clinical_summary(res, a, [cert], trace, "es")
    if res.pea < 0.95 and not any(str(x.get("reason", "")).startswith("irreducible") for x in trace):
        assert "física de mayor fidelidad" in es
    if cert and cert["to"].startswith("continue"):
        shifted = a.observed_depth_mm.mean + cert["delta"]
        assert f"≈{shifted:.1f} mm" in es
