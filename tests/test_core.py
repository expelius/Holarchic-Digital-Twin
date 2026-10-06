import numpy as np
import pytest

from tavr_decide import (Anatomy, Uncertain, evolut_like_grammar, balloon_expandable_grammar,
                         ConductionProxy, PVLProxy, PluggableModule, Utility, evaluate,
                         margin_certificate, value_of_computation, select_modules, render_markdown)
from tavr_decide.decision import evppi_of_module
from tavr_decide.grammar import Action


def anatomy(calcium=21.0, ms=4.0, annulus=22.0, observed=None, sd_scale=1.0):
    return Anatomy(
        annulus_diameter_mm=Uncertain(annulus, 0.6 * sd_scale),
        ms_length_mm=Uncertain(ms, 1.0 * sd_scale),
        upper_lvot_calcium_mm3=Uncertain(calcium, 6.0 * sd_scale),
        observed_depth_mm=None if observed is None else Uncertain(observed, 0.8 * sd_scale),
    )


LOW = {"conduction": ConductionProxy(), "pvl": PVLProxy()}


# --- grammar -----------------------------------------------------------------------------

def test_evolut_grammar_marks_release_irreversible_and_recapture_reversible():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
    assert not g.is_reversible("assess", "release")
    assert g.is_reversible("assess", "partial_deploy")
    acts = g.actions("assess")
    cont = [a for a in acts if a.maneuver == "continue"][0]
    rec = [a for a in acts if a.maneuver == "recapture"][0]
    assert cont.target_depth_mm is None and cont.key == "continue/size26/current"
    assert rec.target_depth_mm == 3.0 and rec.recaptures_so_far == 1
    assert g.crosses_irreversible(cont) and not g.crosses_irreversible(rec)
    assert g.kappa(cont) > g.kappa(rec) == 1.0


def test_position_state_offers_size_and_depth_choices():
    g = evolut_like_grammar(sizes=(26, 29), depths=(3.0, 5.0))
    keys = {a.key for a in g.actions("position")}
    assert keys == {"position/size26/depth3", "position/size26/depth5",
                    "position/size29/depth3", "position/size29/depth5"}


def test_balloon_grammar_inflation_is_irreversible():
    g = balloon_expandable_grammar()
    assert not g.is_reversible("position", "inflate")
    assert all(g.crosses_irreversible(a) for a in g.actions("position"))


# --- anatomy -----------------------------------------------------------------------------

def test_anatomy_sampling_and_shift():
    a = anatomy(observed=4.0)
    s = a.sample(np.random.default_rng(0), 500)
    assert s["ms_length_mm"].shape == (500,) and "observed_depth_mm" in s
    assert abs(s["ms_length_mm"].mean() - 4.0) < 0.2
    b = a.shifted("ms_length_mm", -1.5)
    assert b.ms_length_mm.mean == pytest.approx(2.5)
    assert a.ms_length_mm.mean == 4.0  # frozen, original untouched


# --- proxies ----------------------------------------------------------------------------

def test_conduction_proxy_monotone_in_depth_and_ms_length():
    m = ConductionProxy()
    s = {"ms_length_mm": np.array([4.0, 4.0, 6.0]), "annulus_diameter_mm": np.array([22.0] * 3)}
    shallow = m.predict(s, Action(26, 3.0))
    deep = m.predict(s, Action(26, 5.0))
    assert deep[0] > shallow[0]            # deeper implant -> more conduction risk
    assert shallow[2] < shallow[0]         # longer membranous septum -> less risk
    assert np.all((shallow >= 0) & (shallow <= 1))


def test_continue_uses_observed_depth():
    m = ConductionProxy()
    s = {"ms_length_mm": np.array([4.0, 4.0]), "annulus_diameter_mm": np.array([22.0, 22.0]),
         "observed_depth_mm": np.array([2.0, 6.0])}
    r = m.predict(s, Action(26, None, "continue"))
    assert r[1] > r[0]
    with pytest.raises(ValueError):
        m.predict({"ms_length_mm": np.array([4.0]), "annulus_diameter_mm": np.array([22.0])},
                  Action(26, None, "continue"))


def test_pvl_proxy_monotone_in_calcium_and_oversizing():
    m = PVLProxy()
    s = {"upper_lvot_calcium_mm3": np.array([10.0, 40.0]), "annulus_diameter_mm": np.array([22.0, 22.0])}
    r26 = m.predict(s, Action(26, 3.0))
    r29 = m.predict(s, Action(29, 3.0))
    assert r26[1] > r26[0]                 # more calcium -> more leak
    assert r29[0] < r26[0]                 # more oversizing -> less leak


# --- decision layer ---------------------------------------------------------------------

def test_evaluate_is_deterministic_and_pea_in_range():
    g = evolut_like_grammar(size_in_situ=26)
    r1 = evaluate(anatomy(observed=4.5), g, "assess", LOW, Utility(), n=2000, seed=1)
    r2 = evaluate(anatomy(observed=4.5), g, "assess", LOW, Utility(), n=2000, seed=1)
    assert r1.best.key == r2.best.key and r1.pea == r2.pea
    assert 0.0 <= r1.pea <= 1.0
    assert set(r1.expected_utility) == {a.key for a in g.actions("assess")}


def test_deep_valve_with_short_septum_is_recaptured_shallower():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
    r = evaluate(anatomy(ms=3.0, observed=7.0), g, "assess", LOW, Utility(), n=2000, seed=0)
    assert r.best.maneuver == "recapture"
    g2 = evolut_like_grammar(size_in_situ=26, retarget_depths=(7.0,))
    r2 = evaluate(anatomy(ms=8.0, observed=3.0), g2, "assess", LOW, Utility(), n=2000, seed=0)
    assert r2.best.maneuver == "continue"   # already shallow and safe: do not recapture


def test_zero_uncertainty_gives_full_stability():
    g = evolut_like_grammar(size_in_situ=26)
    a = anatomy(observed=4.5, sd_scale=0.0)
    mods = {"conduction": ConductionProxy(error_sd=0.0), "pvl": PVLProxy(error_sd=0.0)}
    import tavr_decide.calibration as C
    old = C.DEPTH_ACHIEVEMENT_SD_MM
    C.DEPTH_ACHIEVEMENT_SD_MM = 0.0
    try:
        r = evaluate(a, g, "assess", mods, Utility(), n=500, seed=0)
    finally:
        C.DEPTH_ACHIEVEMENT_SD_MM = old
    assert r.pea == 1.0


def test_recapture_penalty_makes_second_recapture_worse_than_first():
    g0 = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,), recaptures_so_far=0)
    g1 = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,), recaptures_so_far=1)
    r0 = evaluate(anatomy(observed=4.5), g0, "assess", LOW, Utility(), n=1500, seed=3)
    r1 = evaluate(anatomy(observed=4.5), g1, "assess", LOW, Utility(), n=1500, seed=3)
    assert r1.expected_utility["recapture/size26/depth3"] < r0.expected_utility["recapture/size26/depth3"]


def test_margin_certificate_finds_flip_near_boundary_and_not_far_from_it():
    g = evolut_like_grammar(sizes=(26, 29), depths=(3.0,))
    u = Utility(w_conduction=1.0, w_pvl=1.0)
    near = margin_certificate(anatomy(calcium=21.0), g, "position", LOW, u,
                              variable="upper_lvot_calcium_mm3", max_shift=40.0, step=2.0, n=1500, seed=0)
    assert near is not None and near["from"] != near["to"]
    far = margin_certificate(anatomy(calcium=80.0), g, "position", LOW, u,
                             variable="upper_lvot_calcium_mm3", max_shift=20.0, step=2.0, n=1500, seed=0)
    assert far is None or abs(far["delta"]) > abs(near["delta"])


def test_depth_certificate_in_millimetres():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
    c = margin_certificate(anatomy(ms=4.0, observed=4.5), g, "assess", LOW, Utility(),
                           variable="observed_depth_mm", max_shift=4.0, step=0.1, n=1500, seed=0)
    assert c is not None and c["variable"] == "observed_depth_mm" and abs(c["delta"]) <= 4.0


def test_evppi_nonnegative_and_zero_without_error():
    g = evolut_like_grammar(size_in_situ=26)
    r = evaluate(anatomy(observed=4.5), g, "assess", LOW, Utility(), n=3000, seed=0)
    assert evppi_of_module(r, "conduction") >= 0.0
    mods = {"conduction": ConductionProxy(error_sd=0.0), "pvl": PVLProxy()}
    r0 = evaluate(anatomy(observed=4.5), g, "assess", mods, Utility(), n=3000, seed=0)
    assert evppi_of_module(r0, "conduction") == 0.0


def test_voc_scales_with_irreversibility():
    g_rev = evolut_like_grammar(size_in_situ=26, kappa_irreversible=1.0)
    g_irr = evolut_like_grammar(size_in_situ=26, kappa_irreversible=3.0)
    cand = PluggableModule(fn=ConductionProxy().predict, name="FE-contact", outcome="conduction",
                           cost_s=10.0, error_sd=0.01)
    r_rev = evaluate(anatomy(observed=4.5), g_rev, "assess", LOW, Utility(), n=3000, seed=0)
    r_irr = evaluate(anatomy(observed=4.5), g_irr, "assess", LOW, Utility(), n=3000, seed=0)
    v_rev = value_of_computation(r_rev, "conduction", cand, g_rev, lam=0.0)
    v_irr = value_of_computation(r_irr, "conduction", cand, g_irr, lam=0.0)
    if r_irr.best.maneuver == "continue":
        assert v_irr >= v_rev
    else:
        assert v_irr == pytest.approx(v_rev)


def test_select_modules_activates_only_when_it_pays_and_raises_stability():
    g = evolut_like_grammar(sizes=(26, 29), depths=(3.0,))
    high = {
        "conduction": PluggableModule(fn=ConductionProxy().predict, name="FE-contact",
                                      outcome="conduction", cost_s=60.0, error_sd=0.005),
        "pvl": PluggableModule(fn=PVLProxy().predict, name="FE-gap", outcome="pvl",
                               cost_s=60.0, error_sd=0.005),
    }
    base = evaluate(anatomy(calcium=35.0), g, "position", LOW, Utility(), n=3000, seed=0, eta=0.99)
    res, trace = select_modules(anatomy(calcium=35.0), g, "position", LOW, high, Utility(),
                                eta=0.99, lam=1e-6, n=3000, seed=0)
    assert res.pea >= base.pea
    activated = [t for t in trace if t.get("activated")]
    if base.pea < 0.99:
        assert activated, "a boundary case must trigger at least one activation"
        assert all(t["voc"] > 0 for t in activated)
    # with prohibitive cost nothing is activated
    _res2, trace2 = select_modules(anatomy(calcium=35.0), g, "position", LOW, high, Utility(),
                                   eta=0.99, lam=10.0, n=3000, seed=0)
    assert not [t for t in trace2 if t.get("activated")]


def test_report_renders():
    g = evolut_like_grammar(size_in_situ=26)
    r = evaluate(anatomy(observed=4.5), g, "assess", LOW, Utility(), n=500, seed=0)
    md = render_markdown(r, certificates=[{"variable": "ms_length_mm", "delta": -1.0, "from": "a", "to": "b"}], trace=[])
    assert "Recommended action" in md and "Margin certificates" in md and "decision-sufficient" in md
