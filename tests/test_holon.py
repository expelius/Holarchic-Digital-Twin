import numpy as np
import pytest

from tavr_decide import (Anatomy, Uncertain, evolut_like_grammar, ConductionProxy, PVLProxy,
                         Utility, evaluate, tolerance_contracts, holarchic_select, Holon, Rung, leaf)
from tavr_decide.decision import _z


def anatomy(calcium=21.0, ms=4.0, annulus=22.0, observed=4.5):
    return Anatomy(annulus_diameter_mm=Uncertain(annulus, 0.6), ms_length_mm=Uncertain(ms, 1.0),
                   upper_lvot_calcium_mm3=Uncertain(calcium, 6.0), observed_depth_mm=Uncertain(observed, 0.8))


CREAON = ("ms_length_mm", "annulus_diameter_mm", "upper_lvot_calcium_mm3")


def conduction_holon():
    p = ConductionProxy().predict
    return leaf("conduction", "conduction", CREAON, [
        ("dMSID proxy", "low", 0.001, 0.08, p),
        ("ROM septal contact (stand-in)", "mid", 60.0, 0.03, p),
        ("FE septal contact (stand-in)", "high", 3600.0, 0.005, p),
    ])


def pvl_holon():
    p = PVLProxy().predict
    return leaf("pvl", "pvl", CREAON, [
        ("calcium proxy", "low", 0.001, 0.10, p),
        ("FE gap (stand-in)", "high", 3600.0, 0.005, p),
    ])


# --- holon mechanics ----------------------------------------------------------------------

def test_leaf_meets_contract_with_cheapest_sufficient_rung():
    h = conduction_holon()
    assert h.meet(0.5) and h.rung.name == "dMSID proxy"
    assert h.meet(0.05) and h.rung.name == "ROM septal contact (stand-in)"
    assert h.meet(0.01) and h.rung.name == "FE septal contact (stand-in)"
    assert not h.meet(0.001) and h.rung.fidelity == "high"   # cannot meet: stays at best rung


def test_holon_is_a_module_and_checks_its_creaon():
    h = conduction_holon()
    g = evolut_like_grammar(size_in_situ=26)
    r = evaluate(anatomy(), g, "assess", {"conduction": h, "pvl": pvl_holon()}, Utility(), n=500)
    assert 0 <= r.pea <= 1
    with pytest.raises(ValueError):
        h.predict({"ms_length_mm": np.array([4.0])}, g.actions("assess")[0])


def test_composite_holon_delegates_and_combines_error_in_quadrature():
    frame = leaf("frame", "mechanics", CREAON, [("stat", "low", 0.001, 0.06, ConductionProxy().predict),
                                               ("fe", "high", 100.0, 0.01, ConductionProxy().predict)])
    tissue = leaf("tissue", "mechanics", CREAON, [("pop", "low", 0.001, 0.06, PVLProxy().predict),
                                                 ("cal", "high", 100.0, 0.01, PVLProxy().predict)])
    mech = Holon("mechanics", "conduction", CREAON, [Rung("assembly", "composite", 1.0, 0.0)],
                 parts=[frame, tissue], combine=lambda rs: 0.5 * (rs[0] + rs[1]))
    assert mech.error_sd == pytest.approx(np.sqrt(0.06 ** 2 * 2))
    assert mech.meet(0.2) and frame.rung.name == "stat" and tissue.rung.name == "pop"
    assert mech.meet(0.03) and frame.rung.name == "fe" and tissue.rung.name == "cal"
    assert mech.error_sd == pytest.approx(np.sqrt(2 * 0.01 ** 2))
    assert mech.cost_s == pytest.approx(201.0)
    n = mech.narrate()
    assert n["holon"] == "mechanics" and [p["rung"] for p in n["parts"]] == ["fe", "cal"]
    r = mech.predict({k: np.array([4.0, 22.0, 20.0])[i:i + 1] for i, k in enumerate(CREAON)} | {"observed_depth_mm": np.array([4.0])},
                     evolut_like_grammar(size_in_situ=26).actions("assess")[0])
    assert r.shape == (1,)


# --- contracts ------------------------------------------------------------------------------

def test_normal_quantile_matches_known_values():
    assert _z(0.95) == pytest.approx(1.6449, abs=2e-3)
    assert _z(0.975) == pytest.approx(1.9600, abs=2e-3)
    assert _z(0.5) == 0.0


def test_contracts_grow_with_margin_and_shrink_with_eta():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
    hol = {"conduction": conduction_holon(), "pvl": pvl_holon()}
    easy = evaluate(anatomy(ms=3.0, observed=7.5), g, "assess", hol, Utility(), n=1500, seed=0)
    hard = evaluate(anatomy(ms=4.0, observed=4.0), g, "assess", hol, Utility(), n=1500, seed=0)
    assert easy.margin > hard.margin
    c_easy, c_hard = tolerance_contracts(easy, Utility()), tolerance_contracts(hard, Utility())
    assert c_easy["conduction"] > c_hard["conduction"]
    assert tolerance_contracts(hard, Utility(), eta=0.99)["pvl"] < tolerance_contracts(hard, Utility(), eta=0.90)["pvl"]
    # a heavier weight on an outcome tightens that outcome's contract
    assert tolerance_contracts(hard, Utility(w_pvl=3.0))["pvl"] < c_hard["pvl"]


# --- delegated selection --------------------------------------------------------------------

def test_holarchic_select_leaves_easy_cases_cheap_and_opens_hard_ones_locally():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
    hol = {"conduction": conduction_holon(), "pvl": pvl_holon()}
    res_easy, tr_easy = holarchic_select(anatomy(ms=3.0, observed=7.5), g, "assess", hol, Utility(), eta=0.95, n=2000)
    assert res_easy.pea >= 0.95 and tr_easy == []
    assert hol["conduction"].rung.fidelity == "low" and hol["pvl"].rung.fidelity == "low"

    # A case whose instability is driven by model-form error (coarse low rungs), so that
    # computation can actually buy stability.
    p_c, p_p = ConductionProxy().predict, PVLProxy().predict
    coarse = {
        "conduction": leaf("conduction", "conduction", CREAON, [("proxy", "low", 0.001, 0.30, p_c),
                                                                 ("FE (stand-in)", "high", 3600.0, 0.005, p_c)]),
        "pvl": leaf("pvl", "pvl", CREAON, [("proxy", "low", 0.001, 0.30, p_p),
                                           ("FE (stand-in)", "high", 3600.0, 0.005, p_p)]),
    }
    hard = anatomy(ms=3.0, observed=7.5)
    base = evaluate(hard, g, "assess", coarse, Utility(), n=2000, seed=0)
    res_hard, tr_hard = holarchic_select(hard, g, "assess", coarse, Utility(), eta=0.95, n=2000)
    assert base.pea < 0.95 < res_hard.pea
    assert tr_hard and "contracts" in tr_hard[0] and set(tr_hard[0]["contracts"]) == {"conduction", "pvl"}
    assert tr_hard[0]["narratives"]["conduction"]["holon"] == "conduction"
    assert any(h.rung.fidelity == "high" for h in coarse.values()), "delegation must have opened a holon"


def test_holarchic_select_does_not_compute_when_instability_is_irreducible():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
    hol = {"conduction": conduction_holon(), "pvl": pvl_holon()}
    # near the boundary with precise modules: what is left is anatomy / depth-achievement noise
    res, tr = holarchic_select(anatomy(ms=4.0, observed=4.0), g, "assess", hol, Utility(), eta=0.99, n=2000)
    assert res.needs_more_information
    assert tr and tr[0].get("reason", "").startswith("irreducible")
    assert all(h.rung.fidelity == "low" for h in hol.values()), "no physics should have been opened"


def test_holarchic_select_reports_unmet_contract_instead_of_pretending():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(4.0,))
    # ladders whose best rung is still too coarse for a razor-thin margin
    coarse = leaf("conduction", "conduction", CREAON, [("proxy", "low", 0.001, 0.3, ConductionProxy().predict)])
    coarse_p = leaf("pvl", "pvl", CREAON, [("proxy", "low", 0.001, 0.3, PVLProxy().predict)])
    res, tr = holarchic_select(anatomy(ms=4.0, observed=4.0), g, "assess",
                               {"conduction": coarse, "pvl": coarse_p}, Utility(), eta=0.99, n=1500)
    assert res.needs_more_information
    assert tr and tr[0]["unmet"], "an unmeetable contract must be reported, not hidden"


def test_contracts_spend_only_what_the_anatomy_leaves():
    g = evolut_like_grammar(size_in_situ=26, retarget_depths=(3.0,))
    hol = {"conduction": conduction_holon(), "pvl": pvl_holon()}
    r = evaluate(anatomy(ms=4.0, observed=4.0), g, "assess", hol, Utility(), n=1500, seed=0)
    free = tolerance_contracts(r, Utility(), 0.95)
    floored = tolerance_contracts(r, Utility(), 0.95, var_floor=0.5 * (r.margin / _z(0.95)) ** 2, split=True)
    assert all(floored[o] < free[o] for o in free)
    # half the budget spent by anatomy and shared by two holons: each gets 1/2 of the free tolerance
    assert floored["pvl"] == pytest.approx(free["pvl"] * 0.5, rel=1e-6)
    gone = tolerance_contracts(r, Utility(), 0.95, var_floor=2 * (r.margin / _z(0.95)) ** 2)
    assert all(v == 0.0 for v in gone.values())
