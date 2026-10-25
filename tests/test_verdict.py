import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from gravitas import Experiment, screen

STD = 9.80665
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load(name):
    return Experiment.from_yaml(os.path.join(EXAMPLES, name))


def _worst(v):
    return max(v.observables, key=lambda o: o.delta_center)


def test_standard_gravity_is_fail_with_zero_change():
    v = screen(_load("reference_case.yaml"), STD)
    assert v.result == "FAIL" and _worst(v).delta_center == 0.0


def test_reference_case_golden_values():
    # golden regression values (heuristic coupling); guards cross-platform determinism in CI
    v = screen(_load("reference_case.yaml"), 1e-3 * STD)
    assert v.result == "PASS" and v.limiting_mechanism == "oxygen"
    assert _worst(v).delta_center == pytest.approx(0.17314929314, rel=1e-6)
    assert v.uncertainty_threshold == pytest.approx(0.04881951092, rel=1e-6)


def test_lunar_gravity_is_marginal_for_reference_case():
    v = screen(_load("reference_case.yaml"), 1.62)
    assert v.result == "MARGINAL"
    assert _worst(v).delta_center == pytest.approx(0.05891627179, rel=1e-6)


def test_organoid_golden_value():
    v = screen(_load("organoid_case.yaml"), 1e-3 * STD)
    assert v.result == "PASS"
    assert _worst(v).delta_center == pytest.approx(0.52148565205, rel=1e-6)


def test_every_verdict_has_reason_and_provenance():
    for g in (STD, 1.62, 1e-3 * STD):
        v = screen(_load("reference_case.yaml"), g)
        assert v.result in ("PASS", "MARGINAL", "FAIL")
        assert len(v.limiting_reason) > 20          # a FAIL is argued as well as a PASS
        assert v.provenance and all(isinstance(s, str) and s for s in v.provenance.values())


def test_verdict_is_deterministic():
    e = _load("reference_case.yaml")
    a, b = screen(e, 1e-3 * STD), screen(e, 1e-3 * STD)
    assert [o.delta_center for o in a.observables] == [o.delta_center for o in b.observables]
    assert a.result == b.result and a.uncertainty_threshold == b.uncertainty_threshold


def test_change_grows_monotonically_as_gravity_falls():
    e = _load("reference_case.yaml")
    deltas = [_worst(screen(e, f * STD)).delta_center for f in (1.0, 0.5, 0.1, 0.01, 1e-3)]
    assert all(d2 > d1 for d1, d2 in zip(deltas, deltas[1:]))


def test_centre_concentration_stays_in_physical_bounds():
    v = screen(_load("organoid_case.yaml"), 1e-3 * STD)
    for o in v.observables:
        assert 0.0 <= o.center_concentration_target_g and 0.0 <= o.surface_concentration_target_g


def test_marginal_band_changes_classification():
    e = _load("reference_case.yaml")
    assert screen(e, 1.62, marginal_band=1.5).result == "MARGINAL"
    assert screen(e, 1.62, marginal_band=1.1).result == "PASS"
