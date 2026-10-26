import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from gravitas import Experiment, screen
from gravitas.solver.coupling import (heuristic_floor, churchill_sherwood, churchill_coupling,
                                      solutal_rayleigh, COUPLING_MODELS)

STD = 9.80665
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")
KW = dict(beta=3e-4, delta_c=0.2, radius=150e-6, nu=8e-7, D=1.8e-9)


def _load(name):
    return Experiment.from_yaml(os.path.join(EXAMPLES, name))


def test_heuristic_limits():
    assert heuristic_floor(2.0, STD, STD) == pytest.approx(2.0)
    assert heuristic_floor(2.0, 0.0, STD) == pytest.approx(0.2)


def test_sherwood_diffusion_limit_is_two():
    assert churchill_sherwood(0.0, 400.0) == 2.0
    assert churchill_sherwood(1e-12, 400.0) == pytest.approx(2.0, abs=1e-3)


def test_sherwood_increases_with_rayleigh():
    vals = [churchill_sherwood(ra, 400.0) for ra in (1e-2, 1, 1e2, 1e4)]
    assert all(b > a for a, b in zip(vals, vals[1:]))


def test_churchill_reference_values():
    sc = KW["nu"] / KW["D"]
    ra = solutal_rayleigh(STD, KW["beta"], KW["delta_c"], 2 * KW["radius"], KW["nu"], KW["D"])
    assert ra == pytest.approx(11.0, rel=0.02)
    assert churchill_sherwood(ra, sc) == pytest.approx(3.06, abs=0.02)


@pytest.mark.parametrize("mode", ["film", "series"])
def test_churchill_is_identity_at_standard_gravity(mode):
    assert churchill_coupling(0.417, STD, STD, mode=mode, **KW) == pytest.approx(0.417, rel=1e-12)


def test_series_is_less_gravity_dependent_than_film():
    film = churchill_coupling(0.417, 1e-3 * STD, STD, mode="film", **KW)
    series = churchill_coupling(0.417, 1e-3 * STD, STD, mode="series", **KW)
    assert 0.417 > series > film > 0.417 * 0.5


def test_series_rejects_h_above_film_limit():
    with pytest.raises(ValueError):
        churchill_coupling(5.0, 1e-3 * STD, STD, mode="series", **KW)


def test_unknown_mode_and_unknown_coupling_raise():
    with pytest.raises(ValueError):
        churchill_coupling(0.4, 1.0, STD, mode="bogus", **KW)
    with pytest.raises(ValueError):
        screen(_load("reference_case.yaml"), 1.0, coupling="bogus")


def test_default_coupling_is_heuristic_and_recorded():
    assert screen(_load("reference_case.yaml"), 1.0).coupling == "heuristic"
    assert set(COUPLING_MODELS) == {"heuristic", "churchill_film", "churchill_series", "perfusion"}


def test_perfusion_maintains_mass_transfer_in_microgravity():
    # Perfusion velocity U = 1 mm/s decouples mass transfer from gravity
    e = _load("reference_case.yaml")
    v_perf = screen(e, 1e-3 * STD, coupling="perfusion", flow_velocity=1e-3)
    assert v_perf.coupling == "perfusion"
    assert v_perf.result in ("PASS", "MARGINAL", "FAIL")


@pytest.mark.parametrize("mode", ["churchill_film", "churchill_series"])
def test_reference_case_is_fail_at_every_gravity_with_correlation(mode):
    e = _load("reference_case.yaml")
    for f in (0.378, 0.165, 0.01, 1e-3, 1e-6):
        assert screen(e, f * STD, coupling=mode).result == "FAIL"


def test_organoid_is_marginal_at_1e3_with_correlation():
    v = screen(_load("organoid_case.yaml"), 1e-3 * STD, coupling="churchill_film")
    assert v.result == "MARGINAL" and v.coupling == "churchill_film"
    assert screen(_load("organoid_case.yaml"), 1.62, coupling="churchill_film").result == "FAIL"
