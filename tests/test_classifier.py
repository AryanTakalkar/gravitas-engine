import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import pytest
from gravitas import Experiment
from gravitas.classifier import classify

STD = 9.80665
EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def _load(name):
    return Experiment.from_yaml(os.path.join(EXAMPLES, name))


def test_four_mechanisms_reported():
    names = {m.name for m in classify(_load("reference_case.yaml"), STD, 1e-3 * STD)}
    assert names == {"buoyancy_convection", "sedimentation", "multiphase_separation", "hydrostatic_gradient"}


def test_ranked_by_relative_change_descending():
    res = classify(_load("reference_case.yaml"), STD, 1e-3 * STD)
    changes = [m.relative_change for m in res]
    assert changes == sorted(changes, reverse=True)


def test_all_indicators_scale_linearly_with_g():
    for m in classify(_load("reference_case.yaml"), STD, 1e-3 * STD):
        assert m.value_target_g == pytest.approx(1e-3 * m.value_1g, rel=1e-12)
        assert m.relative_change == pytest.approx(1 - 1e-3, rel=1e-12)


def test_reference_case_only_sedimentation_crosses_boundary():
    crossed = {m.name for m in classify(_load("reference_case.yaml"), STD, 1e-3 * STD) if m.crossed}
    assert crossed == {"sedimentation"}


def test_grashof_is_below_one_at_standard_gravity():
    # no buoyancy-convection regime to lose, even at 1 g (documented finding)
    for f in ("reference_case.yaml", "organoid_case.yaml"):
        gr = [m for m in classify(_load(f), STD, 1e-3 * STD) if m.name == "buoyancy_convection"][0]
        assert gr.value_1g < 1.0


def test_no_change_when_target_equals_standard():
    for m in classify(_load("reference_case.yaml"), STD, STD):
        assert m.relative_change == 0.0 and not m.crossed
