import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from gravitas import Experiment
from gravitas.solver.discretisation import RadialGrid
from gravitas.solver.kinetics import dimensionless_groups, reaction_term
from gravitas.solver.reaction_diffusion import solve_steady_state, solve_transient
from gravitas.mechanisms.convection import grashof_number, convection_active
from gravitas.mechanisms.multiphase import bond_number, phase_separation_active, surface_tension_dominant
from gravitas.mechanisms.sedimentation import (stokes_settling_velocity, peclet_number, sedimentation_active)
from gravitas.mechanisms.hydrostatic import hydrostatic_pressure_difference, hydrostatic_active

EXAMPLES = os.path.join(os.path.dirname(__file__), "..", "examples")


def test_transient_relaxes_to_the_steady_state():
    tau, C, grid = solve_transient(alpha=0.7, K=0.5, H=5.0, tau_end=30.0, n_cells=50, n_output=20)
    _, C_ss = solve_steady_state(alpha=0.7, K=0.5, H=5.0, n_cells=50)
    assert C.shape == (20, 50) and tau[0] == 0.0
    assert np.max(np.abs(C[-1] - C_ss)) < 1e-6
    assert np.all(C >= -1e-9) and np.all(C <= 1.0 + 1e-9)


def test_transient_centre_concentration_decreases_monotonically_for_consumption():
    _, C, _ = solve_transient(alpha=2.0, K=0.1, H=3.0, tau_end=5.0, n_cells=40, n_output=15)
    centre = C[:, 0]
    assert np.all(np.diff(centre) <= 1e-9)


def test_grid_rejects_too_coarse_mesh():
    with pytest.raises(ValueError):
        RadialGrid(n_cells=5)


def test_grid_geometry():
    g = RadialGrid(n_cells=20)
    assert g.R_faces[0] == 0.0 and g.R_faces[-1] == pytest.approx(1.0)
    assert g.R[0] == pytest.approx(0.5 * g.dR)


def test_dimensionless_groups_match_definition():
    e = Experiment.from_yaml(os.path.join(EXAMPLES, "reference_case.yaml"))
    s = e.species[0]
    a, K = dimensionless_groups(s, e.aggregate_radius)
    assert a == pytest.approx(s.max_uptake_rate * e.aggregate_radius ** 2 / (s.diffusivity * s.bulk_concentration))
    assert K == pytest.approx(s.michaelis_constant / s.bulk_concentration)


def test_non_positive_bulk_concentration_is_rejected():
    class S:
        name = "x"; bulk_concentration = 0.0; max_uptake_rate = 1.0; diffusivity = 1.0; michaelis_constant = 1.0
    with pytest.raises(ValueError):
        dimensionless_groups(S(), 1e-4)


def test_reaction_term_saturates():
    c = np.array([0.0, 1.0, 100.0])
    r = reaction_term(c, alpha=2.0, K=0.5)
    assert r[0] == 0.0 and r[1] == pytest.approx(2.0 / 1.5) and r[2] < 2.0 and r[2] > r[1]


def test_mechanism_helpers_consistent_with_numbers():
    gr = grashof_number(9.80665, 3e-4, 0.2, 300e-6, 8e-7)
    assert convection_active(9.80665, 3e-4, 0.2, 300e-6, 8e-7) == (gr > 1.0)
    bo = bond_number(50.0, 9.80665, 300e-6, 0.07)
    assert phase_separation_active(50.0, 9.80665, 300e-6, 0.07) == (bo > 1.0)
    assert surface_tension_dominant(50.0, 9.80665, 300e-6, 0.07) == (not phase_separation_active(50.0, 9.80665, 300e-6, 0.07))
    v = stokes_settling_velocity(9.80665, 1055.0, 1005.0, 150e-6, 8e-7)
    assert v > 0 and stokes_settling_velocity(0.0, 1055.0, 1005.0, 150e-6, 8e-7) == 0.0
    pe = peclet_number(v, 300e-6, 1.8e-9)
    assert sedimentation_active(9.80665, 1055.0, 1005.0, 150e-6, 8e-7, 300e-6, 1.8e-9) == (pe > 1.0)
    assert hydrostatic_pressure_difference(1005.0, 9.80665, 300e-6) == pytest.approx(1005.0 * 9.80665 * 300e-6)
    assert hydrostatic_active(1005.0, 9.80665, 300e-6, threshold_pa=1.0) is True
