"""Validate the nonlinear Michaelis-Menten solver against two exact
closed-form limits: zero-order kinetics (large K) and first-order
kinetics (small alpha/large K in the other direction -- see below).

This is the core scientific validation of the numerical core (Objective 5 /
"agreement with analytical solutions" KPI).
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pytest

from gravitas.solver.reaction_diffusion import solve_steady_state
from analytical_limits import zero_order_exact, first_order_exact


def test_zero_order_limit():
    """When K >> C everywhere (achieved here with a very large K and alpha/K
    tuned so the sink is effectively constant), the MM term alpha*C/(K+C)
    ~ (alpha/K)*C ... but we want the *zero-order* limit (C >> K), which is
    achieved with K -> 0 (small K relative to C ~ O(1))."""
    alpha = 0.5
    H = 5.0
    K = 1e-6  # K << C (C ~ O(1)) => zero-order limit: alpha*C/(K+C) ~ alpha

    R, C_numeric = solve_steady_state(alpha=alpha, K=K, H=H, n_cells=200)
    C_exact = zero_order_exact(R, alpha=alpha, H=H)

    max_rel_err = np.max(np.abs(C_numeric - C_exact) / np.abs(C_exact))
    assert max_rel_err < 5e-3, f"zero-order limit mismatch: {max_rel_err:.2e}"


def test_first_order_limit():
    """When K >> C everywhere (large K), alpha*C/(K+C) ~ (alpha/K)*C: first-order."""
    alpha = 0.5
    K = 500.0   # K >> C (C ~ O(1)) => first-order limit
    H = 5.0

    R, C_numeric = solve_steady_state(alpha=alpha, K=K, H=H, n_cells=200)
    C_exact = first_order_exact(R, alpha=alpha, K=K, H=H)

    max_rel_err = np.max(np.abs(C_numeric - C_exact) / np.abs(C_exact))
    assert max_rel_err < 5e-3, f"first-order limit mismatch: {max_rel_err:.2e}"


def test_solution_is_physical():
    """Concentration must stay within [0, 1] (bounded by bulk value) and be
    monotonically non-increasing toward the centre for a pure sink."""
    R, C = solve_steady_state(alpha=1.0, K=1.0, H=5.0, n_cells=100)
    assert np.all(C >= -1e-6), "concentration went negative"
    assert np.all(C <= 1.0 + 1e-6), "concentration exceeded bulk value"
    assert np.all(np.diff(C) >= -1e-6), "concentration should increase outward for a pure sink"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
