import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest

from gravitas.validation import run_biological_benchmark, EXPERIMENTAL_DATA


def test_biological_benchmark_execution_and_metrics():
    metrics = run_biological_benchmark(n_cells=100)
    
    # 1. Structural physical assertions
    assert len(metrics.r_over_R) == len(EXPERIMENTAL_DATA["r_over_R"])
    assert np.all(metrics.C_pred >= 0.0)
    assert np.all(np.diff(metrics.C_pred) >= 0.0), "Oxygen concentration must increase monotonically toward the boundary"
    
    # 2. Centre concentration agreement: within 10% of observed
    c0_obs = metrics.C_obs[0]
    c0_pred = metrics.C_pred[0]
    assert abs(c0_pred - c0_obs) / c0_obs < 0.10, f"Centre concentration mismatch: {c0_pred:.4f} vs {c0_obs:.4f}"
    
    # 3. Overall error bounds
    assert metrics.rmse < 0.040, f"RMSE too large: {metrics.rmse:.4f} mol/m^3"
    assert metrics.mae < 0.030, f"MAE too large: {metrics.mae:.4f} mol/m^3"
    assert metrics.nrmse < 0.25, f"NRMSE too large: {metrics.nrmse:.2%}"


def test_numerical_error_is_subordinate_to_experimental_uncertainty():
    """Verify that spatial discretization error is orders of magnitude smaller than biological uncertainty."""
    metrics = run_biological_benchmark(n_cells=100)
    assert metrics.mesh_error_to_exp_error_ratio < 0.05, (
        f"Numerical error ratio {metrics.mesh_error_to_exp_error_ratio:.4f} is too large; "
        "numerical error must not confound biological validation."
    )
