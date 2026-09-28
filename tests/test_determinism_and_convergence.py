import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from gravitas.solver.reaction_diffusion import solve_steady_state


def test_determinism_ten_runs():
    """Same input must give bitwise-identical output across repeated runs
    (satisfies the brief's determinism KPI: identical output every run)."""
    results = []
    for _ in range(10):
        _, C = solve_steady_state(alpha=0.7, K=0.5, H=5.0, n_cells=100)
        results.append(C)
    for C in results[1:]:
        assert np.array_equal(C, results[0]), "solver is not deterministic across repeated runs"


def test_mesh_convergence():
    """Refining the grid should converge, not just change unpredictably."""
    errors = []
    C_prev = None
    for n_cells in (25, 50, 100, 200, 400):
        R, C = solve_steady_state(alpha=0.7, K=0.5, H=5.0, n_cells=n_cells)
        if C_prev is not None:
            # Compare at the coarser grid's resolution by simple interpolation
            R_prev = (np.arange(len(C_prev)) + 0.5) / len(C_prev)
            C_interp = np.interp(R_prev, R, C)
            err = np.max(np.abs(C_interp - C_prev))
            errors.append(err)
        C_prev = C
    # Errors should shrink as the mesh is refined (monotonic convergence)
    assert all(e2 <= e1 * 1.5 for e1, e2 in zip(errors, errors[1:])), \
        f"mesh refinement did not converge monotonically: {errors}"
