import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
import pytest
from scipy.special import i0, i1

from gravitas.solver.discretisation import RadialGrid, laplacian_1d, SUPPORTED_GEOMETRIES
from gravitas.solver.reaction_diffusion import solve_steady_state, solve_transient


def exact_zero_order(R, alpha, H, geometry):
    """Exact zero-order solution for 1D diffusion with constant uptake alpha."""
    p = {"slab": 0, "cylinder": 1, "sphere": 2}[geometry]
    return 1.0 - alpha / ((p + 1) * H) - (alpha / (2.0 * (p + 1))) * (1.0 - R ** 2)


def exact_first_order(R, phi, H, geometry):
    """Exact first-order solution for linear reaction-diffusion d^2C/dR^2 + (p/R)dC/dR = phi^2 C."""
    if geometry == "slab":
        denom = phi * np.sinh(phi) + H * np.cosh(phi)
        return (H / denom) * np.cosh(phi * R)
    elif geometry == "cylinder":
        denom = phi * i1(phi) + H * i0(phi)
        return (H / denom) * i0(phi * R)
    elif geometry == "sphere":
        denom = phi * np.cosh(phi) + (H - 1.0) * np.sinh(phi)
        # Avoid 0/0 at R=0 using limit sinh(phi*R)/R -> phi
        with np.errstate(divide='ignore', invalid='ignore'):
            res = (H / denom) * np.sinh(phi * R) / R
        res[R == 0] = (H / denom) * phi
        return res


@pytest.mark.parametrize("geom", ["slab", "cylinder", "sphere"])
def test_zero_order_analytical_limits_all_geometries(geom):
    alpha = 0.15
    K = 1e-8  # saturates to zero-order
    H = 2.5
    R, C_num = solve_steady_state(alpha=alpha, K=K, H=H, n_cells=100, geometry=geom)
    C_exact = exact_zero_order(R, alpha, H, geom)
    rel_err = np.max(np.abs(C_num - C_exact) / C_exact)
    assert rel_err < 1e-3, f"Failed for {geom}: rel_err={rel_err}"


@pytest.mark.parametrize("geom", ["slab", "cylinder", "sphere"])
def test_first_order_analytical_limits_all_geometries(geom):
    # For first-order limit, alpha = K = 1e6 -> phi = sqrt(alpha/K) = 1.0
    alpha = 1e6
    K = 1e6
    phi = 1.0
    H = 3.0
    R, C_num = solve_steady_state(alpha=alpha, K=K, H=H, n_cells=100, geometry=geom)
    C_exact = exact_first_order(R, phi, H, geom)
    rel_err = np.max(np.abs(C_num - C_exact) / C_exact)
    assert rel_err < 1e-3, f"Failed for {geom}: rel_err={rel_err}"


@pytest.mark.parametrize("geom", ["slab", "cylinder", "sphere"])
def test_mesh_convergence_all_geometries(geom):
    alpha = 0.2
    K = 1e-8
    H = 2.0
    errs = []
    ns = [25, 50, 100]
    for n in ns:
        R, C = solve_steady_state(alpha=alpha, K=K, H=H, n_cells=n, geometry=geom)
        C_ex = exact_zero_order(R, alpha, H, geom)
        errs.append(np.max(np.abs(C - C_ex)))
    # Check monotonic convergence with rate ~ 2 (O(h^2))
    p1 = np.log2(errs[0] / errs[1])
    p2 = np.log2(errs[1] / errs[2])
    assert p1 > 1.8 and p2 > 1.8


def test_transient_slab_and_cylinder_relax_to_steady():
    for geom in ["slab", "cylinder"]:
        tau, C, _ = solve_transient(alpha=0.5, K=0.3, H=4.0, tau_end=25.0, n_cells=40, n_output=20, geometry=geom)
        _, C_ss = solve_steady_state(alpha=0.5, K=0.3, H=4.0, n_cells=40, geometry=geom)
        assert np.max(np.abs(C[-1] - C_ss)) < 1e-5
