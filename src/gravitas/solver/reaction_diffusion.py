"""1D transient and steady-state reaction-diffusion solver.

Supports:
  - "sphere":   spherical cell aggregate / organoid / spheroid
  - "cylinder": cylindrical bioprinted filament / hollow fiber
  - "slab":     planar tissue slice / bilayer construct

Solves, in dimensionless form:
    dC/dtau = Laplacian_1d(C, geometry) - alpha * C / (K + C)
with symmetry at R=0 and Robin (membrane/film) condition at R=1.

Deterministic: fixed grid, fixed solver tolerances, no randomness.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

from .discretisation import RadialGrid, laplacian_1d, laplacian_spherical, SUPPORTED_GEOMETRIES
from .kinetics import reaction_term


def _rhs(t, C, grid: RadialGrid, alpha: float, K: float, H: float, geometry: str = "sphere"):
    return laplacian_1d(C, grid, H, geometry=geometry) - reaction_term(C, alpha, K)


def solve_transient(
    alpha: float,
    K: float,
    H: float,
    tau_end: float,
    n_cells: int = 100,
    n_output: int = 50,
    geometry: str = "sphere",
):
    """Integrate the dimensionless PDE from a uniform initial condition C=1.

    Returns (tau_array, C_array, grid) with C_array shape (n_output, n_cells).
    """
    grid = RadialGrid(n_cells=n_cells)
    C0 = np.ones(n_cells)
    tau_eval = np.linspace(0.0, tau_end, n_output)

    sol = solve_ivp(
        _rhs,
        t_span=(0.0, tau_end),
        y0=C0,
        args=(grid, alpha, K, H, geometry),
        method="BDF",       # implicit, stable for the stiff reaction term
        t_eval=tau_eval,
        rtol=1e-8,
        atol=1e-10,
    )
    if not sol.success:
        raise RuntimeError(f"solver failed to converge: {sol.message}")
    return sol.t, sol.y.T, grid


def solve_steady_state(alpha: float, K: float, H: float, n_cells: int = 100, geometry: str = "sphere"):
    """Solve the steady-state profile directly via a nonlinear root find.

    Steady state satisfies Laplacian_1d(C, geometry) - alpha*C/(K+C) = 0 everywhere.

    Deterministic multi-stage strategy.
    """
    grid = RadialGrid(n_cells=n_cells)

    def residual(C):
        return laplacian_1d(C, grid, H, geometry=geometry) - reaction_term(C, alpha, K)

    guess_value = H / (H + 1.0)
    initial_guesses = [
        np.full(n_cells, guess_value),
        np.ones(n_cells),
        np.full(n_cells, 0.5),
        np.linspace(0.1, 1.0, n_cells),
    ]

    last_message = ""
    for C0 in initial_guesses:
        sol = root(residual, C0, method="hybr", tol=1e-10)
        if sol.success:
            return grid.R, sol.x
        last_message = sol.message

    raise RuntimeError(
        f"steady-state solve failed to converge after {len(initial_guesses)} "
        f"deterministic fallback attempts: {last_message}"
    )


def solve_steady_state_pinn(alpha: float, K: float, H: float, n_cells: int = 100, model_path: str = "surrogate_pinn.pth"):
    """
    Solve the steady-state profile INSTANTLY using the trained Physics-Informed Neural Network (PINN).
    This acts as a drop-in replacement for solve_steady_state but relies on the AI surrogate.
    """
    import torch
    from ..ai.pinn import ReactionDiffusionPINN
    
    grid = RadialGrid(n_cells=n_cells)
    
    pinn = ReactionDiffusionPINN()
    try:
        pinn.load_state_dict(torch.load(model_path, weights_only=True))
    except FileNotFoundError:
        raise RuntimeError(f"PINN model not found at {model_path}. Please run train_pinn.py first.")
        
    pinn.eval()
    
    R_tensor = torch.tensor(grid.R, dtype=torch.float32).unsqueeze(1)
    alpha_tensor = torch.full_like(R_tensor, alpha)
    K_tensor = torch.full_like(R_tensor, K)
    H_tensor = torch.full_like(R_tensor, H)
    
    with torch.no_grad():
        C_pred = pinn(R_tensor, alpha_tensor, K_tensor, H_tensor).squeeze().numpy()
        
    return grid.R, C_pred
