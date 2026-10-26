"""Biological benchmark validation module.

Compares GRAVITAS reaction-diffusion predictions directly against published,
spatially-resolved microelectrode oxygen measurements from:
Müller-Klieser, W., Freyer, J. P., & Sutherland, R. M. (1986).
Influence of glucose and oxygen supply conditions on the oxygenation of multicellular spheroids.
British Journal of Cancer, 53(3), 345-353.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
import numpy as np

from ..experiment import Experiment
from ..solver.kinetics import dimensionless_groups
from ..solver.reaction_diffusion import solve_steady_state

# Digitized experimental microelectrode measurements (Müller-Klieser et al. 1986, Figure 3 & Table 1)
# 200 um radius EMT6 spheroids at 37 C in atmospheric equilibrium
EXPERIMENTAL_DATA = {
    "citation": "Müller-Klieser, Freyer, & Sutherland (1986), Br. J. Cancer 53:345-353",
    "r_over_R": np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0]),
    # Measured oxygen partial pressure (mmHg) and standard deviation
    "pO2_mmHg": np.array([19.0, 28.0, 54.0, 88.0, 118.0, 138.0]),
    "pO2_std_mmHg": np.array([4.0, 5.0, 7.0, 8.0, 6.0, 4.0]),
    # Converted concentration in mol/m^3 (Henry's law at 37 C: ~1.30e-3 mol/m^3 per mmHg)
    "C_obs_mol_m3": np.array([0.0247, 0.0364, 0.0702, 0.1145, 0.1536, 0.1796]),
    "C_std_mol_m3": np.array([0.0052, 0.0065, 0.0091, 0.0104, 0.0078, 0.0052]),
}


@dataclass
class BenchmarkMetrics:
    r_over_R: np.ndarray
    C_obs: np.ndarray
    C_std: np.ndarray
    C_pred: np.ndarray
    residuals: np.ndarray
    rmse: float
    mae: float
    nrmse: float
    max_error: float
    reduced_chi_squared: float
    all_within_95_ci: bool
    mesh_error_to_exp_error_ratio: float


def run_biological_benchmark(config_path: str | None = None, n_cells: int = 100) -> BenchmarkMetrics:
    """Run the biological benchmark and compute validation metrics."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_path = os.path.join(base_dir, "examples", "benchmark_muller_klieser_1986.yaml")

    exp = Experiment.from_yaml(config_path)
    species = exp.species[0]  # oxygen

    alpha, K = dimensionless_groups(species, exp.aggregate_radius)
    H = exp.membrane_permeability_1g["oxygen"] * exp.aggregate_radius / species.diffusivity

    # Solve steady state
    R_grid, C_dimless = solve_steady_state(alpha=alpha, K=K, H=H, n_cells=n_cells)
    C_dim_solver = C_dimless * species.bulk_concentration

    # Interpolate solver prediction onto the exact experimental measurement locations
    r_obs = EXPERIMENTAL_DATA["r_over_R"]
    C_obs = EXPERIMENTAL_DATA["C_obs_mol_m3"]
    C_std = EXPERIMENTAL_DATA["C_std_mol_m3"]

    C_pred = np.interp(r_obs, R_grid, C_dim_solver)
    residuals = C_pred - C_obs

    n = len(r_obs)
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    mae = float(np.mean(np.abs(residuals)))
    delta_range = float(C_obs[-1] - C_obs[0])
    nrmse = float(rmse / delta_range)
    max_error = float(np.max(np.abs(residuals)))

    # Reduced chi-squared (goodness of fit relative to experimental error)
    chi2 = float(np.sum((residuals / C_std) ** 2))
    dof = max(n - 2, 1)
    reduced_chi2 = chi2 / dof

    # 95% Confidence Interval check (within 1.96 standard deviations)
    within_ci = np.abs(residuals) <= 1.96 * C_std
    all_within_95 = bool(np.all(within_ci))

    # Mesh vs experimental error: check numerical error by doubling mesh
    _, C_fine = solve_steady_state(alpha=alpha, K=K, H=H, n_cells=n_cells * 2)
    mesh_err = float(abs(C_dimless[0] - C_fine[0]) * species.bulk_concentration)
    mesh_to_exp_ratio = mesh_err / C_std[0]

    return BenchmarkMetrics(
        r_over_R=r_obs,
        C_obs=C_obs,
        C_std=C_std,
        C_pred=C_pred,
        residuals=residuals,
        rmse=rmse,
        mae=mae,
        nrmse=nrmse,
        max_error=max_error,
        reduced_chi_squared=reduced_chi2,
        all_within_95_ci=all_within_95,
        mesh_error_to_exp_error_ratio=mesh_to_exp_ratio,
    )
