#!/usr/bin/env python
"""Biological benchmark validation script.

Validates the spherical reaction-diffusion model against microelectrode
measurements from Müller-Klieser et al. (1986).
Generates:
  - results/figures/fig_benchmark_profiles.pdf / .png
  - results/figures/fig_benchmark_residuals.pdf / .png
  - results/tables/tab_benchmark_validation.tex
  - results/benchmark_validation.json

Run: PYTHONPATH=src python scripts/validate_biological_benchmark.py results
"""
import sys, os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gravitas.validation import run_biological_benchmark, EXPERIMENTAL_DATA

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in (OUT, FIG, TAB):
    os.makedirs(d, exist_ok=True)

metrics = run_biological_benchmark(n_cells=120)

print("=" * 60)
print("BIOLOGICAL BENCHMARK VALIDATION RESULTS")
print(f"Benchmark: {EXPERIMENTAL_DATA['citation']}")
print(f"RMSE:        {metrics.rmse:.5f} mol/m^3 ({metrics.rmse * 1e3:.2f} uM)")
print(f"MAE:         {metrics.mae:.5f} mol/m^3 ({metrics.mae * 1e3:.2f} uM)")
print(f"NRMSE:       {metrics.nrmse * 100:.2f}% (normalized by concentration span)")
print(f"Max Error:   {metrics.max_error:.5f} mol/m^3")
print(f"Reduced Chi^2: {metrics.reduced_chi_squared:.3f}")
print(f"All within 95% CI: {metrics.all_within_95_ci}")
print(f"Mesh error / Experimental error: {metrics.mesh_error_to_exp_error_ratio * 100:.3f}%")
print("=" * 60)

# 1. Figure: Measured vs Predicted Profiles
fig, ax = plt.subplots(figsize=(4.5, 3.2))
r_fine = np.linspace(0.0, 1.0, 100)
# Solve on fine grid for smooth line
from gravitas.solver.reaction_diffusion import solve_steady_state
from gravitas.solver.kinetics import dimensionless_groups
from gravitas.experiment import Experiment

exp = Experiment.from_yaml("examples/benchmark_muller_klieser_1986.yaml")
s = exp.species[0]
a, K = dimensionless_groups(s, exp.aggregate_radius)
H = exp.membrane_permeability_1g["oxygen"] * exp.aggregate_radius / s.diffusivity
rf, cf = solve_steady_state(alpha=a, K=K, H=H, n_cells=150)
c_dim_line = cf * s.bulk_concentration

ax.plot(rf * 200.0, c_dim_line * 1e3, "b-", label="GRAVITAS Model Prediction", lw=1.6)
ax.errorbar(metrics.r_over_R * 200.0, metrics.C_obs * 1e3, yerr=metrics.C_std * 1e3,
            fmt="ro", capsize=3, label="Microelectrode Data (Müller-Klieser 1986)", ms=5)

ax.set_xlabel(r"Radius $r$ ($\mu$m)")
ax.set_ylabel(r"Oxygen Concentration ($\mu$mol/L)")
ax.set_title("Biological Validation: Radial Oxygen Profile")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_benchmark_profiles.pdf"))
fig.savefig(os.path.join(FIG, "fig_benchmark_profiles.png"), dpi=200)
plt.close(fig)

# 2. Figure: Residuals & Uncertainty Band
fig2, ax2 = plt.subplots(figsize=(4.5, 2.5))
ax2.axhline(0, color="k", ls="--", lw=0.8)
ax2.fill_between(metrics.r_over_R * 200.0, -1.96 * metrics.C_std * 1e3, 1.96 * metrics.C_std * 1e3,
                 color="gray", alpha=0.2, label=r"$\pm 1.96\sigma$ (95% CI of experiment)")
ax2.plot(metrics.r_over_R * 200.0, metrics.residuals * 1e3, "o-", color="darkred", label="Residual ($C_{pred} - C_{obs}$)")

ax2.set_xlabel(r"Radius $r$ ($\mu$m)")
ax2.set_ylabel(r"Residual ($\mu$mol/L)")
ax2.set_title("Model Residuals vs Experimental Uncertainty")
ax2.grid(alpha=0.3)
ax2.legend(loc="best", fontsize=7.5)
fig2.tight_layout()
fig2.savefig(os.path.join(FIG, "fig_benchmark_residuals.pdf"))
fig2.savefig(os.path.join(FIG, "fig_benchmark_residuals.png"), dpi=200)
plt.close(fig2)

# 3. LaTeX Table
rows = ""
for i in range(len(metrics.r_over_R)):
    r = metrics.r_over_R[i]
    obs = metrics.C_obs[i] * 1e3
    std = metrics.C_std[i] * 1e3
    pred = metrics.C_pred[i] * 1e3
    res = metrics.residuals[i] * 1e3
    rows += f"{r:.1f} & {r*200.0:.0f} & {obs:.1f} $\\pm$ {std:.1f} & {pred:.1f} & {res:+.2f}\\\\ \\hline\n"

tab_tex = (
    "\\begin{table}[htbp]\n\\centering\\small\n"
    "\\caption{Biological benchmark validation: Comparison of GRAVITAS predicted oxygen concentrations "
    "against published microelectrode measurements in EMT6 spheroids (M\\\"uller-Klieser et al., 1986). "
    f"Overall RMSE = {metrics.rmse*1e3:.2f}~$\\mu$M, MAE = {metrics.mae*1e3:.2f}~$\\mu$M, reduced $\\chi^2={metrics.reduced_chi_squared:.2f}$.}}\n"
    "\\label{tab:benchmark_val}\n"
    "\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n"
    "$r/a$ & $r$ ($\\mu$m) & $C_{\\mathrm{obs}}$ ($\\mu$M) & $C_{\\mathrm{pred}}$ ($\\mu$M) & Residual ($\\mu$M)\\\\ \\hline\n"
    + rows +
    "\\end{tabular}\n\\end{table}\n"
)
with open(os.path.join(TAB, "tab_benchmark_validation.tex"), "w") as fh:
    fh.write(tab_tex)

# 4. JSON Summary
data_dict = {
    "citation": EXPERIMENTAL_DATA["citation"],
    "rmse_mol_m3": metrics.rmse,
    "rmse_uM": metrics.rmse * 1e3,
    "mae_mol_m3": metrics.mae,
    "mae_uM": metrics.mae * 1e3,
    "nrmse_pct": metrics.nrmse * 100,
    "max_error_uM": metrics.max_error * 1e3,
    "reduced_chi2": metrics.reduced_chi_squared,
    "all_within_95_ci": metrics.all_within_95_ci,
    "mesh_to_exp_error_pct": metrics.mesh_error_to_exp_error_ratio * 100,
    "points": [
        {"r_over_R": float(metrics.r_over_R[i]), "C_obs_uM": float(metrics.C_obs[i] * 1e3),
         "C_std_uM": float(metrics.C_std[i] * 1e3), "C_pred_uM": float(metrics.C_pred[i] * 1e3),
         "residual_uM": float(metrics.residuals[i] * 1e3)}
        for i in range(len(metrics.r_over_R))
    ]
}
with open(os.path.join(OUT, "benchmark_validation.json"), "w") as fh:
    json.dump(data_dict, fh, indent=2)

print("Validation figures, table, and JSON generated successfully.")
