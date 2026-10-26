#!/usr/bin/env python
"""Scientific PINN validation script (Step 8).

Validates the Physics-Informed Neural Network (PINN) surrogate against the
deterministic finite-difference numerical solver on an independent,
unseen held-out parameter test set.

Measures:
  1. MAE, RMSE, Max error across full concentration profiles C(R).
  2. Boundary condition residuals at R=0 (symmetry) and R=1 (Robin).
  3. Decision/verdict agreement (PASS/MARGINAL/FAIL classification).
  4. Hardware-matched runtime benchmark and actual speedup on identical CPU.

Outputs:
  - results/figures/fig_pinn_validation.pdf / .png
  - results/tables/tab_pinn_validation.tex
  - results/pinn_validation.json

Run: PYTHONPATH=src python scripts/validate_pinn.py results
"""
import sys, os, time, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import qmc

import torch
import torch.optim as optim

from gravitas.ai.pinn import ReactionDiffusionPINN, physics_loss
from gravitas.solver.reaction_diffusion import solve_steady_state
from gravitas.solver.discretisation import RadialGrid

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in (OUT, FIG, TAB):
    os.makedirs(d, exist_ok=True)

MODEL_PATH = "surrogate_pinn.pth"

print("=" * 65)
print("SCIENTIFIC PINN VALIDATION (STEP 8)")
print("=" * 65)

# 1. Train or Load Model
pinn = ReactionDiffusionPINN(hidden_layers=4, hidden_neurons=64)

if not os.path.exists(MODEL_PATH):
    print("Training surrogate PINN for validation (800 epochs, seed 42)...")
    torch.manual_seed(42)
    optimizer = optim.Adam(pinn.parameters(), lr=1e-3)
    sampler = qmc.LatinHypercube(d=4, seed=42)
    l_bounds = [0.0, 0.01, 0.01, 0.1]
    u_bounds = [1.0, 50.0, 5.0, 50.0]
    
    t_train_start = time.perf_counter()
    for ep in range(800):
        optimizer.zero_grad()
        sample = qmc.scale(sampler.random(n=500), l_bounds, u_bounds)
        R = torch.tensor(sample[:, 0:1], dtype=torch.float32, requires_grad=True)
        al = torch.tensor(sample[:, 1:2], dtype=torch.float32)
        K = torch.tensor(sample[:, 2:3], dtype=torch.float32)
        H = torch.tensor(sample[:, 3:4], dtype=torch.float32)
        loss, _, _, _ = physics_loss(pinn, R, al, K, H)
        loss.backward()
        optimizer.step()
    t_train = time.perf_counter() - t_train_start
    torch.save(pinn.state_dict(), MODEL_PATH)
    print(f"Training completed in {t_train:.2f} s. Saved to {MODEL_PATH}.")
else:
    print(f"Loading existing surrogate PINN from {MODEL_PATH}...")
    pinn.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    t_train = None

pinn.eval()

# 2. Generate Independent Held-Out Test Set (Unseen Parameter Combinations)
N_HELDOUT = 60
sampler_test = qmc.LatinHypercube(d=3, seed=99999)  # completely independent seed
# Sample (alpha, K, H)
test_params = qmc.scale(sampler_test.random(n=N_HELDOUT), [0.05, 0.05, 0.2], [40.0, 4.0, 40.0])

grid = RadialGrid(n_cells=100)
R_nodes = grid.R
R_tensor = torch.tensor(R_nodes, dtype=torch.float32).unsqueeze(1)

rmses = []
maes = []
max_errors = []
bc_core_res = []
bc_surf_res = []
verdict_matches = 0

t_num_total = 0.0
t_pinn_total = 0.0

cases_evaluated = 0

for i in range(N_HELDOUT):
    al_val, K_val, H_val = float(test_params[i, 0]), float(test_params[i, 1]), float(test_params[i, 2])
    
    # Numerical solve (Ground Truth)
    t0 = time.perf_counter()
    try:
        _, C_num = solve_steady_state(alpha=al_val, K=K_val, H=H_val, n_cells=100)
    except RuntimeError:
        continue  # skip non-converged numerical solve
    t_num_total += (time.perf_counter() - t0)
    
    # PINN Inference
    al_tensor = torch.full_like(R_tensor, al_val)
    K_tensor = torch.full_like(R_tensor, K_val)
    H_tensor = torch.full_like(R_tensor, H_val)
    
    t0 = time.perf_counter()
    with torch.no_grad():
        C_pinn = pinn(R_tensor, al_tensor, K_tensor, H_tensor).squeeze().numpy()
    t_pinn_total += (time.perf_counter() - t0)
    
    # Profile errors
    diff = C_pinn - C_num
    rmses.append(float(np.sqrt(np.mean(diff ** 2))))
    maes.append(float(np.mean(np.abs(diff))))
    max_errors.append(float(np.max(np.abs(diff))))
    
    # Boundary Condition Residuals
    # Core (R=0): dC/dR = 0
    R_0 = torch.zeros(1, 1, requires_grad=True)
    C_0 = pinn(R_0, torch.tensor([[al_val]]), torch.tensor([[K_val]]), torch.tensor([[H_val]]))
    dC0 = torch.autograd.grad(C_0, R_0, create_graph=False)[0].item()
    bc_core_res.append(abs(dC0))
    
    # Surface (R=1): dC/dR - H*(1 - C) = 0
    R_1 = torch.ones(1, 1, requires_grad=True)
    C_1 = pinn(R_1, torch.tensor([[al_val]]), torch.tensor([[K_val]]), torch.tensor([[H_val]]))
    dC1 = torch.autograd.grad(C_1, R_1, create_graph=False)[0].item()
    bc_surf_res.append(abs(dC1 - H_val * (1.0 - C_1.item())))
    
    # Compare Decision (e.g. Center Concentration drop vs threshold 0.05)
    # Using 1g reference C(0) vs target g H/2
    num_drop = abs(C_num[0] - 1.0)
    pinn_drop = abs(C_pinn[0] - 1.0)
    # Threshold at 5% drop
    num_pass = num_drop > 0.05
    pinn_pass = pinn_drop > 0.05
    if num_pass == pinn_pass:
        verdict_matches += 1
        
    cases_evaluated += 1

mean_rmse = float(np.mean(rmses))
mean_mae = float(np.mean(maes))
mean_max_err = float(np.max(max_errors))
mean_bc_core = float(np.mean(bc_core_res))
mean_bc_surf = float(np.mean(bc_surf_res))
decision_acc = float(verdict_matches / cases_evaluated) * 100.0

avg_num_time_ms = (t_num_total / cases_evaluated) * 1e3
avg_pinn_time_ms = (t_pinn_total / cases_evaluated) * 1e3
speedup = avg_num_time_ms / max(avg_pinn_time_ms, 1e-6)

print(f"Held-out test cases evaluated: {cases_evaluated}/{N_HELDOUT}")
print(f"Mean Profile RMSE:             {mean_rmse:.4f}")
print(f"Mean Profile MAE:              {mean_mae:.4f}")
print(f"Max Pointwise Error:           {mean_max_err:.4f}")
print(f"Mean Core BC Residual:         {mean_bc_core:.4f}")
print(f"Mean Surface BC Residual:      {mean_bc_surf:.4f}")
print(f"Decision Consistency:          {decision_acc:.1f}%")
print("-" * 65)
print(f"Numerical Solver Mean Time:    {avg_num_time_ms:.2f} ms")
print(f"PINN Surrogate Mean Time:      {avg_pinn_time_ms:.2f} ms")
print(f"Hardware-Matched Speedup:      {speedup:.1f}x")
print("=" * 65)

# 3. Figures: Parity Plot & Error Distribution
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.8, 3.0))

# Parity plot for first 6 test cases
for k in range(min(5, cases_evaluated)):
    al_val, K_val, H_val = float(test_params[k, 0]), float(test_params[k, 1]), float(test_params[k, 2])
    _, C_num = solve_steady_state(alpha=al_val, K=K_val, H=H_val, n_cells=100)
    with torch.no_grad():
        C_pinn = pinn(R_tensor, torch.full_like(R_tensor, al_val),
                      torch.full_like(R_tensor, K_val), torch.full_like(R_tensor, H_val)).squeeze().numpy()
    ax1.plot(C_num, C_pinn, "o", ms=3, alpha=0.6, label=f"Case {k+1}")

ax1.plot([0, 1], [0, 1], "k--", lw=0.9, label="Perfect agreement")
ax1.set_xlabel("Numerical Solver $C(R)$")
ax1.set_ylabel("PINN Surrogate $C(R)$")
ax1.set_title("Parity: Solver vs PINN")
ax1.legend(loc="upper left", fontsize=6.5)
ax1.grid(alpha=0.3)

# Error histogram
ax2.hist(rmses, bins=12, color="teal", edgecolor="black", alpha=0.7)
ax2.axvline(mean_rmse, color="red", ls="--", label=f"Mean RMSE = {mean_rmse:.3f}")
ax2.set_xlabel("Profile RMSE")
ax2.set_ylabel("Held-Out Test Cases")
ax2.set_title("Held-Out Generalization Error")
ax2.legend(loc="upper right", fontsize=7.5)
ax2.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig_pinn_validation.pdf"))
fig.savefig(os.path.join(FIG, "fig_pinn_validation.png"), dpi=200)
plt.close(fig)

# 4. LaTeX Summary Table
tab_tex = (
    "\\begin{table}[htbp]\n\\centering\\small\n"
    "\\caption{PINN surrogate validation on an independent held-out test set ($N=" f"{cases_evaluated}" "$ unseen parameter combinations). "
    "All timings measured on identical CPU hardware.}\\label{tab:pinn_validation}\n"
    "\\begin{tabular}{|l|c|l|}\n\\hline\n"
    "\\textbf{Metric} & \\textbf{Measured Value} & \\textbf{Interpretation} \\\\ \\hline\n"
    f"Mean Profile RMSE & {mean_rmse:.4f} & Overall solution accuracy \\\\ \\hline\n"
    f"Mean Profile MAE & {mean_mae:.4f} & Mean absolute deviation \\\\ \\hline\n"
    f"Max Pointwise Error & {mean_max_err:.4f} & Worst-case local error \\\\ \\hline\n"
    f"Core Symmetry Residual ($R=0$) & {mean_bc_core:.4f} & $|\\partial C/\\partial R|_{{R=0}}$ constraint \\\\ \\hline\n"
    f"Surface Robin Residual ($R=1$) & {mean_bc_surf:.4f} & Boundary flux constraint \\\\ \\hline\n"
    f"Decision Consistency & {decision_acc:.1f}\\% & Agreement with numerical screening \\\\ \\hline\n"
    f"Numerical Solver Runtime (CPU) & {avg_num_time_ms:.1f}~ms & Ground truth solver latency \\\\ \\hline\n"
    f"PINN Inference Runtime (CPU) & {avg_pinn_time_ms:.2f}~ms & Surrogate forward pass \\\\ \\hline\n"
    f"Measured Speedup & {speedup:.1f}$\\times$ & Reproducible same-hardware speedup \\\\ \\hline\n"
    "\\end{tabular}\n\\end{table}\n"
)
with open(os.path.join(TAB, "tab_pinn_validation.tex"), "w") as fh:
    fh.write(tab_tex)

# 5. JSON Export
results_json = {
    "n_held_out": cases_evaluated,
    "mean_rmse": mean_rmse,
    "mean_mae": mean_mae,
    "max_pointwise_error": mean_max_err,
    "core_bc_residual": mean_bc_core,
    "surface_bc_residual": mean_bc_surf,
    "decision_accuracy_pct": decision_acc,
    "numerical_mean_time_ms": avg_num_time_ms,
    "pinn_mean_time_ms": avg_pinn_time_ms,
    "measured_speedup_factor": speedup,
    "status": "Implemented; validation quantified on held-out set",
}
with open(os.path.join(OUT, "pinn_validation.json"), "w") as fh:
    json.dump(results_json, fh, indent=2)

print("PINN validation report, table, and figures generated successfully.")
