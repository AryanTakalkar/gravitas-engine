"""Physics-based check of the gravity-to-boundary coupling (replaces the assumed 10% floor).

External film coefficient from Churchill's laminar natural-convection correlation for a sphere,
Sh = 2 + 0.589 Ra^(1/4) / [1 + (0.469/Sc)^(9/16)]^(4/9)   (Ra <= 1e11, Sc >= 0.7; heat-mass analogy Pr -> Sc),
Ra = g*beta*c_inf*d^3/(nu*D) with d = 2a (same Delta-c convention as the classifier: Delta-c = species bulk concentration).
Model A (upper bound): the whole boundary resistance is the film -> H(g) = H1g * Sh(g)/Sh(g0).
Model B (series): H1g = membrane in series with the film at g0; H(g) = 1/(1/H_mem + 1/H_film(g)), H_film = Sh/2.
Run from repo root:  PYTHONPATH=src python scripts/physical_coupling.py results
"""
import sys, os, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from gravitas import Experiment
from gravitas.verdict import STANDARD_G as G0
from gravitas.solver.kinetics import dimensionless_groups
from gravitas.solver.reaction_diffusion import solve_steady_state

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
MOON, MARS = 1.62, 3.71
CASES = {"reference": Experiment.from_yaml("examples/reference_case.yaml"),
         "organoid": Experiment.from_yaml("examples/organoid_case.yaml")}
LAB = {"reference": "Reference (300 $\\mu$m aggregate)", "organoid": "Hepatic organoid (1 mm)"}

def sh(e, s, g):
    d = 2 * e.aggregate_radius
    Ra = g * e.medium_solutal_expansion * s.bulk_concentration * d ** 3 / (e.medium_kinematic_viscosity * s.diffusivity)
    Sc = e.medium_kinematic_viscosity / s.diffusivity
    return 2 + 0.589 * Ra ** 0.25 / (1 + (0.469 / Sc) ** (9 / 16)) ** (4 / 9), Ra

def H_model(e, s, H1, g, model):
    if model == "engine":
        gr = min(g / G0, 1.0); return H1 * (0.1 + 0.9 * gr)
    s0, _ = sh(e, s, G0); sg, _ = sh(e, s, g)
    if model == "A":
        return H1 * sg / s0
    Hf0, Hfg = s0 / 2, sg / 2
    Hm = 1 / (1 / H1 - 1 / Hf0)
    return 1 / (1 / Hm + 1 / Hfg)

def verdict(e, g, model):
    obs = {}
    for s in e.species:
        a, K = dimensionless_groups(s, e.aggregate_radius)
        H1 = e.membrane_permeability_1g[s.name] * e.aggregate_radius / s.diffusivity
        c1 = solve_steady_state(alpha=a, K=K, H=H1)[1][0]
        ct = solve_steady_state(alpha=a, K=K, H=H_model(e, s, H1, g, model))[1][0]
        obs[s.name] = (abs(c1 - ct), s.measurement_uncertainty * c1)
    w = max(obs, key=lambda k: obs[k][0]); d, thr = obs[w]
    return ("PASS" if d > 1.5 * thr else "MARGINAL" if d > thr else "FAIL"), w, d, thr

R = {}; rows = ""
LV = [("0.378 (Mars)", MARS), ("0.165 (Moon)", MOON), ("$10^{-3}$", 1e-3 * G0), ("$\\to 0$", 1e-9 * G0)]
for name, e in CASES.items():
    s = e.species[0]; H1 = e.membrane_permeability_1g["oxygen"] * e.aggregate_radius / s.diffusivity
    s0, Ra0 = sh(e, s, G0)
    R[name] = dict(Ra_1g=Ra0, Sh_1g=s0, H1g=H1, Hfilm_1g=s0 / 2, rows=[])
    rows += f"\\multicolumn{{6}}{{|l|}}{{\\textit{{{LAB[name]}: oxygen $\\mathrm{{Ra}}(g_0)={Ra0:.3g}$, $\\mathrm{{Sh}}(g_0)={s0:.2f}$}}}}\\\\ \\hline\n"
    for mdl, mlab in [("engine", "engine rule"), ("A", "film only (A)"), ("B", "membrane + film (B)")]:
        for glab, g in LV:
            v, w, d, thr = verdict(e, g, mdl)
            ratio_H = H_model(e, s, H1, g, mdl) / H1
            rows += f" & {mlab} & {glab} & {ratio_H:.3f} & {d/thr:.2f} & {v}\\\\ \\hline\n"
            R[name]["rows"].append(dict(model=mdl, g=g / G0, H_ratio=ratio_H, delta=d, theta=thr, ratio=d / thr, verdict=v, limiting=w))
os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
os.makedirs(os.path.join(OUT, "figures"), exist_ok=True)
open(os.path.join(OUT, "tables", "tab_physical.tex"), "w").write(
 "\\begin{table}[htbp]\n\\centering\\tabfont\n\\caption{Verdicts when the gravity-to-$H$ coupling is derived from Churchill's natural-convection correlation for a sphere instead of the assumed 10\\% floor. "
 "Model A attributes the whole boundary resistance to the film (an upper bound on gravity dependence); model B puts a membrane resistance in series with the film, calibrated so that $H(g_0)=H_{1g}$. "
 "$H/H_{1g}$ is for oxygen.}\\label{tab:physical}\n"
 "\\begin{tabular}{|l|l|c|c|c|c|}\n\\hline\n\\textbf{Case} & \\textbf{Coupling} & $\\boldsymbol{g/g_0}$ & $H/H_{1g}$ & $\\Delta/\\theta$ & \\textbf{Verdict}\\\\ \\hline\n"
 + rows + "\\end{tabular}\n\\end{table}\n")
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6), sharey=True)
gs = np.logspace(-6, 0, 61)
for ax, (name, e) in zip(axs, CASES.items()):
    s = e.species[0]; H1 = e.membrane_permeability_1g["oxygen"] * e.aggregate_radius / s.diffusivity
    for mdl, lab, st in [("engine", "engine rule", "k--"), ("A", "film only (A)", "-"), ("B", "membrane + film (B)", "-")]:
        ax.plot(gs, [H_model(e, s, H1, g * G0, mdl) / H1 for g in gs], st, label=lab)
    ax.set_xscale("log"); ax.set_xlim(1, 1e-6); ax.set_ylim(0, 1.05); ax.set_xlabel("$g/g_0$"); ax.set_title(LAB[name]); ax.grid(alpha=.25)
axs[0].set_ylabel("$H(g)/H_{1g}$ (oxygen)"); axs[0].legend(loc="lower left", fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figures", "fig_physical_coupling.pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, "figures", "fig_physical_coupling.png"), dpi=200, bbox_inches="tight")
json.dump(R, open(os.path.join(OUT, "physical.json"), "w"), indent=1)
for n in R:
    print(n, "Ra=%.3g Sh=%.3f Hfilm=%.3f H1g=%.3f" % (R[n]["Ra_1g"], R[n]["Sh_1g"], R[n]["Hfilm_1g"], R[n]["H1g"]))
    for r in R[n]["rows"]: print("  %-6s g=%.3g Hr=%.3f ratio=%.3f %s" % (r["model"], r["g"], r["H_ratio"], r["ratio"], r["verdict"]))
