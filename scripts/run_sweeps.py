#!/usr/bin/env python
"""GRAVITAS Stage-I results generator (task X1: sweep runner).

Run from the repository root:
    PYTHONPATH=src python scripts/run_sweeps.py results
Everything is deterministic (fixed grids, fixed RNG seed for the Monte Carlo).
Outputs: results/figures/*.pdf|png, results/tables/*.tex, results/results.json
"""
import os, sys, json, time, dataclasses
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.stats import spearmanr

from gravitas import Experiment
from gravitas.verdict import screen, STANDARD_G as G0
from gravitas.solver.kinetics import dimensionless_groups
from gravitas.solver.reaction_diffusion import solve_steady_state, solve_transient

OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
FIG, TAB = os.path.join(OUT, "figures"), os.path.join(OUT, "tables")
for d in (OUT, FIG, TAB):
    os.makedirs(d, exist_ok=True)

MOON, MARS = 1.62, 3.71            # m/s^2 (surface gravity, rounded)
C_PASS, C_MARG, C_FAIL = "#2e8b57", "#e69f00", "#c0392b"
CASES = {
    "reference": Experiment.from_yaml("examples/reference_case.yaml"),
    "organoid": Experiment.from_yaml("examples/organoid_case.yaml"),
}
LABEL = {"reference": "Reference (300 $\\mu$m aggregate)",
         "organoid": "Hepatic organoid (1 mm)"}
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "legend.fontsize": 7, "savefig.bbox": "tight",
                     "axes.grid": True, "grid.alpha": 0.25})
R = {}   # results dictionary dumped to json


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200)
    plt.close(fig)


def sci(x, digits=3):
    if x == 0:
        return "0"
    if 0.01 <= abs(x) < 1000:
        return f"{x:.{digits}g}"
    m, e = f"{x:.{digits-1}e}".split("e")
    return f"${float(m):g}\\times10^{{{int(e)}}}$"


def write(name, s):
    with open(os.path.join(TAB, name), "w") as fh:
        fh.write(s)


# ---------------------------------------------------------------- core helpers
_cache = {}
FAILS = []
def steady(a, K, H, n=100):
    key = (a, K, H, n)
    if key not in _cache:
        try:
            _cache[key] = solve_steady_state(alpha=a, K=K, H=H, n_cells=n)
        except RuntimeError:
            _cache[key] = None
            FAILS.append(dict(alpha=a, K=K, H=H))
    return _cache[key]


def H1_of(exp, s):
    return exp.membrane_permeability_1g[s.name] * exp.aggregate_radius / s.diffusivity


def H_of_g(H1, gr, f=0.1, n=1.0):
    """Gravity coupling family. (f=0.1, n=1) is exactly the engine's rule."""
    gr = min(max(gr, 0.0), 1.0)
    return H1 * (f + (1.0 - f) * gr ** n)


def observables(exp, gr, f=0.1, n=1.0):
    out = {}
    for s in exp.species:
        a, K = dimensionless_groups(s, exp.aggregate_radius)
        H1 = H1_of(exp, s)
        r1, rt = steady(a, K, H1), steady(a, K, H_of_g(H1, gr, f, n))
        if r1 is None or rt is None:
            continue          # non-converged solve: species skipped (logged in FAILS)
        C1, Ct = r1[1], rt[1]
        out[s.name] = (C1[0], Ct[0], abs(C1[0] - Ct[0]), s.measurement_uncertainty)
    return out


def verdict_from(obs, band=1.5):
    worst = max(obs, key=lambda k: obs[k][2])
    c1, ct, d, u = obs[worst]
    thr = u * max(c1, 1e-30)
    res = "PASS" if d > band * thr else ("MARGINAL" if d > thr else "FAIL")
    return res, worst, d, thr


# --------------------------------------------- sanity: variant == engine screen()
for name, e in CASES.items():
    for gr in [1, MARS / G0, MOON / G0, 0.1, 1e-3, 1e-6]:
        v = screen(e, gr * G0)
        res, w, d, thr = verdict_from(observables(e, gr))
        assert v.result == res and v.limiting_mechanism == w, (name, gr)
        assert abs(v.uncertainty_threshold - thr) < 1e-12
print("variant model reproduces engine screen(): OK", flush=True)

# ------------------------------------------------------------------- E1: sweep
gs = np.logspace(-6, 0, 61)
sweep = {}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
for ax, (name, e) in zip(axs, CASES.items()):
    obs = [observables(e, g) for g in gs]
    verd = [verdict_from(o) for o in obs]
    spn = [s.name for s in e.species]
    edges = np.sqrt(gs[:-1] * gs[1:])
    step = gs[1] / gs[0]
    edges = np.concatenate([[gs[0] / np.sqrt(step)], edges, [gs[-1] * np.sqrt(step)]])
    for i, v in enumerate(verd):
        ax.axvspan(edges[i], edges[i + 1], lw=0, alpha=0.13,
                   color={"PASS": C_PASS, "MARGINAL": C_MARG, "FAIL": C_FAIL}[v[0]])
    for sp, col in zip(spn, ["tab:blue", "tab:purple", "tab:gray"]):
        d = np.array([o[sp][2] for o in obs]); d[d <= 0] = np.nan
        ax.plot(gs, d, color=col, label=sp)
    c1o = obs[0]["oxygen"][0]; uo = e.species[0].measurement_uncertainty
    thr = uo * c1o
    ax.axhline(thr, color="k", ls="--", lw=0.8); ax.axhline(1.5 * thr, color="k", ls=":", lw=0.9)
    ax.text(1.4e-6, thr * 1.12, "$\\theta$", fontsize=7); ax.text(1.4e-6, 1.5 * thr * 1.12, "$1.5\\theta$", fontsize=7)
    for gg, lab in [(MOON / G0, "Moon"), (MARS / G0, "Mars")]:
        ax.axvline(gg, color="gray", lw=0.7, ls="-.")
        ax.text(gg * 1.1, 0.55, lab, rotation=90, fontsize=6.5, va="top", ha="right")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(1e-4, 1.0)
    ax.set_xlim(1.0, 1e-6)
    ax.set_xlabel("$g/g_0$"); ax.set_title(LABEL[name])
    sweep[name] = dict(g=gs.tolist(), delta={sp: [o[sp][2] for o in obs] for sp in spn},
                       verdict=[v[0] for v in verd], limiting=[v[1] for v in verd])
axs[0].set_ylabel("centre-concentration change $\\Delta_k$")
axs[0].legend(loc="lower left")
save(fig, "fig_gravity_sweep")

rows = ""
G_ROWS = [(1.0, "1 (Earth)"), (MARS / G0, f"{MARS/G0:.3f} (Mars)"), (MOON / G0, f"{MOON/G0:.3f} (Moon)"),
          (0.1, "0.1"), (0.01, "0.01"), (1e-3, "$10^{-3}$"), (1e-6, "$10^{-6}$")]
R["sweep_table"] = {}
for name, e in CASES.items():
    rows += f"\\multicolumn{{7}}{{|l|}}{{\\textit{{{LABEL[name]}}}}}\\\\ \\hline\n"
    R["sweep_table"][name] = []
    for gr, lab in G_ROWS:
        v = screen(e, gr * G0)
        o = max(v.observables, key=lambda x: x.delta_center)
        ratio = o.delta_center / v.uncertainty_threshold
        rows += (f" & {lab} & {v.result} & {v.limiting_mechanism} & {sci(o.delta_center)} & "
                 f"{sci(v.uncertainty_threshold)} & {ratio:.2f}\\\\ \\hline\n")
        R["sweep_table"][name].append(dict(gr=gr, result=v.result, species=v.limiting_mechanism,
                                           delta=o.delta_center, theta=v.uncertainty_threshold, ratio=ratio))
write("tab_sweep.tex",
      "\\begin{table}[htbp]\n\\centering\\tabfont\n"
      "\\caption{Engine verdicts from the gravity sweep (steady state, engine coupling rule). "
      "$\\Delta$ is the largest centre-concentration change, $\\theta=u_{k^\\dagger}C_{k^\\dagger}(0;g_0)$.}"
      "\\label{tab:sweep}\n\\begin{tabular}{|l|l|l|l|c|c|c|}\n\\hline\n"
      "\\textbf{Case} & $\\boldsymbol{g/g_0}$ & \\textbf{Verdict} & \\textbf{Limiting} & $\\boldsymbol\\Delta$ & $\\boldsymbol\\theta$ & $\\boldsymbol{\\Delta/\\theta}$\\\\ \\hline\n"
      + rows + "\\end{tabular}\n\\end{table}\n")
print("E1 done", flush=True)

# --------------------------------------------------------------- E2: profiles
fig, axs = plt.subplots(2, 3, figsize=(6.3, 4.0), sharex=True)
for i, (name, e) in enumerate(CASES.items()):
    for j in range(3):
        ax = axs[i, j]
        if j >= len(e.species):
            ax.axis("off"); continue
        s = e.species[j]; a, K = dimensionless_groups(s, e.aggregate_radius); H1 = H1_of(e, s)
        for gr, st, col, lab in [(1.0, "--", "k", "$g_0$"), (MOON / G0, "-", "tab:orange", "Moon"),
                                 (1e-3, "-", C_FAIL, "$10^{-3}g_0$")]:
            Rg, C = steady(a, K, H_of_g(H1, gr))
            ax.plot(Rg, C, st, color=col, label=lab)
        ax.set_title(f"{s.name}", fontsize=8)
        if j == 0:
            ax.set_ylabel(f"{LABEL[name].split(' (')[0]}\n$C=c/c_\\infty$")
        if i == 1 or j == 2:
            ax.set_xlabel("$R=r/a$")
axs[0, 0].legend(loc="best")
fig.tight_layout()
save(fig, "fig_profiles")
print("E2 done", flush=True)

# ------------------------------------------------- E3: convergence/verification
from gravitas.solver.discretisation import RadialGrid
ns = [20, 40, 80, 160, 320]
def zero_exact(Rg, al, H):
    return 1 - al / (3 * H) - al / 6 + (al / 6) * Rg ** 2
def first_exact(Rg, al, K, H):
    phi = np.sqrt(al / K); A = H / (phi * np.cosh(phi) + (H - 1) * np.sinh(phi))
    return A * np.sinh(phi * Rg) / Rg
err0, err1 = [], []
for n in ns:
    Rg, C = solve_steady_state(alpha=0.1, K=1e-8, H=2.0, n_cells=n)
    err0.append(float(np.max(np.abs(C - zero_exact(Rg, 0.1, 2.0)) / zero_exact(Rg, 0.1, 2.0))))
    Rg, C = solve_steady_state(alpha=1e8, K=1e8, H=2.0, n_cells=n)
    err1.append(float(np.max(np.abs(C - first_exact(Rg, 1e8, 1e8, 2.0)) / first_exact(Rg, 1e8, 1e8, 2.0))))
def order(err):
    return [float(np.log2(err[i] / err[i + 1])) for i in range(len(err) - 1)]
R["analytical"] = dict(n=ns, zero_order_err=err0, first_order_err=err1,
                       p_zero=order(err0), p_first=order(err1))
qoi = {}
for name, e in CASES.items():
    s = e.species[0]; a, K = dimensionless_groups(s, e.aggregate_radius); H1 = H1_of(e, s)
    c1, ct = [], []
    for n in ns:
        c1.append(float(solve_steady_state(alpha=a, K=K, H=H1, n_cells=n)[1][0]))
        ct.append(float(solve_steady_state(alpha=a, K=K, H=0.1 * H1 + 0.9 * H1 * 1e-3, n_cells=n)[1][0]))
    d = [abs(x - y) for x, y in zip(c1, ct)]
    diffs = [abs(d[i] - d[i + 1]) for i in range(len(d) - 1)]
    qoi[name] = dict(n=ns, C0_1g=c1, C0_target=ct, delta=d, succ_diff=diffs,
                     p=[float(np.log2(diffs[i] / diffs[i + 1])) for i in range(len(diffs) - 1)])
R["qoi_convergence"] = qoi
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6))
axs[0].loglog(ns, err0, "o-", label="zero-order limit"); axs[0].loglog(ns, err1, "s-", label="first-order limit")
nn = np.array(ns, float); axs[0].loglog(nn, err0[0] * (nn[0] / nn) ** 2, "k:", lw=0.8, label="slope $-2$")
axs[0].loglog(nn, err0[0] * (nn[0] / nn), "k--", lw=0.8, label="slope $-1$")
axs[0].set_xlabel("radial cells $N$"); axs[0].set_ylabel("max relative error vs. exact"); axs[0].legend(); axs[0].set_title("(a) analytical limits")
for name in CASES:
    axs[1].loglog(ns[:-1], qoi[name]["succ_diff"], "o-", label=LABEL[name].split(" (")[0])
axs[1].set_xlabel("radial cells $N$"); axs[1].set_ylabel("$|\\Delta_N-\\Delta_{2N}|$"); axs[1].legend(); axs[1].set_title("(b) verdict observable")
fig.tight_layout(); save(fig, "fig_convergence")

write("tab_conv.tex",
      "\\begin{table}[htbp]\n\\centering\\tabfont\n"
      "\\caption{Mesh refinement: maximum relative error of the steady solver against the two exact limits, "
      "and convergence of the oxygen verdict observable $\\Delta$ at $g^{*}=10^{-3}g_0$ (engine coupling).}\\label{tab:conv}\n"
      "\\begin{tabular}{|c|c|c|c|c|}\n\\hline\n"
      "$N$ & zero-order error & first-order error & $\\Delta_{\\mathrm{ref}}$ & $\\Delta_{\\mathrm{org}}$\\\\ \\hline\n"
      + "".join(f"{n} & {sci(err0[i])} & {sci(err1[i])} & {qoi['reference']['delta'][i]:.6f} & {qoi['organoid']['delta'][i]:.6f}\\\\ \\hline\n" for i, n in enumerate(ns))
      + f"observed order (last pair) & {R['analytical']['p_zero'][-1]:.2f} & {R['analytical']['p_first'][-1]:.2f} & "
        f"{qoi['reference']['p'][-1]:.2f} & {qoi['organoid']['p'][-1]:.2f}\\\\ \\hline\n"
      "\\end{tabular}\n\\end{table}\n")
print("E3 done", flush=True)

# --------------------------------------------------------- E4: transient check
relax = {}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.6))
rows = ""
for ax, (name, e) in zip(axs, CASES.items()):
    relax[name] = {}
    rows += f"\\multicolumn{{6}}{{|l|}}{{\\textit{{{LABEL[name]}}}}}\\\\ \\hline\n"
    for sp in e.species:
        a, K = dimensionless_groups(sp, e.aggregate_radius); H1 = H1_of(e, sp)
        t_scale = e.aggregate_radius ** 2 / sp.diffusivity
        entry = {"a2_over_D_s": t_scale}
        cells = []
        for tag, H in [("1g", H1), ("target", H_of_g(H1, 1e-3))]:
            tau, Cs, _ = solve_transient(alpha=a, K=K, H=H, tau_end=100.0, n_cells=100, n_output=2001)
            c0 = Cs[:, 0]; css = steady(a, K, H)[1][0]
            dev = np.abs(c0 - css); tot = abs(1.0 - css)
            idx = np.where(dev <= 0.01 * max(tot, 1e-12))[0]
            t99 = float(tau[idx[0]] * t_scale / 60.0) if len(idx) else float("nan")
            entry[tag] = dict(t99_min=t99, final_abs_diff=float(abs(c0[-1] - css)))
            cells.append(t99)
            if sp.name == "oxygen":
                ax.plot(tau * t_scale / 60.0, c0, "--" if tag == "1g" else "-", color="k" if tag == "1g" else C_FAIL,
                        label="$g_0$" if tag == "1g" else "$10^{-3}g_0$")
        relax[name][sp.name] = entry
        rows += (f" & {sp.name} & {t_scale:.1f} & {cells[0]:.2f} & {cells[1]:.2f} & "
                 f"{72*60/cells[1]:.0f}\\\\ \\hline\n")
    ax.set_xlabel("time (min)"); ax.set_title(LABEL[name] + ": oxygen"); ax.set_xlim(0, 25)
axs[0].set_ylabel("centre $C(0,t)$"); axs[0].legend()
fig.tight_layout(); save(fig, "fig_transient")
R["relaxation"] = relax
write("tab_relax.tex",
      "\\begin{table}[htbp]\n\\centering\\tabfont\n"
      "\\caption{Transient solver started from $C=1$: time for the centre concentration to come within 1\\% of its total change to the steady value ($t_{99}$). "
      "The last column is the 72~h duration divided by $t_{99}$ at $10^{-3}g_0$.}\\label{tab:relax}\n"
      "\\begin{tabular}{|l|l|c|c|c|c|}\n\\hline\n"
      "\\textbf{Case} & \\textbf{Species} & $a^2/D$ (s) & $t_{99}$ at $g_0$ (min) & $t_{99}$ at $10^{-3}g_0$ (min) & $72\\,\\mathrm{h}/t_{99}$\\\\ \\hline\n"
      + rows + "\\end{tabular}\n\\end{table}\n")
print("E4 done", flush=True)

# --------------------------------------------- E5: robustness to coupling rule
fs = [0.05, 0.1, 0.2, 0.3, 0.5]; ns_exp = [0.25, 0.5, 1.0]
gfine = np.logspace(-6, 0, 61)
coup = {}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
cmap = plt.get_cmap("viridis")
for ax, (name, e) in zip(axs, CASES.items()):
    coup[name] = {}
    for fi, f in enumerate(fs):
        for n in ns_exp:
            ratios, verds = [], []
            for g in gfine:
                v, w, d, thr = verdict_from(observables(e, g, f, n))
                ratios.append(d / thr); verds.append(v)
            pass_g = [g for g, v in zip(gfine, verds) if v == "PASS"]
            marg_g = [g for g, v in zip(gfine, verds) if v in ("PASS", "MARGINAL")]
            gp = max(pass_g) if pass_g else None
            gm = max(marg_g) if marg_g else None
            monotone = (gp is None) or all(v == "PASS" for g, v in zip(gfine, verds) if g <= gp)
            at = {k: verdict_from(observables(e, gr, f, n))[0] for k, gr in
                  [("moon", MOON / G0), ("mars", MARS / G0), ("1e-3", 1e-3)]}
            coup[name][f"f={f},n={n}"] = dict(f=f, n=n, g_pass=gp, g_marg=gm, monotone=monotone, at=at)
            if n in (1.0, 0.25):
                ax.plot(gfine, ratios, color=cmap(fi / (len(fs) - 1)), ls="-" if n == 1.0 else "--", lw=1.0,
                        label=f"$f_{{\\min}}$={f}" if n == 1.0 else None)
    ax.axhline(1.0, color="k", ls="--", lw=0.8); ax.axhline(1.5, color="k", ls=":", lw=0.9)
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1.0, 1e-6); ax.set_ylim(1e-3, 60)
    ax.set_xlabel("$g/g_0$"); ax.set_title(LABEL[name])
axs[0].set_ylabel("$\\Delta/\\theta$ (limiting species)")
axs[0].legend(loc="lower right", ncol=1)
axs[1].plot([], [], "k-", label="$n=1$ (engine)"); axs[1].plot([], [], "k--", label="$n=0.25$"); axs[1].legend(loc="lower right")
fig.tight_layout(); save(fig, "fig_coupling")
R["coupling"] = coup

def cell(c):
    return "none" if c["g_pass"] is None else sci(c["g_pass"], 2)
rows = ""
for f in fs:
    rows += f"{f}"
    for name in CASES:
        for n in ns_exp:
            rows += " & " + cell(coup[name][f"f={f},n={n}"])
    rows += "\\\\ \\hline\n"
write("tab_coupling.tex",
      "\\begin{table}[htbp]\n\\centering\\tabfont\n"
      "\\caption{Sensitivity of the PASS onset to the gravity-to-$H$ coupling "
      "$H(g)=H_{1g}[f_{\\min}+(1-f_{\\min})(g/g_0)^{n}]$. Entries are the largest $g/g_0$ on a log grid "
      "(0.1-decade steps, $10^{-6}$ to 1) at which the verdict is PASS; ``none'' means no PASS anywhere on the grid. "
      "$n=1$, $f_{\\min}=0.1$ is the engine rule. These are scenarios, not calibrated correlations.}\\label{tab:coupling}\n"
      "\\begin{tabular}{|c|ccc|ccc|}\n\\hline\n"
      " & \\multicolumn{3}{c|}{\\textbf{Reference}} & \\multicolumn{3}{c|}{\\textbf{Organoid}}\\\\ \\hline\n"
      "$f_{\\min}$ & $n{=}0.25$ & $n{=}0.5$ & $n{=}1$ & $n{=}0.25$ & $n{=}0.5$ & $n{=}1$\\\\ \\hline\n"
      + rows + "\\end{tabular}\n\\end{table}\n")
print("E5 done", flush=True)

# ------------------------------------------- E6: sensitivity (OAT + Monte Carlo)
PAR = ["Vmax", "D", "Km", "c", "h", "a", "f"]
PLAB = {"Vmax": "$V_{\\max}$", "D": "$D$", "Km": "$K_m$", "c": "$c_\\infty$", "h": "$h_{1g}$", "a": "radius $a$", "f": "$f_{\\min}$"}
def oxy_ratio(e, gr, sc):
    s = [x for x in e.species if x.name == "oxygen"][0]
    s2 = dataclasses.replace(s, max_uptake_rate=s.max_uptake_rate * sc["Vmax"], diffusivity=s.diffusivity * sc["D"],
                             michaelis_constant=s.michaelis_constant * sc["Km"], bulk_concentration=s.bulk_concentration * sc["c"])
    perm = dict(e.membrane_permeability_1g); perm["oxygen"] = perm["oxygen"] * sc["h"]
    e2 = dataclasses.replace(e, species=(s2,), aggregate_radius=e.aggregate_radius * sc["a"], membrane_permeability_1g=perm)
    a, K = dimensionless_groups(s2, e2.aggregate_radius); H1 = perm["oxygen"] * e2.aggregate_radius / s2.diffusivity
    C1 = solve_steady_state(alpha=a, K=K, H=H1)[1][0]
    Ct = solve_steady_state(alpha=a, K=K, H=H_of_g(H1, gr, 0.1 * sc["f"]))[1][0]
    return abs(C1 - Ct) / (s.measurement_uncertainty * C1)
ONE = {p: 1.0 for p in PAR}
oat = {}
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.7), sharey=True)
for ax, (name, e) in zip(axs, CASES.items()):
    nom = oxy_ratio(e, 1e-3, ONE); oat[name] = {"nominal": nom}
    lo, hi = [], []
    for p in PAR:
        l = oxy_ratio(e, 1e-3, {**ONE, p: 0.5}); h = oxy_ratio(e, 1e-3, {**ONE, p: 1.5})
        oat[name][p] = dict(x0p5=l, x1p5=h); lo.append(l); hi.append(h)
    order_idx = np.argsort([-abs(h - l) for l, h in zip(lo, hi)])
    for k, i in enumerate(order_idx):
        ax.barh(k, lo[i] - nom, left=nom, color="tab:blue", height=0.6, label="$\\times0.5$" if k == 0 else None)
        ax.barh(k, hi[i] - nom, left=nom, color="tab:orange", height=0.6, label="$\\times1.5$" if k == 0 else None)
    ax.set_yticks(range(len(PAR))); ax.set_yticklabels([PLAB[PAR[i]] for i in order_idx]); ax.invert_yaxis()
    ax.axvline(nom, color="k", lw=0.8); ax.axvline(1.5, color="k", ls=":", lw=0.9)
    ax.set_xlabel("$\\Delta/\\theta$ (oxygen, $10^{-3}g_0$)"); ax.set_title(LABEL[name])
axs[0].legend(loc="lower right")
fig.tight_layout(); save(fig, "fig_tornado")
R["oat"] = oat

N_MC = int(os.environ.get("N_MC", "800")); rng = np.random.default_rng(12345)
LEVELS = [("Moon", MOON / G0), ("Mars", MARS / G0), ("$10^{-3}g_0$", 1e-3)]
mc = {}
t0 = time.time()
for name, e in CASES.items():
    X = rng.uniform(0.5, 1.5, size=(N_MC, len(PAR)))
    Y = np.full((N_MC, len(LEVELS)), np.nan)
    for i in range(N_MC):
        sc = dict(zip(PAR, X[i]))
        try:
            for j, (_, gr) in enumerate(LEVELS):
                Y[i, j] = oxy_ratio(e, gr, sc)
        except Exception:
            pass
    ok = ~np.isnan(Y).any(axis=1)
    mc[name] = dict(n_ok=int(ok.sum()), n_total=N_MC, levels={}, spearman={})
    for j, (lab, gr) in enumerate(LEVELS):
        y = Y[ok, j]
        mc[name]["levels"][lab] = dict(pass_=float(np.mean(y > 1.5)), marg=float(np.mean((y > 1.0) & (y <= 1.5))),
                                       fail=float(np.mean(y <= 1.0)), median=float(np.median(y)),
                                       p5=float(np.percentile(y, 5)), p95=float(np.percentile(y, 95)))
    y = Y[ok, 2]
    mc[name]["spearman"] = {p: float(spearmanr(X[ok, k], y)[0]) for k, p in enumerate(PAR)}
    print(f"MC {name}: {time.time()-t0:.0f}s", flush=True)
R["monte_carlo"] = mc
rows = ""
for name in CASES:
    rows += f"\\multicolumn{{6}}{{|l|}}{{\\textit{{{LABEL[name]} ($N={mc[name]['n_ok']}$)}}}}\\\\ \\hline\n"
    for lab, _ in LEVELS:
        L = mc[name]["levels"][lab]
        rows += (f" & {lab} & {100*L['pass_']:.0f}\\% & {100*L['marg']:.0f}\\% & {100*L['fail']:.0f}\\% & "
                 f"{L['median']:.2f} ({L['p5']:.2f}--{L['p95']:.2f})\\\\ \\hline\n")
write("tab_mc.tex",
      "\\begin{table}[htbp]\n\\centering\\tabfont\n"
      "\\caption{Monte Carlo propagation of assumed $\\pm50\\%$ uniform input ranges (oxygen only, seven inputs varied together, seed 12345). "
      "Verdict classes use $\\Delta/\\theta$ with the engine's $1.5$ band; the last column gives the median and 5--95\\% range of $\\Delta/\\theta$. "
      "The ranges are illustrative, not calibrated.}\\label{tab:mc}\n"
      "\\begin{tabular}{|l|l|c|c|c|c|}\n\\hline\n"
      "\\textbf{Case} & $\\boldsymbol{g^*}$ & PASS & MARGINAL & FAIL & $\\Delta/\\theta$ median (5--95\\%)\\\\ \\hline\n"
      + rows + "\\end{tabular}\n\\end{table}\n")
rows = ""
for p in PAR:
    rows += f"{PLAB[p]} & {mc['reference']['spearman'][p]:+.2f} & {mc['organoid']['spearman'][p]:+.2f}\\\\ \\hline\n"
write("tab_spearman.tex",
      "\\begin{table}[htbp]\n\\centering\\tabfont\n"
      "\\caption{Spearman rank correlation between each scaled input and $\\Delta/\\theta$ at $g^*=10^{-3}g_0$ in the Monte Carlo sample.}\\label{tab:spearman}\n"
      "\\begin{tabular}{|l|c|c|}\n\\hline\n\\textbf{Input} & \\textbf{Reference} & \\textbf{Organoid}\\\\ \\hline\n"
      + rows + "\\end{tabular}\n\\end{table}\n")
print("E6 done", flush=True)

# ------------------------------------------------------ E7: radius-g regime map
radii = np.linspace(50e-6, 750e-6, 29); grs = np.logspace(-4, 0, 25)
cm = ListedColormap([C_FAIL, C_MARG, C_PASS])
fig, axs = plt.subplots(1, 2, figsize=(6.3, 2.8), sharey=True)
rmap = {}
for ax, (name, e) in zip(axs, CASES.items()):
    Z = np.full((len(grs), len(radii)), np.nan)
    for i, gr in enumerate(grs):
        for j, r in enumerate(radii):
            try:
                res = screen(dataclasses.replace(e, aggregate_radius=float(r)), gr * G0).result
                Z[i, j] = {"FAIL": 0, "MARGINAL": 1, "PASS": 2}[res]
            except Exception:
                pass
    ax.pcolormesh(radii * 1e6, grs, Z, cmap=cm, vmin=0, vmax=2, shading="nearest", alpha=0.8)
    ax.plot(e.aggregate_radius * 1e6, 1e-3, "k*", ms=8)
    ax.set_yscale("log"); ax.set_xlabel("aggregate radius $a$ ($\\mu$m)"); ax.set_title(LABEL[name] + " kinetics")
    jj = [j for j in range(len(radii)) if Z[list(grs).index(min(grs, key=lambda x: abs(x - 1e-3))), j] == 2]
    row = int(np.argmin(np.abs(grs - 1e-3)))
    pass_r = [float(radii[j] * 1e6) for j in range(len(radii)) if Z[row, j] == 2]
    rmap[name] = dict(radii_um=(radii * 1e6).tolist(), g=grs.tolist(), Z=Z.tolist(),
                      min_pass_radius_um_at_1e3=min(pass_r) if pass_r else None,
                      nan_cells=int(np.isnan(Z).sum()))
axs[0].set_ylabel("$g/g_0$")
from matplotlib.patches import Patch
axs[1].legend(handles=[Patch(color=C_PASS, label="PASS"), Patch(color=C_MARG, label="MARGINAL"), Patch(color=C_FAIL, label="FAIL")],
              loc="lower right")
fig.tight_layout(); save(fig, "fig_regime_map")
R["regime_map"] = rmap
print("E7 done", flush=True)

R["solver_failures"] = dict(count=len(FAILS), cases=FAILS)
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(R, fh, indent=1)
print("ALL DONE", flush=True)
