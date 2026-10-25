"""Mechanism indicators (Gr, Pe, Bo, dP) at 1g, lunar and 1e-3 g. Run: PYTHONPATH=src python scripts/mech_table.py results"""
import sys, os, json
from gravitas import Experiment
from gravitas.verdict import screen, STANDARD_G as G0
OUT = sys.argv[1] if len(sys.argv) > 1 else "results"
def sci(x):
    if x == 0: return "0"
    if 0.01 <= abs(x) < 1000: return f"{x:.3g}"
    m, e = f"{x:.2e}".split("e"); return f"${float(m):g}\\times10^{{{int(e)}}}$"
NAMES = {"buoyancy_convection": "Gr (convection)", "sedimentation": "Pe (sedimentation)",
         "multiphase_separation": "Bo (multiphase)", "hydrostatic_gradient": "$\\Delta P$ (Pa)"}
rows, js = "", {}
for name, f in [("Reference", "reference_case"), ("Organoid", "organoid_case")]:
    e = Experiment.from_yaml(f"examples/{f}.yaml")
    moon = {m.name: m for m in screen(e, 1.62).mechanism_ranking}
    mu = {m.name: m for m in screen(e, 1e-3 * G0).mechanism_ranking}
    rows += f"\\multicolumn{{6}}{{|l|}}{{\\textit{{{name}}}}}\\\\ \\hline\n"
    js[name] = {}
    for k, lab in NAMES.items():
        a, b = moon[k], mu[k]
        rows += f" & {lab} & {sci(a.value_1g)} & {sci(a.value_target_g)} & {sci(b.value_target_g)} & {'yes' if b.crossed else 'no'}\\\\ \\hline\n"
        js[name][k] = dict(v1g=a.value_1g, moon=a.value_target_g, micro=b.value_target_g, crossed_micro=bool(b.crossed), crossed_moon=bool(a.crossed))
open(os.path.join(OUT, "tables", "tab_mech.tex"), "w").write(
 "\\begin{table}[htbp]\n\\centering\\tabfont\n\\caption{Gravity-sensitive mechanism indicators from the classifier at $g_0$, lunar gravity (1.62~m/s$^2$) and $10^{-3}g_0$. "
 "``Crossed'' refers to the $O(1)$ regime boundary between $g_0$ and $10^{-3}g_0$.}\\label{tab:mech_res}\n"
 "\\begin{tabular}{|l|l|c|c|c|c|}\n\\hline\n\\textbf{Case} & \\textbf{Indicator} & $g_0$ & Moon & $10^{-3}g_0$ & \\textbf{Crossed}\\\\ \\hline\n"
 + rows + "\\end{tabular}\n\\end{table}\n")
json.dump(js, open(os.path.join(OUT, "mech.json"), "w"), indent=1)
print(json.dumps(js, indent=1))
