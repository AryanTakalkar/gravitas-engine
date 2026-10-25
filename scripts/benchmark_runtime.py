"""Runtime benchmark for the reference case (KPI: reference-case runtime).
Run from repo root:  PYTHONPATH=src python scripts/benchmark_runtime.py
"""
import platform, statistics, sys, time
from gravitas import Experiment, screen
from gravitas.verdict import STANDARD_G

N = 20
exp = Experiment.from_yaml("examples/reference_case.yaml")
screen(exp, 1e-3 * STANDARD_G)                     # warm-up (imports, caches)
times = []
for _ in range(N):
    t0 = time.perf_counter(); v = screen(exp, 1e-3 * STANDARD_G); times.append(time.perf_counter() - t0)
print(f"python {sys.version.split()[0]} on {platform.platform()} ({platform.processor() or 'cpu'})")
print(f"reference case, screen() at 1e-3 g: verdict={v.result}, n={N}")
print(f"median {statistics.median(times)*1e3:.1f} ms | min {min(times)*1e3:.1f} ms | max {max(times)*1e3:.1f} ms")
