# GRAVITAS — Gravity Sensitivity Screening Engine

Deterministic Python library that screens a proposed biological experiment
for gravity sensitivity: identifies which transport mechanisms change regime
at a target effective-gravity level, predicts the resulting change in
measurable observables, and issues a provenance-complete PASS/MARGINAL/FAIL
verdict.


## Quick start

```bash
pip install numpy scipy pyyaml pytest
python3 -m pytest tests/ -v          # run the validation suite
```

```python
from gravitas import Experiment, screen

exp = Experiment.from_yaml("examples/reference_case.yaml")
verdict = screen(exp, target_g=9.80665e-3)  # 1/1000 standard g
print(verdict.result, verdict.limiting_mechanism, verdict.limiting_reason)
```

