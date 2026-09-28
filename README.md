# GRAVITAS — Gravity Sensitivity Screening Engine

Deterministic Python library that screens a proposed biological experiment
for gravity sensitivity: identifies which transport mechanisms change regime
at a target effective-gravity level, predicts the resulting change in
measurable observables, and issues a provenance-complete PASS/MARGINAL/FAIL
verdict.

## Status (working core, built and validated in this session)

- ✅ **Numerical core** (`solver/`): 1D radial transient/steady-state
  reaction-diffusion solver, Michaelis-Menten kinetics, spherical geometry.
- ✅ **Validated against two independent exact closed-form solutions**
  (`tests/test_analytical_limits.py`): zero-order kinetics limit and
  first-order kinetics limit of the same governing PDE. Both pass to
  < 0.5% relative error.
- ✅ **Determinism check**: identical output across 10 repeated runs
  (`tests/test_determinism_and_convergence.py`).
- ✅ **Mesh convergence check**: refining the grid converges monotonically.
- ✅ **Mechanism formalization** (`mechanisms/`): Grashof (convection),
  Péclet/Stokes (sedimentation), Bond/Eötvös (multiphase & surface tension),
  hydrostatic pressure — each with a cited regime boundary.
- ✅ **Regime classifier** (`classifier.py`): computes every mechanism at
  1g and target g, ranks by relative change.
- ✅ **Verdict logic + public API** (`verdict.py`): `screen(experiment,
  target_g)` returns a full `Verdict` with mechanism ranking, observable
  deltas, PASS/MARGINAL/FAIL, and a stated reason.
- ✅ **Parameter registry** (`registry/parameters.yaml`): every physical
  constant sourced and cited.
- ✅ **Reference case** (`examples/reference_case.yaml`): the brief's
  300 μm / 72 h / 1e-3 g case, runs end to end.

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

## Not yet built (fast follow-ups, in priority order)

1. `provenance.py` — wrap registry lookups so every number in a `Verdict`
   carries its citation automatically (currently the registry is cited but
   not yet wired through the output object).
2. Uncertainty model refinement — currently a single `uncertainty_fraction`
   knob; swap in a real per-species measurement-uncertainty spec.
3. Dashboard (Streamlit/HTML) — visualize the mechanism ranking + profile
   comparison for the reference case.
4. CI (`.github/workflows/ci.yml`) — run `pytest` on every push.
5. `pyproject.toml` — make it `pip install -e .`-able.
6. A second worked example (different geometry/species) to demonstrate the
   YAML schema generalizes beyond the mandatory reference case.

## Known simplifications to flag in your report

- The g→boundary-condition coupling (`verdict._membrane_permeability_at_g`)
  is a linear interpolation with a diffusion floor — physically motivated
  (Sherwood number tracks convection strength) but not itself derived from
  a full Navier-Stokes solve. State this explicitly as a modeling
  assumption, not a limitation you're hiding.
- `metabolite` parameters in the registry are flagged as placeholders
  pending a species-specific literature value — replace before screening
  a real (non-reference) experiment.
- Particle density (for sedimentation) is approximated as 1.05× medium
  density — reasonable for cell aggregates but should be an explicit
  experiment input, not a hardcoded ratio, in the next iteration.
