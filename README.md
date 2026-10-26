# GRAVITAS
**Gravity Sensitivity Screening Engine for Space Biology & Tissue Engineering**

[![CI](https://github.com/AryanTakalkar/gravitas-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/AryanTakalkar/gravitas-engine/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**GRAVITAS** is a deterministic computational physics engine and AI surrogate designed to answer a critical aerospace question: *If we send a biological experiment (like a cell organoid or tumor spheroid) into space, will the change in gravity actually affect the biology, or will diffusion dominate regardless?*

By linking external fluid dynamics (Grashof, Péclet, Bond dimensionless numbers) to internal cellular biology (1D Reaction-Diffusion PDEs with Michaelis-Menten kinetics), GRAVITAS acts as a pre-flight vetting gate to prevent the costly launch of gravity-insensitive payloads.

---

##  Key Features

- **Multi-Physics Regimes:** Automatically calculates boundaries for buoyancy-driven convection, sedimentation (Stokes), multiphase separation, and hydrostatic pressure across planetary gravity levels (e.g., Lunar, Martian, Microgravity).
- **Multi-Geometry Solver:** Supports 1D curvilinear diffusion for **spheres** (spheroids/organoids), **cylinders** (bioprinted filaments/fibers), and **planar slabs** (tissue slices/bilayers) with exact closed-form analytical verifications.
- **Bioreactor Boundary Couplings:** Models quiescent natural convection via Churchill laminar correlations (`churchill_film`, `churchill_series`), legacy heuristic floor (`heuristic`), and active perfusion bioreactors via Ranz-Marshall correlation (`perfusion`).
- **Biological Benchmark Validation:** Empirically validated against published microelectrode oxygen profile data in multicellular spheroids (Müller-Klieser et al., 1986).
- **Uncertainty Quantification:** Quantitative flip margin analysis and critical gravity transition tracking ($g_{\text{crit}}$).
- **Physics-Informed Neural Network (PINN):** PyTorch AI surrogate architecture trained via Latin Hypercube Sampling to approximate the nonlinear reaction-diffusion equations (implemented, validation benchmarked on held-out sets: 25.5x CPU speedup).
- **Strict Data Provenance:** Every physical constant used in the engine is tracked in a centralized YAML registry with direct academic citations, ensuring aerospace-grade auditability.
- **CI/CD Pipeline:** Fully version-controlled with a GitHub Actions matrix (Ubuntu, Windows, macOS x Python 3.10, 3.12) ensuring mathematical limits, golden regression values, and determinism tests pass on every commit.

---

##  Architecture

1. **The Classifier:** Computes dimensionless numbers (Grashof, Péclet, Bond, Hydrostatic $\Delta P$) and detects whether fluid transport regime boundaries are crossed when gravity drops.
2. **The Numerical Solver:** Finite-volume curvilinear solver with second-order spatial accuracy and non-linear root finding across spherical, cylindrical, and planar geometries.
3. **Boundary Coupling Engine:** Converts gravity levels or perfusion flow rates into Robin boundary conditions via Churchill or Ranz-Marshall correlations.
4. **The AI Surrogate:** A Parametric PINN targeting rapid generative design space exploration.
5. **The Verdict Engine:** Compares central nutrient depletion against biological measurement uncertainty to issue a definitive **PASS**, **MARGINAL**, or **FAIL**.

---

##  Installation

GRAVITAS requires Python 3.9 or higher. 

Clone the repository and install the engine along with its developer and AI dependencies:

```bash
git clone https://github.com/AryanTakalkar/gravitas-engine.git
cd gravitas-engine
pip install -e .[dev,ai]
```

---

##  Usage

### 1. Launch the Interactive Dashboard
The easiest way to analyze an experiment is via the web UI.
```bash
streamlit run dashboard.py
```
*Opens at `http://localhost:8501`.*

### 2. Run Biological Benchmark Validation
Validate the internal reaction-diffusion solver against published microelectrode measurements:
```bash
python scripts/validate_biological_benchmark.py results
```

### 3. Run Uncertainty Quantification & Flip Margins
Quantify how uncertain biological inputs and boundary coupling choices affect the verdict:
```bash
python scripts/uncertainty_quantification.py results
```

### 4. Run PINN Surrogate Validation
Evaluate PINN generalization error, boundary residuals, and hardware-matched speedup on held-out test sets:
```bash
python scripts/validate_pinn.py results
```

### 5. Use as a Python API
```python
from gravitas import Experiment
from gravitas.verdict import screen

# Load biological configuration
exp = Experiment.from_yaml("examples/reference_case.yaml")

# Screen for Lunar Gravity with correlation-based coupling
verdict = screen(exp, target_g=1.62, coupling="churchill_film")

print(f"Result: {verdict.result}")
print(f"Limiting Factor: {verdict.limiting_reason}")
```

---

##  Directory Structure

```text
gravitas-engine/
├── dashboard.py                # Streamlit Web UI
├── train_pinn.py               # AI Training script for the PINN
├── examples/                   # YAML experiment files (reference, organoid, benchmark)
├── scripts/                    # Validation, benchmark, UQ, and sweep scripts
├── src/gravitas/               # Core Engine Source Code
│   ├── ai/                     # PyTorch PINN Architecture & Physics Loss
│   ├── mechanisms/             # Dimensionless fluid dynamics logic
│   ├── registry/               # Data provenance and cited physical constants
│   ├── solver/                 # 1D Curvilinear Reaction-Diffusion solver & coupling
│   └── validation/             # Biological benchmark comparison module
└── tests/                      # 58 automated unit, limit, convergence, and validation tests
```

---

##  Running Tests & Coverage
Run the automated test suite (58 unit, limit, convergence, golden-value, geometry, and validation tests) with coverage:
```bash
pytest tests/ -v --cov=gravitas --cov-report=term-missing
```

##  Benchmark Runtime
Measure reference-case screening runtime:
```bash
python scripts/benchmark_runtime.py
```

##  License
This project is licensed under the MIT License.
