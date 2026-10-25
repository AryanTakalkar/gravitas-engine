"""Gravity -> boundary mass-transfer coupling models.

The solver's Robin condition uses a dimensionless boundary coefficient H = h*a/D.
Gravity enters the model only through H. Two explicit, auditable options are provided:

* ``heuristic``  -- H relaxes linearly from its 1g value to a floor (default 10 % of H_1g).
  Engineering assumption; not a calibrated correlation.
* ``churchill_film`` / ``churchill_series`` -- H derived from Churchill's laminar natural-convection
  correlation for a sphere (heat-mass analogy, Pr -> Sc):

      Sh = 2 + 0.589 Ra^(1/4) / [1 + (0.469/Sc)^(9/16)]^(4/9)      (Ra <= 1e11, Sc >= 0.7)
      Ra = g * beta * delta_c * d^3 / (nu * D),   d = 2a

  ``churchill_film``:   H(g) = H_1g * Sh(g)/Sh(g0)   (whole boundary resistance is the film; upper bound
                        on gravity dependence).
  ``churchill_series``: membrane resistance in series with the film, calibrated so H(g0) = H_1g;
                        H(g) = 1 / (1/H_mem + 2/Sh(g)).

Limits of validity: the correlation is a laminar boundary-layer result for a quiescent medium; below Ra ~ 1 it
acts as a smooth interpolation towards Sh = 2. Taking delta_c equal to the bulk concentration overstates the
density difference, so Ra is an upper bound.
"""
from __future__ import annotations

CHURCHILL_COEFF = 0.589
CHURCHILL_PR_CONST = 0.469
COUPLING_MODELS = ("heuristic", "churchill_film", "churchill_series")


def heuristic_floor(H_1g: float, g: float, standard_g: float, floor_fraction: float = 0.1) -> float:
    """Linear relaxation of H from H_1g (at standard g) to ``floor_fraction * H_1g`` (at g = 0)."""
    g_ratio = max(min(g / standard_g, 1.0), 0.0)
    floor = H_1g * floor_fraction
    return floor + (H_1g - floor) * g_ratio


def solutal_rayleigh(g: float, beta: float, delta_c: float, diameter: float, nu: float, D: float) -> float:
    """Ra = g * beta * delta_c * d^3 / (nu * D) (solutal Rayleigh number)."""
    return max(g, 0.0) * beta * delta_c * diameter ** 3 / (nu * D)


def churchill_sherwood(ra: float, sc: float) -> float:
    """Sherwood number (based on diameter) for a sphere in a quiescent liquid; -> 2 as Ra -> 0."""
    return 2.0 + CHURCHILL_COEFF * ra ** 0.25 / (1.0 + (CHURCHILL_PR_CONST / sc) ** (9.0 / 16.0)) ** (4.0 / 9.0)


def churchill_coupling(H_1g: float, g: float, standard_g: float, *, beta: float, delta_c: float,
                       radius: float, nu: float, D: float, mode: str = "film") -> float:
    """Return H(g) from Churchill's correlation. ``mode`` is ``"film"`` or ``"series"``."""
    if mode not in ("film", "series"):
        raise ValueError(f"unknown Churchill mode {mode!r}")
    sc = nu / D
    d = 2.0 * radius
    sh0 = churchill_sherwood(solutal_rayleigh(standard_g, beta, delta_c, d, nu, D), sc)
    shg = churchill_sherwood(solutal_rayleigh(g, beta, delta_c, d, nu, D), sc)
    if mode == "film":
        return H_1g * shg / sh0
    h_film0 = sh0 / 2.0
    if h_film0 <= H_1g:
        raise ValueError("series mode needs H_1g below the natural-convection film limit Sh(g0)/2; "
                         f"got H_1g={H_1g:.4g} >= {h_film0:.4g}")
    h_mem = 1.0 / (1.0 / H_1g - 1.0 / h_film0)
    return 1.0 / (1.0 / h_mem + 2.0 / shg)
