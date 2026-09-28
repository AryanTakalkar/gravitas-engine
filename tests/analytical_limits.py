"""Exact closed-form solutions in the two limiting regimes of the
Michaelis-Menten spherical reaction-diffusion problem. These are standard,
independently-derivable results (not reproduced from any single paper's
text) used to validate the general nonlinear solver in
solver/reaction_diffusion.py.

Regime 1 -- zero-order kinetics (C >> K): the sink term alpha*C/(K+C) -> alpha
(constant). The steady equation becomes the Poisson equation
    (1/R^2) d/dR(R^2 dC/dR) = alpha
whose solution regular at R=0 is:
    C(R) = C(1) + (alpha/6) * (1 - R^2)
with C(1) fixed by the Robin condition C'(1) + H*C(1) = H.
C'(R) = -(alpha/3) R  =>  C'(1) = -alpha/3.
Robin: -alpha/3 + H*C(1) = H  =>  C(1) = 1 + alpha/(3H)... wait sign convention,
see derivation in zero_order_exact() docstring for the exact algebra used.

Regime 2 -- first-order kinetics (C << K): the sink term alpha*C/(K+C) ->
(alpha/K)*C. The steady equation becomes the linear modified-Helmholtz
equation
    (1/R^2) d/dR(R^2 dC/dR) = phi^2 * C,   phi^2 = alpha/K
whose solution regular at R=0 is:
    C(R) = A * sinh(phi*R) / R
with A fixed by the Robin condition at R=1.
"""
from __future__ import annotations

import numpy as np


def zero_order_exact(R: np.ndarray, alpha: float, H: float) -> np.ndarray:
    """Exact steady-state profile in the zero-order-kinetics limit (C >> K).

    Derivation: let u = R^2 * dC/dR. The equation (1/R^2) d/dR(R^2 dC/dR) = alpha
    becomes du/dR = alpha * R^2, so u = (alpha/3) R^3 + c0. Regularity at R=0
    (flux R^2*C' must vanish there) forces c0 = 0, so:
        dC/dR = (alpha/3) R  =>  C(R) = C0 + (alpha/6) R^2
    with C'(1) = alpha/3.

    Applying the Robin BC C'(1) + H*C(1) = H:
        alpha/3 + H*(C0 + alpha/6) = H
        C0 = 1 - alpha/(3H) - alpha/6

    So: C(R) = 1 - alpha/(3H) - (alpha/6)(1 - R^2)

    (Concentration is lowest at the centre and highest at the surface, as
    expected for a diffusion-limited sink -- the previous draft of this
    derivation had a sign error and predicted the opposite, unphysical trend.)
    """
    C0 = 1.0 - alpha / (3.0 * H) - alpha / 6.0
    return C0 + (alpha / 6.0) * R**2


def first_order_exact(R: np.ndarray, alpha: float, K: float, H: float) -> np.ndarray:
    """Exact steady-state profile in the first-order-kinetics limit (C << K).

    General regular-at-origin solution of (1/R^2)(R^2 C')' = phi^2 C is
    C(R) = A * sinh(phi R) / R, phi = sqrt(alpha/K).

    C'(R) = A * [phi*cosh(phi R)/R - sinh(phi R)/R^2]
    At R=1: C(1) = A*sinh(phi), C'(1) = A*[phi*cosh(phi) - sinh(phi)]

    Robin BC: C'(1) + H*C(1) = H
      A*[phi*cosh(phi) - sinh(phi) + H*sinh(phi)] = H
      A = H / [phi*cosh(phi) + (H-1)*sinh(phi)]
    """
    phi = np.sqrt(alpha / K)
    denom = phi * np.cosh(phi) + (H - 1.0) * np.sinh(phi)
    A = H / denom
    # sinh(phi*R)/R is singular-looking at R=0 but has a finite limit (-> A*phi);
    # numpy handles small R fine since sinh(x)/x -> 1 smoothly for the R values
    # used in the radial grid (R > 0 always, cell centres).
    return A * np.sinh(phi * R) / R
