"""Michaelis-Menten reaction kinetics, in dimensionless form.

Dimensionless mapping (following the Lin 1976 / McElwain 1978 convention):
    R = r / a                      (radius, a = aggregate radius)
    C = c / c_bulk                 (concentration, normalised by bulk value)
    tau = t * D / a^2               (dimensionless time)
    alpha = Vmax * a^2 / (D * c_bulk)   (Thiele modulus -- reaction vs diffusion)
    K = km / c_bulk                     (dimensionless Michaelis constant)

Governing PDE (dimensionless, spherical symmetry):
    dC/dtau = (1/R^2) d/dR( R^2 dC/dR ) - alpha * C / (K + C)
"""
from __future__ import annotations


def dimensionless_groups(species, radius: float) -> tuple[float, float]:
    """Compute (alpha, K) for a species given the aggregate radius (m)."""
    c_bulk = species.bulk_concentration
    if c_bulk <= 0:
        raise ValueError(f"species '{species.name}' has non-positive bulk concentration")
    alpha = species.max_uptake_rate * radius**2 / (species.diffusivity * c_bulk)
    K = species.michaelis_constant / c_bulk
    return alpha, K


def reaction_term(C, alpha: float, K: float):
    """Michaelis-Menten sink/source term: alpha * C / (K + C).

    Works elementwise on numpy arrays. Guards against C < 0 (unphysical, can occur
    transiently in a stiff explicit scheme) by clamping to zero for the kinetics term only.
    """
    C_safe = C.clip(min=0) if hasattr(C, "clip") else max(C, 0.0)
    return alpha * C_safe / (K + C_safe)
