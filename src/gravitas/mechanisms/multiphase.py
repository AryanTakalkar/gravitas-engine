"""Multiphase separation and surface-tension-dominance mechanisms.

Bond (Eotvos) number: Bo = delta_rho * g * L^2 / sigma
(ratio of gravitational to surface-tension forces). Gas-liquid phase
separation is gravity-driven when Bo > 1; below that, surface tension
dominates and phases stay mixed/emulsified regardless of orientation.
These two mechanisms are complementary regimes of the same dimensionless
group, per standard multiphase-flow theory.
"""
from __future__ import annotations

BOND_REGIME_BOUNDARY = 1.0


def bond_number(delta_rho: float, g: float, length: float, sigma: float) -> float:
    return delta_rho * g * length**2 / sigma


def phase_separation_active(delta_rho: float, g: float, length: float, sigma: float) -> bool:
    return bond_number(delta_rho, g, length, sigma) > BOND_REGIME_BOUNDARY


def surface_tension_dominant(delta_rho: float, g: float, length: float, sigma: float) -> bool:
    return not phase_separation_active(delta_rho, g, length, sigma)
