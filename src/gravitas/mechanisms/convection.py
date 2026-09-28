"""Buoyancy-driven convection mechanism.

Grashof number: Gr = g * beta * delta_c * L^3 / nu^2
(ratio of buoyant to viscous forces; see PLOS ONE microfluidic PCG
microgravity study for this formulation in a cell-culture-scale context.)

Convection is considered "active" (changes regime) when Gr > 1 at 1g and
Gr < threshold at target g -- i.e. the mechanism crosses from buoyancy-
dominated to diffusion-dominated transport.
"""
from __future__ import annotations

GRASHOF_REGIME_BOUNDARY = 1.0  # Gr ~ O(1): convection becomes negligible below this


def grashof_number(g: float, beta: float, delta_c: float, length: float, nu: float) -> float:
    return g * beta * delta_c * length**3 / nu**2


def convection_active(g: float, beta: float, delta_c: float, length: float, nu: float) -> bool:
    return grashof_number(g, beta, delta_c, length, nu) > GRASHOF_REGIME_BOUNDARY
