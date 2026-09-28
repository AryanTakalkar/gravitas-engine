"""Sedimentation mechanism.

Stokes settling velocity (creeping flow, valid for the Re < 1 regime
reported for cell aggregates by Rivera-Solorio et al. 2006):
    v_settle = 2/9 * (rho_p - rho_f) * g * r^2 / mu     [mu = rho_f * nu]

Peclet number (settling): Pe = v_settle * L / D
Sedimentation is considered "active" when Pe > 1 (advective settling
transport competes with diffusion); Rivera-Solorio et al. report Pe in the
O(1)-O(100) range for aggregates in a rotating microgravity bioreactor at
1g-equivalent conditions, so Pe ~ O(1) is the physically grounded threshold.
"""
from __future__ import annotations

PECLET_REGIME_BOUNDARY = 1.0


def stokes_settling_velocity(g: float, rho_particle: float, rho_fluid: float,
                              radius: float, nu: float) -> float:
    mu = rho_fluid * nu
    return (2.0 / 9.0) * (rho_particle - rho_fluid) * g * radius**2 / mu


def peclet_number(v_settle: float, length: float, diffusivity: float) -> float:
    return v_settle * length / diffusivity


def sedimentation_active(g: float, rho_particle: float, rho_fluid: float,
                          radius: float, nu: float, length: float,
                          diffusivity: float) -> bool:
    v = stokes_settling_velocity(g, rho_particle, rho_fluid, radius, nu)
    return peclet_number(v, length, diffusivity) > PECLET_REGIME_BOUNDARY
