"""Hydrostatic pressure gradient mechanism.

delta_P = rho * g * L  -- linear in g by construction, so this mechanism's
regime-change is simply whether delta_P at target g falls below a stated
measurable-pressure-difference threshold (instrument/biological relevance
threshold, supplied per-experiment rather than a universal constant).
"""
from __future__ import annotations


def hydrostatic_pressure_difference(rho: float, g: float, length: float) -> float:
    return rho * g * length


def hydrostatic_active(rho: float, g: float, length: float, threshold_pa: float) -> bool:
    return hydrostatic_pressure_difference(rho, g, length) > threshold_pa
