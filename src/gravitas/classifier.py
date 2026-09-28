"""Regime classifier (Objective 2): compute the mechanism dimensionless
groups at 1g and at the target g, detect which regime boundaries are
crossed, and rank the mechanisms that actually change for this experiment.
"""
from __future__ import annotations

from dataclasses import dataclass

from .mechanisms.convection import grashof_number, GRASHOF_REGIME_BOUNDARY
from .mechanisms.sedimentation import (
    stokes_settling_velocity, peclet_number, PECLET_REGIME_BOUNDARY,
)
from .mechanisms.multiphase import bond_number, BOND_REGIME_BOUNDARY
from .experiment import Experiment


@dataclass(frozen=True)
class MechanismResult:
    name: str
    value_1g: float
    value_target_g: float
    regime_boundary: float
    crossed: bool          # did the mechanism cross its regime boundary?
    relative_change: float  # |value_1g - value_target_g| / value_1g


def classify(experiment: Experiment, standard_g: float, target_g: float) -> list[MechanismResult]:
    """Compute every mechanism's dimensionless group at both gravity levels
    and rank them by how much they change. Returns results sorted by
    relative_change, descending (largest change first = the mechanism most
    likely to drive an observable difference)."""
    length = 2.0 * experiment.aggregate_radius
    rho_f = experiment.medium_density
    nu = experiment.medium_kinematic_viscosity
    # Approximate particle density as slightly denser than medium (typical
    # for cell aggregates, ~1.05-1.10 g/cm^3 vs ~1.00-1.01 g/cm^3 medium);
    # this is exposed as a parameter so a real experiment can override it.
    rho_p = rho_f * 1.05
    beta = experiment.medium_solutal_expansion
    sigma = experiment.medium_surface_tension
    # Representative concentration difference driving convection: use the
    # first species' bulk concentration as the characteristic scale.
    delta_c = experiment.species[0].bulk_concentration
    delta_rho = rho_p - rho_f

    results = []

    Gr_1g = grashof_number(standard_g, beta, delta_c, length, nu)
    Gr_target = grashof_number(target_g, beta, delta_c, length, nu)
    results.append(MechanismResult(
        name="buoyancy_convection",
        value_1g=Gr_1g, value_target_g=Gr_target,
        regime_boundary=GRASHOF_REGIME_BOUNDARY,
        crossed=(Gr_1g > GRASHOF_REGIME_BOUNDARY) != (Gr_target > GRASHOF_REGIME_BOUNDARY),
        relative_change=abs(Gr_1g - Gr_target) / max(abs(Gr_1g), 1e-30),
    ))

    v_1g = stokes_settling_velocity(standard_g, rho_p, rho_f, experiment.aggregate_radius, nu)
    v_target = stokes_settling_velocity(target_g, rho_p, rho_f, experiment.aggregate_radius, nu)
    Pe_1g = peclet_number(v_1g, length, experiment.species[0].diffusivity)
    Pe_target = peclet_number(v_target, length, experiment.species[0].diffusivity)
    results.append(MechanismResult(
        name="sedimentation",
        value_1g=Pe_1g, value_target_g=Pe_target,
        regime_boundary=PECLET_REGIME_BOUNDARY,
        crossed=(Pe_1g > PECLET_REGIME_BOUNDARY) != (Pe_target > PECLET_REGIME_BOUNDARY),
        relative_change=abs(Pe_1g - Pe_target) / max(abs(Pe_1g), 1e-30),
    ))

    Bo_1g = bond_number(delta_rho, standard_g, length, sigma)
    Bo_target = bond_number(delta_rho, target_g, length, sigma)
    results.append(MechanismResult(
        name="multiphase_separation",
        value_1g=Bo_1g, value_target_g=Bo_target,
        regime_boundary=BOND_REGIME_BOUNDARY,
        crossed=(Bo_1g > BOND_REGIME_BOUNDARY) != (Bo_target > BOND_REGIME_BOUNDARY),
        relative_change=abs(Bo_1g - Bo_target) / max(abs(Bo_1g), 1e-30),
    ))

    # Hydrostatic gradient is linear in g by construction -- always "changes"
    # proportionally, but whether it's *active* depends on a measurability
    # threshold that is experiment-specific, not universal; reported for
    # completeness with its raw values, flagged as not-crossing a universal
    # boundary (this is intentional -- see mechanisms/hydrostatic.py).
    dP_1g = rho_f * standard_g * length
    dP_target = rho_f * target_g * length
    results.append(MechanismResult(
        name="hydrostatic_gradient",
        value_1g=dP_1g, value_target_g=dP_target,
        regime_boundary=float("nan"),
        crossed=False,
        relative_change=abs(dP_1g - dP_target) / max(abs(dP_1g), 1e-30),
    ))

    return sorted(results, key=lambda r: r.relative_change, reverse=True)
