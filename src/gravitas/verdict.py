"""Verdict logic (Objective 4) and the top-level screen() entry point.

A FAIL carries the same provenance/reasoning as a PASS -- this is required
by the brief and is good practice: a well-argued null result is a real
finding, not something to hide.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .experiment import Experiment
from .classifier import classify, MechanismResult
from .solver.kinetics import dimensionless_groups
from .solver.reaction_diffusion import solve_steady_state
from .registry.provenance import get_registry_provenance

STANDARD_G = 9.80665  # m/s^2, CODATA


@dataclass
class ObservableResult:
    species: str
    surface_concentration_1g: float     # dimensionless, normalised by bulk
    surface_concentration_target_g: float
    center_concentration_1g: float
    center_concentration_target_g: float
    delta_center: float                  # |center_1g - center_target_g|


@dataclass
class Verdict:
    experiment_name: str
    target_g: float
    mechanism_ranking: list[MechanismResult]
    observables: list[ObservableResult]
    result: str                    # "PASS" | "MARGINAL" | "FAIL"
    limiting_mechanism: str
    limiting_reason: str
    uncertainty_threshold: float
    provenance: dict[str, str]


def _membrane_permeability_at_g(H_1g: float, g: float, standard_g: float,
                                 convection_floor_fraction: float = 0.1) -> float:
    """Couple gravity level to the solver's Robin boundary coefficient H.

    H (mass-transfer / Sherwood-derived permeability) is set by convective
    boundary-layer thinning at 1g. As g -> target, convection weakens (see
    mechanisms/convection.py) and H relaxes toward a purely diffusive floor
    value. Modelled here as a linear interpolation in g, floored at
    `convection_floor_fraction` of the 1g value (pure-diffusion boundary
    layer never fully vanishes) -- this is the explicit, auditable coupling
    described in Section 4.3 of the project spec.
    """
    g_ratio = max(min(g / standard_g, 1.0), 0.0)
    floor = H_1g * convection_floor_fraction
    return floor + (H_1g - floor) * g_ratio


def screen(experiment: Experiment, target_g: float,
           marginal_band: float = 1.5,
           standard_g: float = STANDARD_G) -> Verdict:
    """Screen `experiment` for gravity sensitivity at `target_g` (m/s^2).

    uncertainty_fraction: measurement uncertainty + biological variability,
        expressed as a fraction of the 1g observable value (a simple,
        explicit uncertainty model -- swap in a per-species value from real
        instrument/assay specs for a production screening run).
    marginal_band: a predicted change between 1x and `marginal_band`x the
        uncertainty threshold is reported as MARGINAL rather than a hard
        PASS/FAIL.
    """
    mechanism_ranking = classify(experiment, standard_g=standard_g, target_g=target_g)

    observables: list[ObservableResult] = []
    species_uncertainties = {}
    for species in experiment.species:
        species_uncertainties[species.name] = species.measurement_uncertainty
        alpha, K = dimensionless_groups(species, experiment.aggregate_radius)
        H_1g = experiment.membrane_permeability_1g[species.name] \
            * experiment.aggregate_radius / species.diffusivity  # dimensionless H = h*a/D
        H_target = _membrane_permeability_at_g(H_1g, target_g, standard_g)

        _, C_1g = solve_steady_state(alpha=alpha, K=K, H=H_1g)
        _, C_target = solve_steady_state(alpha=alpha, K=K, H=H_target)

        observables.append(ObservableResult(
            species=species.name,
            surface_concentration_1g=float(C_1g[-1]),
            surface_concentration_target_g=float(C_target[-1]),
            center_concentration_1g=float(C_1g[0]),
            center_concentration_target_g=float(C_target[0]),
            delta_center=abs(float(C_1g[0]) - float(C_target[0])),
        ))

    # Verdict: compare the largest predicted center-concentration change
    # against the uncertainty threshold, using the 1g value as the reference
    # scale for the uncertainty fraction.
    worst = max(observables, key=lambda o: o.delta_center)
    uncertainty_fraction = species_uncertainties[worst.species]
    threshold = uncertainty_fraction * max(worst.center_concentration_1g, 1e-30)

    if worst.delta_center > marginal_band * threshold:
        result = "PASS"
        reason = (f"Predicted center-concentration change for {worst.species} "
                  f"({worst.delta_center:.4g}) exceeds {marginal_band}x the "
                  f"uncertainty threshold ({threshold:.4g}).")
    elif worst.delta_center > threshold:
        result = "MARGINAL"
        reason = (f"Predicted center-concentration change for {worst.species} "
                  f"({worst.delta_center:.4g}) is within the marginal band "
                  f"around the uncertainty threshold ({threshold:.4g}).")
    else:
        result = "FAIL"
        reason = (f"Predicted center-concentration change for {worst.species} "
                  f"({worst.delta_center:.4g}) is below the uncertainty "
                  f"threshold ({threshold:.4g}) -- this experiment is not "
                  f"expected to show a measurable gravity-sensitive effect "
                  f"at target g={target_g:.3g} m/s^2 via this mechanism.")

    return Verdict(
        experiment_name=experiment.name,
        target_g=target_g,
        mechanism_ranking=mechanism_ranking,
        observables=observables,
        result=result,
        limiting_mechanism=worst.species,
        limiting_reason=reason,
        uncertainty_threshold=threshold,
        provenance=get_registry_provenance(),
    )
