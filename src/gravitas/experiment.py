"""Core data structures for GRAVITAS experiments.

Deterministic by construction: an Experiment is a plain, immutable, serialisable
description of a biological experiment. No randomness, no hidden defaults --
everything a screen() call needs must be present on this object or traceable
to the parameter registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import yaml


@dataclass(frozen=True)
class Species:
    """A diffusing/reacting chemical species (oxygen, glucose, a metabolite, ...)."""

    name: str
    diffusivity: float          # m^2/s
    michaelis_constant: float   # mol/m^3
    max_uptake_rate: float      # mol/(m^3 s); negative = net production
    bulk_concentration: float   # mol/m^3, concentration in the surrounding medium
    measurement_uncertainty: float # fractional uncertainty (e.g. 0.05 for 5%)


@dataclass(frozen=True)
class Experiment:
    """A fully-specified biological experiment to be screened."""

    name: str
    aggregate_radius: float          # m
    duration: float                  # s
    temperature: float                # K
    species: tuple[Species, ...]
    medium_density: float             # kg/m^3
    medium_kinematic_viscosity: float  # m^2/s
    medium_solutal_expansion: float    # m^3/mol
    medium_surface_tension: float      # N/m
    membrane_permeability_1g: dict     # species name -> m/s at 1g

    @staticmethod
    def from_yaml(path: str) -> "Experiment":
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh)

        species = tuple(
            Species(
                name=s["name"],
                diffusivity=float(s["diffusivity"]),
                michaelis_constant=float(s["michaelis_constant"]),
                max_uptake_rate=float(s["max_uptake_rate"]),
                bulk_concentration=float(s["bulk_concentration"]),
                measurement_uncertainty=float(s.get("measurement_uncertainty", 0.05)),
            )
            for s in raw["species"]
        )

        return Experiment(
            name=raw["name"],
            aggregate_radius=float(raw["aggregate_radius"]),
            duration=float(raw["duration"]),
            temperature=float(raw["temperature"]),
            species=species,
            medium_density=float(raw["medium_density"]),
            medium_kinematic_viscosity=float(raw["medium_kinematic_viscosity"]),
            medium_solutal_expansion=float(raw["medium_solutal_expansion"]),
            medium_surface_tension=float(raw["medium_surface_tension"]),
            membrane_permeability_1g=dict(raw["membrane_permeability_1g"]),
        )
