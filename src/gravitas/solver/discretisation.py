"""Finite-volume discretisation of the spherical radial diffusion operator.

Deterministic, no adaptive/random elements: a fixed radial grid of N cells,
uniform spacing, second-order central differencing for the diffusive flux.
"""
from __future__ import annotations

import numpy as np


class RadialGrid:
    """Uniform 1D radial grid on R in [0, 1] (dimensionless), N cell centres."""

    def __init__(self, n_cells: int = 100):
        if n_cells < 10:
            raise ValueError("n_cells must be >= 10 for a meaningful spatial resolution")
        self.n_cells = n_cells
        self.dR = 1.0 / n_cells
        # Cell centres, offset by half a cell from R=0 and R=1
        self.R = (np.arange(n_cells) + 0.5) * self.dR
        # Cell faces (n_cells + 1 of them), R=0 ... R=1
        self.R_faces = np.arange(n_cells + 1) * self.dR


def laplacian_spherical(C: np.ndarray, grid: RadialGrid, H: float) -> np.ndarray:
    """Spherical radial Laplacian (1/R^2) d/dR(R^2 dC/dR), finite-volume form.

    Boundary conditions:
      - R=0 (symmetry): zero diffusive flux, enforced by a ghost cell mirror.
      - R=1 (membrane, Robin): flux = H * (1 - C_face), i.e. C'(1) + H*C(1) = H
        in the Lin/McElwain normalisation (bulk concentration = 1 outside).
    """
    n = grid.n_cells
    dR = grid.dR
    R_faces = grid.R_faces

    flux = np.zeros(n + 1)

    # Internal faces: central difference of dC/dR, area-weighted by R_face^2
    for i in range(1, n):
        dCdR = (C[i] - C[i - 1]) / dR
        flux[i] = R_faces[i] ** 2 * dCdR

    # R=0 face: symmetry -> zero flux
    flux[0] = 0.0

    # R=1 face: Robin condition. Flux (outward, per unit area) = H * (1 - C_surface).
    # Extrapolate the surface concentration from the last cell centre.
    C_surface = C[-1] + (C[-1] - C[-2]) * 0.5 if n > 1 else C[-1]
    flux[n] = R_faces[n] ** 2 * H * (1.0 - C_surface)

    dCdt = np.zeros(n)
    for i in range(n):
        # Cell-centred finite volume: divide net flux by cell "volume" R^2 dR
        cell_volume = grid.R[i] ** 2 * dR
        dCdt[i] = (flux[i + 1] - flux[i]) / cell_volume

    return dCdt
