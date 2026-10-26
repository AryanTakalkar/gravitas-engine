"""Finite-volume discretisation of the 1D radial/curvilinear diffusion operator.

Supports:
  - "sphere":   (1/R^2) d/dR(R^2 dC/dR) -- spherical cell aggregate / organoid / spheroid
  - "cylinder": (1/R) d/dR(R dC/dR)     -- cylindrical bioprinted filament / hollow fiber
  - "slab":     d^2C/dR^2                -- planar tissue slice / bilayer construct

Deterministic, fixed grid of N cells, uniform spacing, second-order central differencing.
"""
from __future__ import annotations

import numpy as np

SUPPORTED_GEOMETRIES = ("sphere", "cylinder", "slab")


class RadialGrid:
    """Uniform 1D radial/transverse grid on R in [0, 1] (dimensionless), N cell centres."""

    def __init__(self, n_cells: int = 100):
        if n_cells < 10:
            raise ValueError("n_cells must be >= 10 for a meaningful spatial resolution")
        self.n_cells = n_cells
        self.dR = 1.0 / n_cells
        # Cell centres, offset by half a cell from R=0 and R=1
        self.R = (np.arange(n_cells) + 0.5) * self.dR
        # Cell faces (n_cells + 1 of them), R=0 ... R=1
        self.R_faces = np.arange(n_cells + 1) * self.dR


def laplacian_1d(C: np.ndarray, grid: RadialGrid, H: float, geometry: str = "sphere") -> np.ndarray:
    """1D curvilinear Laplacian with symmetry at R=0 and Robin boundary condition at R=1.

    Boundary conditions:
      - R=0 (symmetry): zero diffusive flux.
      - R=1 (Robin): outward flux = H * (1 - C_surface).
    """
    if geometry == "sphere":
        p = 2
    elif geometry == "cylinder":
        p = 1
    elif geometry == "slab":
        p = 0
    else:
        raise ValueError(f"unknown geometry {geometry!r}; choose one of {SUPPORTED_GEOMETRIES}")

    n = grid.n_cells
    dR = grid.dR
    R_faces = grid.R_faces

    flux = np.zeros(n + 1)

    # Internal faces: central difference of dC/dR, area-weighted by R_face^p
    for i in range(1, n):
        dCdR = (C[i] - C[i - 1]) / dR
        flux[i] = (R_faces[i] ** p) * dCdR

    # R=0 face: symmetry -> zero flux
    flux[0] = 0.0

    # R=1 face: Robin condition. Outward flux per unit area = H * (1 - C_surface).
    C_surface = C[-1] + (C[-1] - C[-2]) * 0.5 if n > 1 else C[-1]
    # At R=1: (1.0)^p = 1.0
    flux[n] = (R_faces[n] ** p) * H * (1.0 - C_surface)

    dCdt = np.zeros(n)
    for i in range(n):
        cell_volume = (grid.R[i] ** p) * dR if p > 0 else dR
        dCdt[i] = (flux[i + 1] - flux[i]) / cell_volume

    return dCdt


def laplacian_spherical(C: np.ndarray, grid: RadialGrid, H: float) -> np.ndarray:
    """Spherical radial Laplacian (1/R^2) d/dR(R^2 dC/dR) (backward-compatible)."""
    return laplacian_1d(C, grid, H, geometry="sphere")
