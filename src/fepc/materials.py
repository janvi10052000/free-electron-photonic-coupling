"""Refractive indices from Sellmeier models, and the group index.

Wavelengths are in metres. The coefficients are in micrometres, as published. Sources
and validity ranges are in docs/theory.md, section 2.1.
"""

from collections.abc import Callable

import numpy as np

from fepc.constants import FloatOrArray

# Pairs (B, C) in n^2 = 1 + sum of B L^2 / (L^2 - C^2), with L and C in micrometres.
SILICON_TERMS = ((10.6684293, 0.301516485), (0.0030434748, 1.13475115), (1.54133408, 1104.0))
SILICON_NITRIDE_TERMS = ((3.0249, 0.1353406), (40314.0, 1239.842))
SILICA_TERMS = ((0.6961663, 0.0684043), (0.4079426, 0.1162414), (0.8974794, 9.896161))


def sellmeier_index(
    wavelength: FloatOrArray, terms: tuple[tuple[float, float], ...]
) -> FloatOrArray:
    """Refractive index (dimensionless) from Sellmeier terms, at a vacuum wavelength (m)."""
    l_squared = (wavelength * 1e6) ** 2
    n_squared = 1.0
    for strength, resonance in terms:
        n_squared = n_squared + strength * l_squared / (l_squared - resonance**2)
    return np.sqrt(n_squared)


def silicon_index(wavelength: FloatOrArray) -> FloatOrArray:
    """Refractive index of silicon at a vacuum wavelength (m), valid from 1.357 to 11.04 um."""
    return sellmeier_index(wavelength, SILICON_TERMS)


def silicon_nitride_index(wavelength: FloatOrArray) -> FloatOrArray:
    """Refractive index of Si3N4 at a vacuum wavelength (m), valid from 0.310 to 5.504 um."""
    return sellmeier_index(wavelength, SILICON_NITRIDE_TERMS)


def silica_index(wavelength: FloatOrArray) -> FloatOrArray:
    """Refractive index of fused silica at a vacuum wavelength (m), valid from 0.21 to 3.71 um."""
    return sellmeier_index(wavelength, SILICA_TERMS)


def air_index(wavelength: FloatOrArray) -> float:
    """Refractive index of the air above a chip, taken as 1 at any wavelength (m)."""
    return 1.0


def group_index(
    index: Callable[[float], float], wavelength: float, step: float = 0.1e-9
) -> float:
    """Group index n - lambda dn/dlambda (dimensionless), by a central difference.

    A central difference cancels the first error term, leaving an error that falls as
    step^2. A step of 0.1 nm is small enough for that and large enough to avoid rounding.

    Args:
        index: Any function that gives the index at a vacuum wavelength (m).
        wavelength: Vacuum wavelength (m).
        step: Wavelength step on either side (m).
    """
    slope = (index(wavelength + step) - index(wavelength - step)) / (2.0 * step)
    return index(wavelength) - wavelength * slope
