"""Units and helpers. Everything inside the package is in SI; eV is only for display."""

import numpy as np
from scipy import constants as sc

FloatOrArray = float | np.ndarray

# FWHM of a Gaussian divided by its standard deviation.
FWHM_PER_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


def ev_to_joule(energy_ev: FloatOrArray) -> FloatOrArray:
    """Convert an energy from eV to joules."""
    return energy_ev * sc.e


def joule_to_ev(energy_j: FloatOrArray) -> FloatOrArray:
    """Convert an energy from joules to eV."""
    return energy_j / sc.e


def angular_frequency(wavelength: FloatOrArray) -> FloatOrArray:
    """Angular frequency (rad/s) of light with vacuum wavelength `wavelength` (m)."""
    return 2.0 * np.pi * sc.c / wavelength


def photon_energy(wavelength: FloatOrArray) -> FloatOrArray:
    """Photon energy hbar*omega (J) of light with vacuum wavelength `wavelength` (m)."""
    return sc.hbar * angular_frequency(wavelength)


def fwhm_to_sigma(fwhm: FloatOrArray) -> FloatOrArray:
    """Standard deviation of a Gaussian from its FWHM, in the same units."""
    return fwhm / FWHM_PER_SIGMA
