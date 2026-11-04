"""Phase matching and mismatch between a free electron and a guided mode.

The theory is in docs/theory.md, section 2.3. Energies are in joules and lengths in m.
The classical and quantum figures of merit join this module in Phase 3.
"""

import numpy as np
from scipy import constants as sc

from fepc.constants import FloatOrArray
from fepc.slab import tm0_thickness_for_neff


def lorentz_factor(kinetic_energy: FloatOrArray) -> FloatOrArray:
    """Lorentz factor gamma (dimensionless) of an electron with kinetic energy (J)."""
    return 1.0 + kinetic_energy / (sc.m_e * sc.c**2)


def beta_from_kinetic_energy(kinetic_energy: FloatOrArray) -> FloatOrArray:
    """Speed v/c (dimensionless) of an electron with kinetic energy (J)."""
    return np.sqrt(1.0 - 1.0 / lorentz_factor(kinetic_energy) ** 2)


def kinetic_energy_from_beta(beta: FloatOrArray) -> FloatOrArray:
    """Kinetic energy (J) of an electron with speed v/c = beta (dimensionless)."""
    return sc.m_e * sc.c**2 * (1.0 / np.sqrt(1.0 - beta**2) - 1.0)


def phase_matched_kinetic_energy(n_eff: FloatOrArray) -> FloatOrArray:
    """Kinetic energy (J) at which the electron matches a mode of index n_eff (dimensionless)."""
    return kinetic_energy_from_beta(1.0 / n_eff)


def phase_matched_thickness(
    kinetic_energy: float, wavelength: float, n_core: float, n_sub: float = 1.0, n_top: float = 1.0
) -> float:
    """Core thickness (m) whose TM0 mode is phase-matched to an electron of kinetic energy (J).

    Raises:
        ValueError: If no TM0 mode of these materials can reach that speed.
    """
    n_eff = 1.0 / beta_from_kinetic_energy(kinetic_energy)
    return tm0_thickness_for_neff(wavelength, n_eff, n_core, n_sub, n_top)


def decay_length_at_phase_matching(beta: FloatOrArray, wavelength: float) -> FloatOrArray:
    """Decay length 1/kappa = beta gamma lambda / (2 pi) (m) of the field in air."""
    gamma = 1.0 / np.sqrt(1.0 - beta**2)
    return beta * gamma * wavelength / (2.0 * np.pi)


def phase_mismatch(omega: float, beta: float, n_eff: float) -> float:
    """Phase mismatch dk = (omega / c) (1 / beta - n_eff) (rad/m)."""
    return omega / sc.c * (1.0 / beta - n_eff)


def mismatch_amplitude(dk: FloatOrArray, length: float) -> FloatOrArray:
    """Length factor L sinc(dk L / 2) (m) that a phase mismatch dk (rad/m) leaves in g.

    Here sinc(x) = sin(x) / x. NumPy's sinc(y) is sin(pi y) / (pi y), hence the division by pi.
    """
    return length * np.sinc(dk * length / (2.0 * np.pi))
