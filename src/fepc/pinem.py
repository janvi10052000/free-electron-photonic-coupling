"""PINEM model: coupling constant, sideband probabilities, spectra and averaging over |g|.

The theory and conventions are in docs/theory.md, Phase 1. The real field is
Re[E_z(z) exp(-i omega t)], so |E_z| is the peak real field. All quantities are in SI
units. Poisson counting noise joins this module in Phase 4.
"""

import numpy as np
from scipy import constants as sc
from scipy.integrate import trapezoid
from scipy.linalg import expm
from scipy.special import jn_zeros, jv

from fepc.constants import FWHM_PER_SIGMA, FloatOrArray, fwhm_to_sigma

# |g| at which P_0 first vanishes: half the first zero of J_0 (dimensionless).
G_AT_P0_ZERO = jn_zeros(0, 1)[0] / 2.0


def coupling_constant(z: np.ndarray, e_z: np.ndarray, omega: float, v: float) -> complex:
    """Coupling constant g (dimensionless) from the field along the electron's path.

    Uses the trapezoid rule. If the field goes smoothly to zero at both ends of z,
    10 points per period of exp(-i omega z / v) are enough, as long as the grid also
    resolves E_z itself. At a hard edge the error falls only as step^2: 10 points per
    period give 3% in |g| for a field with no phase of its own, and 3e-4 at 10% from
    phase matching, so use 100 there. Keep both end points, as np.linspace does.

    Args:
        z: Positions along the path (m).
        e_z: Complex field amplitude at those positions (V/m), peak-amplitude convention.
        omega: Angular frequency of the light (rad/s).
        v: Electron speed (m/s).
    """
    integrand = e_z * np.exp(-1j * omega * z / v)
    return sc.e / (2.0 * sc.hbar * omega) * trapezoid(integrand, z)


def max_sideband_order(g_abs: float) -> int:
    """Largest |n| worth keeping for coupling strength `g_abs` (dimensionless).

    J_n(x) falls off very fast once |n| > x, over a range that widens like x^(1/3).
    The constants keep the neglected probability below 1e-12, which the tests check.
    """
    x = 2.0 * abs(g_abs)
    return int(np.ceil(x + 5.0 * np.cbrt(x) + 4.0))


def sideband_orders(g_abs: float) -> np.ndarray:
    """Integer sideband orders -N, ..., N needed for coupling strength `g_abs` (dimensionless)."""
    n_max = max_sideband_order(g_abs)
    return np.arange(-n_max, n_max + 1)


def sideband_probabilities(g: complex | np.ndarray, n: int | np.ndarray) -> FloatOrArray:
    """Probability P_n = J_n^2(2|g|) of a net exchange of n photons (dimensionless).

    Arrays follow NumPy broadcasting, so equal-length g and n are paired, not crossed.
    For an average over |g|, use `average_over_g`.

    Args:
        g: Coupling constant (dimensionless). Only its modulus matters.
        n: Sideband order. Positive n means the electron gains energy.
    """
    return jv(n, 2.0 * np.abs(g)) ** 2


def zero_loss_peak(energy: FloatOrArray, fwhm: float) -> FloatOrArray:
    """Gaussian zero-loss peak Z(E) (1/J), normalized over energy.

    Args:
        energy: Energy measured from the centre of the peak (J).
        fwhm: Full width at half maximum (J).
    """
    sigma = fwhm_to_sigma(fwhm)
    return np.exp(-(energy**2) / (2.0 * sigma**2)) / (sigma * np.sqrt(2.0 * np.pi))


def spectrum_from_probabilities(
    energy: np.ndarray, n: np.ndarray, probabilities: np.ndarray, hbar_omega: float, fwhm: float
) -> np.ndarray:
    """Sum of zero-loss peaks centred at n*hbar_omega, weighted by probability (1/J).

    Args:
        energy: 1D energy axis, measured from the zero-loss peak, positive for gain (J).
            Values are exact on any axis. A step of sigma/2, where sigma is the peak's
            standard deviation, gives integrals and moments to rounding error. Use
            sigma/10 only for smooth plots, since a peak can otherwise read up to 3% low.
        n: 1D array of sideband orders.
        probabilities: Probability of each order (dimensionless).
        hbar_omega: Photon energy (J).
        fwhm: Full width at half maximum of the zero-loss peak (J).
    """
    # n[:, np.newaxis] is a column, so `peaks` has one row per sideband and one column
    # per energy. The @ then adds the rows, each weighted by its probability.
    peaks = zero_loss_peak(energy - n[:, np.newaxis] * hbar_omega, fwhm)
    return probabilities @ peaks


def spectrum(energy: np.ndarray, g: complex, hbar_omega: float, fwhm: float) -> np.ndarray:
    """Spectrum S(E) (1/J) for one value of the coupling constant g (dimensionless).

    `energy`, `hbar_omega` and `fwhm` are in J, as in `spectrum_from_probabilities`.
    """
    n = sideband_orders(abs(g))
    return spectrum_from_probabilities(energy, n, sideband_probabilities(g, n), hbar_omega, fwhm)


def average_over_g(g_values: np.ndarray, weights: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Incoherently averaged probabilities for sideband orders `n` (dimensionless).

    Any distribution of |g| is passed as a list of values with weights, which turns
    the average into a plain sum. It covers a finite beam and a pulse envelope alike.

    Args:
        g_values: Values of g or |g| (dimensionless).
        weights: Probability of each value. Must sum to 1.
        n: 1D array of sideband orders.
    """
    if not np.isclose(np.sum(weights), 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("weights must sum to 1")
    # One row per sideband and one column per value of g. The @ then adds the columns,
    # each weighted by its probability, which leaves one number per sideband.
    return sideband_probabilities(g_values, n[:, np.newaxis]) @ weights


def averaged_spectrum(
    energy: np.ndarray, g_values: np.ndarray, weights: np.ndarray, hbar_omega: float, fwhm: float
) -> np.ndarray:
    """Spectrum (1/J) averaged over a distribution of |g|.

    `energy`, `hbar_omega` and `fwhm` are in J, as in `spectrum_from_probabilities`.
    `g_values` and `weights` are as in `average_over_g`.
    """
    n = sideband_orders(np.max(np.abs(g_values)))
    probabilities = average_over_g(g_values, weights, n)
    return spectrum_from_probabilities(energy, n, probabilities, hbar_omega, fwhm)


def pulse_envelope_distribution(
    g0: float, envelope_width: float, arrival_spread: float
) -> tuple[np.ndarray, np.ndarray]:
    """Values and weights of |g| for a Gaussian pulse sampled by Gaussian arrival times.

    The coupling follows |g|(tau) = g0 exp(-tau^2 / (2 T^2)). This assumes the envelope
    does not slip past the electron during the interaction (docs/theory.md, section 1.8).

    We use a uniform grid in arrival time rather than in |g|, because the distribution
    of |g| diverges at the pulse peak while everything is smooth in time. For a smooth
    integrand that decays at both ends, a plain sum on a uniform grid converges very fast.

    Args:
        g0: Peak coupling strength (dimensionless).
        envelope_width: T, the width of the envelope of |g| (s).
        arrival_spread: Standard deviation of the arrival times, centred on the peak (s).

    Returns:
        The values of |g| and their weights, ready for `average_over_g`.
    """
    if arrival_spread == 0.0:
        return np.array([g0]), np.array([1.0])
    # J_n^2(2|g|) oscillates in tau with a period of at least about 2.6 T / g0, so this
    # step gives over 25 points per oscillation, and it also resolves the arrival spread.
    step = min(arrival_spread, envelope_width / (1.0 + g0)) / 10.0
    half_range = 8.0 * arrival_spread
    tau = np.arange(-half_range, half_range + step, step)
    weights = np.exp(-(tau**2) / (2.0 * arrival_spread**2))
    return g0 * np.exp(-(tau**2) / (2.0 * envelope_width**2)), weights / np.sum(weights)


def sideband_merge_fwhm(hbar_omega: FloatOrArray) -> FloatOrArray:
    """FWHM (J) at which two equal peaks spaced by hbar_omega (J) lose the dip between them.

    The dip disappears when sigma = hbar_omega / 2. This is a rule of thumb for one
    isolated pair, not a limit on a real spectrum (docs/theory.md, section 1.6).
    """
    return FWHM_PER_SIGMA * hbar_omega / 2.0


def recoil_wavenumber(v: float, omega: float) -> float:
    """Recoil wavenumber k_T = hbar omega^2 / (2 gamma^3 m v^3) = 2 pi / z_T (rad/m).

    Args:
        v: Electron speed (m/s).
        omega: Angular frequency of the light (rad/s).
    """
    gamma = 1.0 / np.sqrt(1.0 - (v / sc.c) ** 2)
    return sc.hbar * omega**2 / (2.0 * gamma**3 * sc.m_e * v**3)


def sideband_amplitudes_with_recoil(
    h_l: complex, k_t_l: float, mismatch_l: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Sideband orders and amplitudes after a uniform wave, including recoil (dimensionless).

    Solves the coupled equations of docs/theory.md, section 1.8, for an electron that
    starts in order 0. Their coefficients do not depend on z, so the solution is a
    matrix exponential. This is a one-dimensional model, for checking when the Bessel
    result can be trusted.

    Args:
        h_l: The product h L, which is the coupling constant g of a phase-matched wave.
        k_t_l: The recoil wavenumber times the length, k_T L.
        mismatch_l: The wavenumber offset from phase matching times the length.
    """
    n = sideband_orders(abs(h_l))
    free_evolution = np.diag(-1j * (k_t_l * n**2 + mismatch_l * n))
    gain = -h_l * np.eye(len(n), k=-1)  # Couples order n-1 into order n.
    loss = np.conj(h_l) * np.eye(len(n), k=1)  # Couples order n+1 into order n.
    amplitude_in = (n == 0).astype(complex)
    return n, expm(free_evolution + gain + loss) @ amplitude_in
