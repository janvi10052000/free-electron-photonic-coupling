"""Validation checks for the PINEM model (docs/theory.md, Phase 1)."""

import math

import numpy as np
import pytest
from scipy import constants as sc
from scipy.integrate import trapezoid
from scipy.special import ive, jn_zeros, jv

from fepc.constants import (
    FWHM_PER_SIGMA,
    angular_frequency,
    ev_to_joule,
    fwhm_to_sigma,
    joule_to_ev,
    photon_energy,
)
from fepc.pinem import (
    G_AT_P0_ZERO,
    average_over_g,
    averaged_spectrum,
    coupling_constant,
    max_sideband_order,
    pulse_envelope_distribution,
    recoil_wavenumber,
    sideband_amplitudes_with_recoil,
    sideband_merge_fwhm,
    sideband_orders,
    sideband_probabilities,
    spectrum,
    spectrum_from_probabilities,
    zero_loss_peak,
)

# Careful with pytest.approx: unless abs= is given, it also allows a fixed error of 1e-12.
# That is far larger than any SI energy (about 1e-19 J) and would hide any mistake, so
# comparisons of very small numbers pass abs=0.0.

WAVELENGTH = 1550e-9  # m
OMEGA = angular_frequency(WAVELENGTH)  # rad/s
HBAR_OMEGA = photon_energy(WAVELENGTH)  # J
J0_FIRST_ZERO = jn_zeros(0, 1)[0]
G_VALUES = [0.0, 1e-3, 0.1, 0.5, 1.0, J0_FIRST_ZERO / 2.0, 3.0, 10.0, 30.0]
ENVELOPE_WIDTH = 1e-13  # s


def energy_axis(g_max: float, fwhm: float) -> np.ndarray:
    """Energy axis (J) with step sigma/10, wide enough to hold every sideband and its tails."""
    sigma = fwhm_to_sigma(fwhm)
    half_range = max_sideband_order(g_max) * HBAR_OMEGA + 8.0 * sigma
    return np.arange(-half_range, half_range + sigma / 10.0, sigma / 10.0)


def spectrum_moments(energy: np.ndarray, s: np.ndarray) -> tuple[float, float, float]:
    """Integral, mean and variance of a spectrum."""
    integral = trapezoid(s, energy)
    mean = trapezoid(energy * s, energy)
    variance = trapezoid(energy**2 * s, energy) - mean**2
    return integral, mean, variance


# Checks listed in section A of the project plan.


@pytest.mark.parametrize("g", G_VALUES)
def test_probabilities_sum_to_one(g):
    n = sideband_orders(g)
    assert np.sum(sideband_probabilities(g, n)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("g", G_VALUES)
def test_probabilities_are_symmetric(g):
    n = np.arange(1, 60)
    assert sideband_probabilities(g, -n) == pytest.approx(sideband_probabilities(g, n), abs=0.0)


@pytest.mark.parametrize("g", G_VALUES)
def test_second_moment_is_two_g_squared(g):
    n = sideband_orders(g)
    second_moment = np.sum(n**2 * sideband_probabilities(g, n))
    assert second_moment == pytest.approx(2.0 * g**2, rel=1e-11, abs=1e-14)


def test_p0_first_vanishes_at_1_2024():
    assert round(J0_FIRST_ZERO, 4) == 2.4048
    assert round(G_AT_P0_ZERO, 4) == 1.2024
    assert G_AT_P0_ZERO == J0_FIRST_ZERO / 2.0
    assert sideband_probabilities(G_AT_P0_ZERO, 0) < 1e-28
    # P_0 falls steadily up to the zero, so no earlier zero can hide between grid points.
    g_below = np.linspace(0.0, G_AT_P0_ZERO, 5001)
    assert np.all(np.diff(sideband_probabilities(g_below, 0)) < 0.0)


@pytest.mark.parametrize("g", [0.3, 0.1, 0.01])
def test_first_sideband_is_g_squared_at_weak_coupling(g):
    for n in (1, -1):
        assert abs(sideband_probabilities(g, n) - g**2) <= g**4


def test_photon_energy_at_1550_nm():
    assert round(joule_to_ev(HBAR_OMEGA), 2) == 0.80
    # h c = 1239.841984 eV nm is exact, because h, c and e have fixed SI values.
    assert joule_to_ev(HBAR_OMEGA) == pytest.approx(1239.841984 / 1550.0, rel=1e-9)


# Further checks from docs/theory.md, sections 1.5 and 1.6.


def test_ev_conversions_use_the_exact_elementary_charge():
    assert ev_to_joule(1.0) == sc.e
    assert joule_to_ev(sc.e) == 1.0


@pytest.mark.parametrize("g", G_VALUES)
def test_first_moment_is_zero(g):
    n = sideband_orders(g)
    assert np.sum(n * sideband_probabilities(g, n)) == pytest.approx(0.0, abs=1e-12)


def test_p0_has_a_double_zero():
    delta = 1e-4
    ratio = sideband_probabilities(J0_FIRST_ZERO / 2.0 + delta, 0) / delta**2
    assert ratio == pytest.approx(4.0 * jv(1, J0_FIRST_ZERO) ** 2, rel=1e-3)
    assert 4.0 * jv(1, J0_FIRST_ZERO) ** 2 == pytest.approx(1.078, abs=1e-3)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_leading_power_law_at_weak_coupling(n):
    g = 1e-2
    expected = g ** (2 * n) / math.factorial(n) ** 2
    assert sideband_probabilities(g, n) == pytest.approx(expected, rel=1e-3, abs=0.0)


@pytest.mark.parametrize("g", [0.0, 1.0, 3.0, 30.0, 300.0])
def test_truncation_rule_loses_less_than_1e_12(g):
    n_max = max_sideband_order(g)
    tail = 2.0 * np.sum(sideband_probabilities(g, np.arange(n_max + 1, n_max + 300)))
    assert tail < 1e-12


def test_only_the_modulus_of_g_matters():
    n = np.arange(-6, 7)
    g = 0.7 * np.exp(0.4j)
    assert sideband_probabilities(g, n) == pytest.approx(sideband_probabilities(0.7, n), abs=1e-15)


# Independent routes to the sideband amplitudes and the prefactor of g.


@pytest.mark.parametrize("g", [0.7 * np.exp(0.4j), 1.2 + 0j, 3.3 * np.exp(-2j)])
def test_phase_modulation_fft_gives_the_sideband_amplitudes(g):
    """Fourier transform the phase factor directly, without any Bessel function."""
    n_samples = 4096
    theta = 2.0 * np.pi * np.arange(n_samples) / n_samples  # theta = omega * tau
    phase_factor = np.exp(np.conj(g) * np.exp(1j * theta) - g * np.exp(-1j * theta))
    fourier = np.fft.fft(phase_factor) / n_samples
    for n in range(-12, 13):
        # Sideband n varies as exp(-i n theta), so it sits in FFT bin -n. n > 0 is gain.
        amplitude = fourier[-n % n_samples]
        expected = jv(n, 2.0 * abs(g)) * np.exp(1j * n * np.angle(-g))
        assert amplitude == pytest.approx(expected, abs=1e-13)
        assert abs(amplitude) ** 2 == pytest.approx(sideband_probabilities(g, n), abs=1e-13)


BETA = 0.5
V = BETA * sc.c
K_MATCHED = OMEGA / V  # rad/m
LENGTH = 10e-6  # m
E0 = 2.4e5  # V/m, peak real field


def matched_wave(z):
    return E0 * np.exp(1j * K_MATCHED * z)


def matched_wave_with_phase(z):
    return E0 * np.exp(1j * (K_MATCHED * z + 0.7))


def mismatched_gaussian(z):
    envelope = np.exp(-((z - LENGTH / 2.0) ** 2) / (2.0 * (2e-6) ** 2))
    return E0 * envelope * np.exp(1j * 1.05 * K_MATCHED * z)


def standing_wave(z):
    return E0 * np.cos(K_MATCHED * z)


FIELDS = [matched_wave, matched_wave_with_phase, mismatched_gaussian, standing_wave]


def real_field_on_path(amplitude, z, tau):
    """Real field E(z, tau + z/v), one row per arrival time and one column per position."""
    t = tau[:, np.newaxis] + z[np.newaxis, :] / V
    return np.real(amplitude(z) * np.exp(-1j * OMEGA * t))


def real_vector_potential_on_path(amplitude, z, tau):
    """Real A_z, defined so that minus its time derivative is the real field."""
    t = tau[:, np.newaxis] + z[np.newaxis, :] / V
    return np.real(amplitude(z) * np.exp(-1j * OMEGA * t) / (1j * OMEGA))


def test_matched_wave_gain_is_force_times_distance():
    z = np.linspace(0.0, LENGTH, 2001)
    g = coupling_constant(z, matched_wave(z), OMEGA, V)
    expected = sc.e * E0 * LENGTH
    assert 2.0 * abs(g) * HBAR_OMEGA == pytest.approx(expected, rel=1e-12, abs=0.0)


def test_coupling_constant_accepts_a_non_uniform_grid():
    """The integrand is constant for a matched wave, so the trapezoid rule is exact here."""
    z = LENGTH * np.linspace(0.0, 1.0, 2001) ** 2
    g = coupling_constant(z, matched_wave(z), OMEGA, V)
    expected = sc.e * E0 * LENGTH
    assert 2.0 * abs(g) * HBAR_OMEGA == pytest.approx(expected, rel=1e-12, abs=0.0)


@pytest.mark.parametrize("amplitude", FIELDS)
def test_classical_work_fixes_the_prefactor_of_g(amplitude):
    """The largest energy change of a classical electron must be 2|g| hbar omega."""
    z = np.linspace(0.0, LENGTH, 4001)
    g = coupling_constant(z, amplitude(z), OMEGA, V)
    period = 2.0 * np.pi / OMEGA
    tau = period * np.arange(1024) / 1024
    work = -sc.e * trapezoid(real_field_on_path(amplitude, z, tau), z, axis=1)
    largest_work = 2.0 * abs(g) * HBAR_OMEGA
    assert np.max(np.abs(work)) == pytest.approx(largest_work, rel=1e-5, abs=0.0)
    # The second moment is exactly classical: <work^2> = 2 |g|^2 (hbar omega)^2.
    mean_square = 2.0 * abs(g) ** 2 * HBAR_OMEGA**2
    assert np.mean(work**2) == pytest.approx(mean_square, rel=1e-10, abs=0.0)


@pytest.mark.parametrize("amplitude", FIELDS)
def test_phase_from_real_vector_potential_gives_the_sidebands(amplitude):
    """Build the quantum phase from the real A_z, then read off the sidebands by FFT."""
    # The grid does not start at z = 0, so this also tests the phase of g.
    z = np.linspace(-0.3 * LENGTH, 0.7 * LENGTH, 4001)
    g = coupling_constant(z, amplitude(z), OMEGA, V)
    n_samples = 1024
    tau = (2.0 * np.pi / OMEGA) * np.arange(n_samples) / n_samples
    a_z = real_vector_potential_on_path(amplitude, z, tau)
    phase = -(sc.e / sc.hbar) * trapezoid(a_z, z, axis=1)
    fourier = np.fft.fft(np.exp(1j * phase)) / n_samples
    for n in range(-12, 13):
        expected = jv(n, 2.0 * abs(g)) * np.exp(1j * n * np.angle(-g))
        assert fourier[-n % n_samples] == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize("wavenumber_factor", [1.0, 0.0, -1.0])
def test_near_field_integral_against_closed_form(wavenumber_factor):
    """A sub-wavelength field tests the integral itself and the sign of the exponential.

    For E0 exp(-z^2 / 2w^2) exp(i q z), the modulus of g is known in closed form.
    Ten points per period are enough for this smooth field.
    """
    w = V / OMEGA
    q = wavenumber_factor * OMEGA / V
    step = (2.0 * np.pi * V / OMEGA) / 10.0
    z = np.arange(-40.0, 40.0 + 1.0) * step
    e_z = E0 * np.exp(-(z**2) / (2.0 * w**2)) * np.exp(1j * q * z)
    g = coupling_constant(z, e_z, OMEGA, V)
    prefactor = sc.e * E0 / (2.0 * sc.hbar * OMEGA) * np.sqrt(2.0 * np.pi) * w
    expected = prefactor * np.exp(-((q - OMEGA / V) ** 2) * w**2 / 2.0)
    assert abs(g) == pytest.approx(expected, rel=1e-10)


# The spectrum.


def test_fwhm_and_sigma_are_related_by_2_355():
    assert FWHM_PER_SIGMA == pytest.approx(2.355, abs=1e-3)
    assert fwhm_to_sigma(0.3) == pytest.approx(0.127, abs=1e-3)
    assert fwhm_to_sigma(1.0) == pytest.approx(0.425, abs=1e-3)


@pytest.mark.parametrize("fwhm_ev", [0.3, 0.6, 1.0])
def test_zero_loss_peak_has_the_requested_fwhm_and_area(fwhm_ev):
    fwhm = ev_to_joule(fwhm_ev)
    half_height = zero_loss_peak(fwhm / 2.0, fwhm) / zero_loss_peak(0.0, fwhm)
    assert half_height == pytest.approx(0.5, rel=1e-12)
    energy = energy_axis(0.0, fwhm)
    assert trapezoid(zero_loss_peak(energy, fwhm), energy) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("g", [0.0, 0.5, J0_FIRST_ZERO / 2.0, 3.0])
@pytest.mark.parametrize("fwhm_ev", [0.3, 0.6, 1.0])
def test_spectrum_moments(g, fwhm_ev):
    fwhm = ev_to_joule(fwhm_ev)
    energy = energy_axis(g, fwhm)
    integral, mean, variance = spectrum_moments(energy, spectrum(energy, g, HBAR_OMEGA, fwhm))
    assert integral == pytest.approx(1.0, abs=1e-11)
    assert joule_to_ev(mean) == pytest.approx(0.0, abs=1e-11)
    expected = fwhm_to_sigma(fwhm) ** 2 + 2.0 * g**2 * HBAR_OMEGA**2
    assert variance == pytest.approx(expected, rel=1e-11, abs=0.0)


def test_spectrum_at_zero_coupling_is_the_zero_loss_peak():
    fwhm = ev_to_joule(0.6)
    energy = energy_axis(0.0, fwhm)
    s = spectrum(energy, 0.0, HBAR_OMEGA, fwhm)
    assert s == pytest.approx(zero_loss_peak(energy, fwhm), rel=1e-12)


def test_spectrum_puts_each_sideband_at_its_photon_energy():
    """With narrow peaks, the spectrum at E = n hbar omega is P_n times the peak height."""
    fwhm = 0.1 * HBAR_OMEGA
    g = 1.0
    orders = np.arange(-3, 4)
    s = spectrum(orders * HBAR_OMEGA, g, HBAR_OMEGA, fwhm)
    expected = sideband_probabilities(g, orders) * zero_loss_peak(0.0, fwhm)
    assert s == pytest.approx(expected, rel=1e-10)


def test_positive_orders_sit_on_the_gain_side():
    """P_n is symmetric here, so only lopsided probabilities can test this sign."""
    fwhm = 0.1 * HBAR_OMEGA
    energy = np.array([HBAR_OMEGA, -HBAR_OMEGA])
    s = spectrum_from_probabilities(energy, np.array([1]), np.array([1.0]), HBAR_OMEGA, fwhm)
    assert s[0] == pytest.approx(zero_loss_peak(0.0, fwhm), rel=1e-12)
    assert s[1] < 1e-6 * s[0]


def test_spectra_accept_a_complex_coupling_constant():
    """`coupling_constant` returns a complex g, so the spectra must accept one directly."""
    fwhm = ev_to_joule(0.6)
    g = 1.3 * np.exp(0.9j)
    energy = energy_axis(abs(g), fwhm)
    s = spectrum(energy, g, HBAR_OMEGA, fwhm)
    assert s == pytest.approx(spectrum(energy, abs(g), HBAR_OMEGA, fwhm), rel=1e-12)
    g_values = np.array([g, 0.5j])
    weights = np.array([0.25, 0.75])
    s = averaged_spectrum(energy, g_values, weights, HBAR_OMEGA, fwhm)
    expected = averaged_spectrum(energy, np.abs(g_values), weights, HBAR_OMEGA, fwhm)
    assert s == pytest.approx(expected, rel=1e-12)


def test_two_equal_peaks_merge_at_0_942_ev():
    fwhm_merge = sideband_merge_fwhm(HBAR_OMEGA)
    assert joule_to_ev(fwhm_merge) == pytest.approx(0.942, abs=1e-3)
    assert fwhm_merge / HBAR_OMEGA == pytest.approx(1.177, abs=1e-3)
    assert fwhm_merge / HBAR_OMEGA == pytest.approx(np.sqrt(2.0 * np.log(2.0)), rel=1e-12)

    def has_a_dip(fwhm):
        energy = np.linspace(-HBAR_OMEGA, HBAR_OMEGA, 20001)
        pair = zero_loss_peak(energy - HBAR_OMEGA / 2.0, fwhm)
        pair = pair + zero_loss_peak(energy + HBAR_OMEGA / 2.0, fwhm)
        return pair[len(energy) // 2] < np.max(pair)

    assert has_a_dip(0.999 * fwhm_merge)
    assert not has_a_dip(1.001 * fwhm_merge)


def test_numbers_quoted_in_the_theory():
    assert joule_to_ev(HBAR_OMEGA) == pytest.approx(0.7999, abs=1e-4)
    assert (2.0 * np.pi / OMEGA) * 1e15 == pytest.approx(5.17, abs=0.01)  # optical period, fs
    # Transform-limited arrival spread hbar / (2 sigma_E) at the ends of the FWHM range, fs.
    assert sc.hbar / (2.0 * fwhm_to_sigma(ev_to_joule(0.3))) * 1e15 == pytest.approx(2.6, abs=0.05)
    assert sc.hbar / (2.0 * fwhm_to_sigma(ev_to_joule(1.0))) * 1e15 == pytest.approx(0.77, abs=0.01)
    # FWHM / hbar omega: 0.7 eV at 800 nm, then 0.3 and 1.0 eV at 1550 nm.
    assert ev_to_joule(0.7) / photon_energy(800e-9) == pytest.approx(0.45, abs=0.005)
    assert ev_to_joule(0.3) / HBAR_OMEGA == pytest.approx(0.38, abs=0.005)
    assert ev_to_joule(1.0) / HBAR_OMEGA == pytest.approx(1.25, abs=0.005)


# Incoherent averages.


@pytest.mark.parametrize("weights", [[0.5, 0.6], [0.5, 0.4]])
def test_weights_must_sum_to_one(weights):
    with pytest.raises(ValueError):
        average_over_g(np.array([1.0, 2.0]), np.array(weights), np.arange(-2, 3))


def test_average_of_two_values_is_their_weighted_sum():
    n = np.arange(-8, 9)
    averaged = average_over_g(np.array([1.3, 0.4]), np.array([0.25, 0.75]), n)
    expected = 0.25 * sideband_probabilities(1.3, n) + 0.75 * sideband_probabilities(0.4, n)
    assert averaged == pytest.approx(expected, abs=1e-15)


@pytest.mark.parametrize("ratio", [0.01, 0.3, 1.0, 3.0])
@pytest.mark.parametrize("g0", [0.0, 1.25, 3.0, 10.0])
def test_pulse_envelope_sum_rules(g0, ratio):
    g_values, weights = pulse_envelope_distribution(g0, ENVELOPE_WIDTH, ratio * ENVELOPE_WIDTH)
    n = sideband_orders(g0)
    averaged = average_over_g(g_values, weights, n)
    assert np.sum(averaged) == pytest.approx(1.0, abs=1e-11)
    assert averaged[::-1] == pytest.approx(averaged, abs=1e-15)
    expected = g0**2 / np.sqrt(1.0 + 2.0 * ratio**2)
    assert np.sum(weights * g_values**2) == pytest.approx(expected, rel=1e-10)
    assert np.sum(n**2 * averaged) == pytest.approx(2.0 * expected, rel=1e-10)


@pytest.mark.parametrize("g0, ratio", [(10.0, 3.0), (30.0, 0.01), (30.0, 3.0), (30.0, 10.0)])
def test_pulse_envelope_grid_is_converged(g0, ratio):
    """Halving the grid step must not change the result, even at strong coupling."""
    n = sideband_orders(g0)
    g_values, weights = pulse_envelope_distribution(g0, ENVELOPE_WIDTH, ratio * ENVELOPE_WIDTH)
    averaged = average_over_g(g_values, weights, n)
    tau = np.linspace(-8.0 * ratio, 8.0 * ratio, 2 * len(g_values) + 1)  # in units of T
    fine_weights = np.exp(-(tau**2) / (2.0 * ratio**2))
    fine_weights /= np.sum(fine_weights)
    fine = average_over_g(g0 * np.exp(-(tau**2) / 2.0), fine_weights, n)
    assert averaged == pytest.approx(fine, abs=1e-12)


def test_pulse_envelope_without_spread_is_the_peak_value():
    g_values, weights = pulse_envelope_distribution(2.0, ENVELOPE_WIDTH, 0.0)
    assert g_values == pytest.approx([2.0])
    assert weights == pytest.approx([1.0])


def test_finite_beam_second_moment():
    """A field decaying as exp(-kappa x), sampled by Gaussian beam positions, no clipping."""
    g0, kappa, s = 1.0, 1.0, 0.3
    x = np.linspace(-8.0 * s, 8.0 * s, 4001)
    weights = np.exp(-(x**2) / (2.0 * s**2))
    weights /= np.sum(weights)
    g_values = g0 * np.exp(-kappa * x)
    n = sideband_orders(np.max(g_values))
    expected = g0**2 * np.exp(2.0 * kappa**2 * s**2)
    assert np.sum(weights * g_values**2) == pytest.approx(expected, rel=1e-10)
    averaged = average_over_g(g_values, weights, n)
    assert np.sum(averaged) == pytest.approx(1.0, abs=1e-11)
    assert np.sum(n**2 * averaged) == pytest.approx(2.0 * expected, rel=1e-10)


def test_narrow_spread_fills_in_the_zero_of_p0():
    delta = 0.01
    deviation = np.linspace(-6.0 * delta, 6.0 * delta, 2001)
    weights = np.exp(-(deviation**2) / (2.0 * delta**2))
    weights /= np.sum(weights)
    p0_bar = average_over_g(J0_FIRST_ZERO / 2.0 + deviation, weights, np.array([0]))[0]
    assert p0_bar == pytest.approx(4.0 * jv(1, J0_FIRST_ZERO) ** 2 * delta**2, rel=1e-3, abs=0.0)


@pytest.mark.parametrize("mu", [0.25, 1.0, 4.0])
def test_rayleigh_average_against_closed_form(mu):
    # This integrand has a non-zero slope at u = 0, so a uniform grid is only second
    # order here. Its error is about step^2 * slope / 12, which is 2e-7 on this grid.
    u = np.linspace(0.0, 8.0 * np.sqrt(mu), 8001)
    weights = (2.0 * u / mu) * np.exp(-(u**2) / mu)
    n = np.arange(-6, 7)
    averaged = average_over_g(u, weights / np.sum(weights), n)
    assert averaged == pytest.approx(ive(np.abs(n), 2.0 * mu), abs=1e-6)


def test_averaged_spectrum_is_the_weighted_sum_of_spectra():
    """A wide spread: the sidebands must reach as far as the largest |g| needs."""
    fwhm = ev_to_joule(0.6)
    energy = energy_axis(30.0, fwhm)
    s = averaged_spectrum(energy, np.array([0.0, 30.0]), np.array([0.9, 0.1]), HBAR_OMEGA, fwhm)
    expected = 0.9 * spectrum(energy, 0.0, HBAR_OMEGA, fwhm)
    expected = expected + 0.1 * spectrum(energy, 30.0, HBAR_OMEGA, fwhm)
    assert s == pytest.approx(expected, rel=1e-10)


def test_averaged_spectrum_variance():
    fwhm = ev_to_joule(0.6)
    g0, ratio = 3.0, 0.7
    g_values, weights = pulse_envelope_distribution(g0, ENVELOPE_WIDTH, ratio * ENVELOPE_WIDTH)
    energy = energy_axis(g0, fwhm)
    s = averaged_spectrum(energy, g_values, weights, HBAR_OMEGA, fwhm)
    integral, mean, variance = spectrum_moments(energy, s)
    assert integral == pytest.approx(1.0, abs=1e-11)
    assert joule_to_ev(mean) == pytest.approx(0.0, abs=1e-11)
    mean_g_squared = np.sum(weights * g_values**2)
    expected = fwhm_to_sigma(fwhm) ** 2 + 2.0 * mean_g_squared * HBAR_OMEGA**2
    assert variance == pytest.approx(expected, rel=1e-10, abs=0.0)


# Recoil (docs/theory.md, section 1.8).


def electron_speed(kinetic_energy_ev: float) -> float:
    """Speed (m/s) of an electron with the given kinetic energy (eV)."""
    gamma = 1.0 + ev_to_joule(kinetic_energy_ev) / (sc.m_e * sc.c**2)
    return sc.c * np.sqrt(1.0 - 1.0 / gamma**2)


def recoil_error(g: float, k_t_l: float) -> float:
    """Half the summed change in P_n caused by recoil at phase matching (dimensionless)."""
    n, c = sideband_amplitudes_with_recoil(g, k_t_l)
    return 0.5 * np.sum(np.abs(np.abs(c) ** 2 - sideband_probabilities(g, n)))


@pytest.mark.parametrize("kinetic_energy_ev, z_t", [(60e3, 0.25), (300e3, 3.7)])
def test_talbot_distance(kinetic_energy_ev, z_t):
    k_t = recoil_wavenumber(electron_speed(kinetic_energy_ev), OMEGA)
    assert 2.0 * np.pi / k_t == pytest.approx(z_t, abs=0.05)


def test_recoil_wavenumber_matches_the_exact_dispersion():
    """Compare with momentum from W^2 = (m c^2)^2 + (p c)^2, for one sideband."""
    v = electron_speed(60e3)
    gamma = 1.0 / np.sqrt(1.0 - (v / sc.c) ** 2)
    energy = gamma * sc.m_e * sc.c**2
    momentum = gamma * sc.m_e * v

    def sideband_momentum(n):
        return np.sqrt((energy + n * HBAR_OMEGA) ** 2 - (sc.m_e * sc.c**2) ** 2) / sc.c

    # The average of orders +1 and -1 cancels the odd terms, leaving -k_T n^2.
    shift = (sideband_momentum(1) + sideband_momentum(-1) - 2.0 * momentum) / (2.0 * sc.hbar)
    assert shift == pytest.approx(-recoil_wavenumber(v, OMEGA), rel=1e-4)


def test_recoil_scale_at_1_mm():
    k_t_l = recoil_wavenumber(electron_speed(60e3), OMEGA) * 1e-3
    assert k_t_l == pytest.approx(0.0256, abs=1e-4)
    assert recoil_wavenumber(electron_speed(300e3), OMEGA) * 1e-3 < k_t_l
    # xi = 2 k_T |g| L reaches 1 at this |g|.
    assert 1.0 / (2.0 * k_t_l) == pytest.approx(20.0, abs=1.0)


def test_ladder_without_recoil_is_the_bessel_result():
    h_l = 1.7 * np.exp(0.6j)
    n, c = sideband_amplitudes_with_recoil(h_l, 0.0)
    expected = jv(n, 2.0 * abs(h_l)) * np.exp(1j * n * np.angle(-h_l))
    # The last orders kept, of size 1e-9, feel the end of the ladder, hence this tolerance.
    assert c == pytest.approx(expected, abs=1e-10)


def test_ladder_with_mismatch_without_recoil_follows_the_sinc():
    h_l, mismatch_l = 1.7, 1.3
    n, c = sideband_amplitudes_with_recoil(h_l, 0.0, mismatch_l)
    g = h_l * np.sin(mismatch_l / 2.0) / (mismatch_l / 2.0)
    assert np.abs(c) ** 2 == pytest.approx(sideband_probabilities(g, n), abs=1e-11)


@pytest.mark.parametrize("k_t_l", [1.0, 3.0])
def test_first_sideband_is_mismatched_by_recoil(k_t_l):
    g = 1e-3
    n, c = sideband_amplitudes_with_recoil(g, k_t_l)
    sinc = np.sin(k_t_l / 2.0) / (k_t_l / 2.0)
    for order in (1, -1):
        probability = np.abs(c[n == order][0]) ** 2
        assert probability == pytest.approx(g**2 * sinc**2, rel=1e-4, abs=0.0)


def test_recoil_breaks_the_symmetry_only_off_phase_matching():
    n, c = sideband_amplitudes_with_recoil(3.0, 0.03)
    p = np.abs(c) ** 2
    assert p[::-1] == pytest.approx(p, abs=1e-12)
    n, c = sideband_amplitudes_with_recoil(3.0, 0.03, mismatch_l=2.0)
    assert abs(np.sum(n * np.abs(c) ** 2)) > 1e-2


@pytest.mark.parametrize("g", [5.0, 20.0])
@pytest.mark.parametrize("xi", [0.1, 0.3])
def test_comb_width_falls_with_xi_squared(g, xi):
    n, c = sideband_amplitudes_with_recoil(g, xi / (2.0 * g))
    ratio = np.sum(n**2 * np.abs(c) ** 2) / (2.0 * g**2)
    assert 1.0 - ratio == pytest.approx(0.11 * xi**2, abs=0.01 * xi**2 + 1e-4)


def test_recoil_error_reaches_one_percent_near_g_of_6_at_60_kev_and_1_mm():
    k_t_l = recoil_wavenumber(electron_speed(60e3), OMEGA) * 1e-3
    assert recoil_error(4.0, k_t_l) < 0.01
    assert recoil_error(8.0, k_t_l) > 0.01
    assert recoil_error(6.0, k_t_l) == pytest.approx(0.01, abs=0.002)


@pytest.mark.parametrize("g", [20.0, 50.0])
def test_one_percent_recoil_error_scales_as_xi_of_1_3_over_root_2g(g):
    xi = 1.3 / np.sqrt(2.0 * g)
    assert recoil_error(g, xi / (2.0 * g)) == pytest.approx(0.01, abs=0.003)


def test_exit_density_is_modulated_by_xi():
    g, xi = 5.0, 0.1
    n, c = sideband_amplitudes_with_recoil(g, xi / (2.0 * g))
    theta = 2.0 * np.pi * np.arange(1024) / 1024  # theta = omega * tau
    wave = np.sum(c[:, np.newaxis] * np.exp(-1j * n[:, np.newaxis] * theta), axis=0)
    density = np.abs(wave) ** 2
    assert (np.max(density) - np.min(density)) / 2.0 == pytest.approx(xi, rel=0.05)
