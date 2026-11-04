"""Validation checks for phase matching and mismatch (docs/theory.md, section 2.3).

Careful with pytest.approx: unless abs= is given, it also allows a fixed error of 1e-12,
which hides small SI quantities, so comparisons of very small numbers pass abs=0.0.
"""

import numpy as np
import pytest
from scipy import constants as sc
from scipy.integrate import trapezoid

from fepc.constants import angular_frequency, ev_to_joule, joule_to_ev, photon_energy
from fepc.coupling import (
    beta_from_kinetic_energy,
    decay_length_at_phase_matching,
    kinetic_energy_from_beta,
    lorentz_factor,
    mismatch_amplitude,
    phase_matched_kinetic_energy,
    phase_matched_thickness,
    phase_mismatch,
)
from fepc.materials import silica_index, silicon_index, silicon_nitride_index
from fepc.pinem import coupling_constant
from fepc.slab import tm0_mode, tm0_thickness_for_neff

WAVELENGTH = 1550e-9  # m
OMEGA = angular_frequency(WAVELENGTH)  # rad/s


# Checks listed in section D of the project plan.


@pytest.mark.parametrize(
    "n_eff, energy_kev", [(2.000, 79.1), (2.478, 47.5), (1.793, 104.6), (1.615, 139.8)]
)
def test_phase_matched_kinetic_energy(n_eff, energy_kev):
    assert round(joule_to_ev(phase_matched_kinetic_energy(n_eff)) / 1e3, 1) == energy_kev


def test_decay_length_at_n_eff_of_two():
    assert decay_length_at_phase_matching(0.5, WAVELENGTH) * 1e9 == pytest.approx(142.0, abs=0.5)


@pytest.mark.parametrize("energy_kev, length_nm", [(60.0, 123.0), (300.0, 304.0)])
def test_decay_length_at_the_ends_of_the_energy_range(energy_kev, length_nm):
    beta = beta_from_kinetic_energy(ev_to_joule(energy_kev * 1e3))
    assert decay_length_at_phase_matching(beta, WAVELENGTH) * 1e9 == pytest.approx(length_nm, abs=0.5)


@pytest.mark.parametrize("n_eff", [1.615, 1.793, 2.0])
def test_decay_length_matches_the_slab_solver(n_eff):
    """Find the air-clad Si slab with this n_eff, and compare its kappa with the formula."""
    n_si = silicon_index(WAVELENGTH)
    thickness = tm0_thickness_for_neff(WAVELENGTH, n_eff, n_si)
    mode = tm0_mode(WAVELENGTH, thickness, n_si)
    expected = decay_length_at_phase_matching(1.0 / n_eff, WAVELENGTH)
    assert 1.0 / mode.kappa_top == pytest.approx(expected, rel=1e-8, abs=0.0)


@pytest.mark.parametrize("dk_l", [0.5, 1.0, 3.0, 7.5, 2.0 * np.pi, 4.0 * np.pi])
def test_mismatch_amplitude_matches_the_integral(dk_l):
    """Compare L sinc(dk L / 2) with the direct integral of exp(-i dk z) along the path."""
    length = 20e-6
    dk = dk_l / length
    z = np.linspace(0.0, length, 20001)
    numerical = trapezoid(np.exp(-1j * dk * z), z)
    amplitude = mismatch_amplitude(dk, length)
    assert abs(amplitude) == pytest.approx(abs(numerical), rel=1e-6, abs=1e-14)
    # The sign matters too: the sinc is negative between its first and second zeros.
    expected = np.sin(dk_l / 2.0) / (dk_l / 2.0) * length
    assert amplitude == pytest.approx(expected, rel=1e-12, abs=1e-18)


def test_mismatch_amplitude_is_the_length_when_matched():
    assert mismatch_amplitude(0.0, 20e-6) == pytest.approx(20e-6, rel=1e-15, abs=0.0)


def test_phase_mismatch_is_zero_at_phase_matching_with_the_stated_sign():
    assert phase_mismatch(OMEGA, 0.5, 2.0) == pytest.approx(0.0, abs=1e-6)
    # An electron slower than the wave (1/beta > n_eff) gives a positive mismatch.
    expected = OMEGA / sc.c * 0.2
    assert phase_mismatch(OMEGA, 0.5, 1.8) == pytest.approx(expected, rel=1e-12)


# Further checks.


def test_beta_and_kinetic_energy_are_inverses():
    beta = np.array([0.1, 0.5, 0.7765])
    energy = kinetic_energy_from_beta(beta)
    assert beta_from_kinetic_energy(energy) == pytest.approx(beta, rel=1e-12, abs=0.0)


def test_lengths_double_with_the_wavelength():
    """All the other checks run at 1550 nm."""
    beta = np.array([0.5, 0.7765])
    double = decay_length_at_phase_matching(beta, 2.0 * WAVELENGTH)
    single = decay_length_at_phase_matching(beta, WAVELENGTH)
    assert double == pytest.approx(2.0 * single, rel=1e-12, abs=0.0)
    energy = ev_to_joule(150e3)
    double = phase_matched_thickness(energy, 2.0 * WAVELENGTH, 3.476)
    single = phase_matched_thickness(energy, WAVELENGTH, 3.476)
    assert double == pytest.approx(2.0 * single, rel=1e-12, abs=0.0)


def test_phase_matched_thickness_uses_the_top_cladding():
    energy = ev_to_joule(150e3)
    n_si, n_silica = silicon_index(WAVELENGTH), silica_index(WAVELENGTH)
    upright = phase_matched_thickness(energy, WAVELENGTH, n_si, n_silica, 1.0)
    flipped = phase_matched_thickness(energy, WAVELENGTH, n_si, 1.0, n_silica)
    assert flipped == pytest.approx(upright, rel=1e-12, abs=0.0)


def test_lorentz_factor_at_the_rest_energy():
    """An electron whose kinetic energy equals its rest energy has gamma = 2."""
    assert lorentz_factor(sc.m_e * sc.c**2) == pytest.approx(2.0, rel=1e-15, abs=0.0)


@pytest.mark.parametrize("energy_kev, n_eff", [(300.0, 1.2878), (197.35, 1.4440), (60.0, 2.2410)])
def test_index_needed_at_project_energies(energy_kev, n_eff):
    beta = beta_from_kinetic_energy(ev_to_joule(energy_kev * 1e3))
    assert 1.0 / beta == pytest.approx(n_eff, abs=1e-4)


def test_mismatched_wave_in_coupling_constant_follows_the_mismatch_amplitude():
    """Section A's integral must reduce to the sinc for a uniform wave with mismatch."""
    beta, n_eff, length, e0 = 0.56, 1.8, 20e-6, 2.4e5
    v = beta * sc.c
    z = np.linspace(0.0, length, 40001)
    k_z = n_eff * OMEGA / sc.c
    g = coupling_constant(z, e0 * np.exp(1j * k_z * z), OMEGA, v)
    dk = phase_mismatch(OMEGA, beta, n_eff)
    # The prefactor uses the photon energy, so joules cancel out here.
    expected = sc.e * e0 / (2.0 * photon_energy(WAVELENGTH)) * abs(mismatch_amplitude(dk, length))
    assert abs(g) == pytest.approx(expected, rel=1e-6)
    # With E_z proportional to exp(i k_z z), the phase of g follows exp(-i dk L / 2).
    phase = np.angle(g * np.exp(1j * dk * length / 2.0))
    assert phase == pytest.approx(0.0, abs=1e-6)


# The energy window that each platform can reach at 1550 nm (TM0).


NM = 1e-9


def window_thickness(energy_kev, core, sub=None):
    n_sub = sub(WAVELENGTH) if sub else 1.0
    energy = ev_to_joule(energy_kev * 1e3)
    return phase_matched_thickness(energy, WAVELENGTH, core(WAVELENGTH), n_sub)


def test_silicon_membrane_window():
    assert window_thickness(300.0, silicon_index) / NM == pytest.approx(191.33, abs=0.02)
    assert window_thickness(60.0, silicon_index) / NM == pytest.approx(271.16, abs=0.02)


def test_silicon_nitride_membrane_cannot_reach_below_about_79_kev():
    assert window_thickness(300.0, silicon_nitride_index) / NM == pytest.approx(365.51, abs=0.02)
    assert window_thickness(100.0, silicon_nitride_index) / NM == pytest.approx(874.83, abs=0.05)
    floor = phase_matched_kinetic_energy(silicon_nitride_index(WAVELENGTH))
    assert joule_to_ev(floor) / 1e3 == pytest.approx(79.42, abs=0.01)
    with pytest.raises(ValueError):
        window_thickness(79.0, silicon_nitride_index)


@pytest.mark.parametrize("core", [silicon_index, silicon_nitride_index])
def test_silica_substrate_caps_phase_matching_near_197_kev(core):
    cap = phase_matched_kinetic_energy(silica_index(WAVELENGTH))
    assert joule_to_ev(cap) / 1e3 == pytest.approx(197.34, abs=0.01)
    assert window_thickness(197.0, core, silica_index) > 0.0
    with pytest.raises(ValueError):
        window_thickness(198.0, core, silica_index)


def test_silicon_on_silica_window_edge():
    assert window_thickness(60.0, silicon_index, silica_index) / NM == pytest.approx(257.04, abs=0.02)


def test_silicon_nitride_on_silica_window_edge():
    thickness = window_thickness(100.0, silicon_nitride_index, silica_index)
    assert thickness / NM == pytest.approx(804.51, abs=0.05)


def test_slab_against_the_design_of_henke_et_al():
    """Their Si3N4 ring of 2 um x 650 nm is designed to phase-match at 115 keV. Their indices.

    The infinite slab should give a higher n_eff, and so a lower energy, than the real guide,
    since a finite width lowers n_eff. This is only a rough check, as the ring also curves.
    """
    mode = tm0_mode(WAVELENGTH, 650e-9, 1.9923, 1.4440)
    assert mode.n_eff == pytest.approx(1.7502, abs=1e-4)
    energy_kev = joule_to_ev(phase_matched_kinetic_energy(mode.n_eff)) / 1e3
    assert energy_kev == pytest.approx(111.6, abs=0.1)
    assert 0.0 < 115.0 - energy_kev < 5.0
