"""Validation checks for the TM0 slab solver (docs/theory.md, section 2.2).

Careful with pytest.approx: unless abs= is given, it also allows a fixed error of 1e-12,
which hides small SI quantities, so comparisons of very small numbers pass abs=0.0.
"""

import numpy as np
import pytest
from scipy import constants as sc
from scipy.integrate import quad

from fepc.constants import angular_frequency
from fepc.materials import (
    air_index,
    group_index,
    silica_index,
    silicon_index,
    silicon_nitride_index,
)
from fepc.slab import (
    _wavenumbers_and_phases,
    tm0_cutoff_thickness,
    tm0_finite_difference,
    tm0_group_index,
    tm0_mode,
    tm0_thickness_for_neff,
)

WAVELENGTH = 1550e-9  # m
OMEGA = angular_frequency(WAVELENGTH)  # rad/s
K0 = 2.0 * np.pi / WAVELENGTH  # rad/m
N_SI = 3.476  # Fixed index used by the checks in section C of the project plan.
N_SIN = 1.996
N_SIO2 = 1.444

# Thickness (m), core index and n_eff, for the symmetric air-clad slab.
SYMMETRIC_CASES = [
    (220e-9, N_SI, 1.615),
    (300e-9, N_SI, 2.478),
    (400e-9, N_SI, 2.931),
    (800e-9, N_SIN, 1.793),
    (400e-9, N_SIN, 1.355),
]
# Thickness (m), core index and n_eff on SiO2 with air on top, from a separate transfer-matrix
# calculation that is not in this repository. The finite-difference tests below also check them.
ASYMMETRIC_CASES = [
    (220e-9, N_SI, 1.890818),
    (300e-9, N_SI, 2.553685),
    (400e-9, N_SI, 2.955972),
    (400e-9, N_SIN, 1.553882),
    (800e-9, N_SIN, 1.822188),
]
ALL_MODES = [(t, n, 1.0) for t, n, _ in SYMMETRIC_CASES] + [
    (t, n, N_SIO2) for t, n, _ in ASYMMETRIC_CASES
]


# Checks listed in section C of the project plan.


@pytest.mark.parametrize("thickness, n_core, n_eff", SYMMETRIC_CASES)
def test_symmetric_slab_values(thickness, n_core, n_eff):
    assert round(tm0_mode(WAVELENGTH, thickness, n_core).n_eff, 3) == n_eff


@pytest.mark.parametrize(
    "thickness, n_core, n_eff",
    [
        (220e-9, N_SI, 1.61491),
        (300e-9, N_SI, 2.47823),
        (400e-9, N_SI, 2.93118),
        (400e-9, N_SIN, 1.35496),
        (800e-9, N_SIN, 1.79250),
    ],
)
def test_table_in_the_theory(thickness, n_core, n_eff):
    assert round(tm0_mode(WAVELENGTH, thickness, n_core).n_eff, 5) == n_eff


@pytest.mark.parametrize("thickness, n_core, n_eff", SYMMETRIC_CASES)
def test_symmetric_slab_satisfies_the_tan_form(thickness, n_core, n_eff):
    """tan(k_x t / 2) = (n1^2 / n2^2) (kappa / k_x), on the fundamental branch."""
    mode = tm0_mode(WAVELENGTH, thickness, n_core)
    half_phase = mode.k_x * thickness / 2.0
    assert 0.0 < half_phase < np.pi / 2.0
    assert np.tan(half_phase) == pytest.approx(n_core**2 * mode.kappa_top / mode.k_x, rel=1e-9)


@pytest.mark.parametrize("n_core", [N_SI, N_SIN])
def test_thin_slab_limit(n_core):
    """As t goes to 0, n_eff - 1 goes to (k0 t)^2 (1 - 1/n1^2)^2 / 8."""
    thickness = 1e-9
    n_eff = tm0_mode(WAVELENGTH, thickness, n_core).n_eff
    expected = (K0 * thickness) ** 2 * (1.0 - 1.0 / n_core**2) ** 2 / 8.0
    assert n_eff - 1.0 == pytest.approx(expected, rel=1e-3, abs=0.0)


@pytest.mark.parametrize("n_core", [N_SI, N_SIN])
def test_thick_slab_limit(n_core):
    """As t grows, n1 - n_eff goes to lambda^2 / (8 n1 t^2)."""
    thickness = 1e-3
    n_eff = tm0_mode(WAVELENGTH, thickness, n_core).n_eff
    expected = WAVELENGTH**2 / (8.0 * n_core * thickness**2)
    assert n_core - n_eff == pytest.approx(expected, rel=1e-3, abs=0.0)


@pytest.mark.parametrize("thickness, n_core, n_eff", SYMMETRIC_CASES)
def test_finite_differences_agree_for_the_symmetric_slab(thickness, n_core, n_eff):
    analytic = tm0_mode(WAVELENGTH, thickness, n_core).n_eff
    numerical, _, _ = tm0_finite_difference(WAVELENGTH, thickness, n_core)
    assert numerical == pytest.approx(analytic, abs=5e-5)


@pytest.mark.parametrize("thickness, n_core, n_eff", ASYMMETRIC_CASES)
def test_asymmetric_slab_values(thickness, n_core, n_eff):
    assert tm0_mode(WAVELENGTH, thickness, n_core, N_SIO2).n_eff == pytest.approx(n_eff, abs=1e-6)


@pytest.mark.parametrize("thickness, n_core, n_eff", ASYMMETRIC_CASES)
def test_finite_differences_agree_for_the_asymmetric_slab(thickness, n_core, n_eff):
    analytic = tm0_mode(WAVELENGTH, thickness, n_core, N_SIO2).n_eff
    numerical, _, _ = tm0_finite_difference(WAVELENGTH, thickness, n_core, N_SIO2)
    assert numerical == pytest.approx(analytic, abs=5e-5)


@pytest.mark.parametrize("thickness, n_core, n_eff", SYMMETRIC_CASES)
def test_e_z_formula_for_the_air_clad_slab(thickness, n_core, n_eff):
    """E_z just outside = (kappa / (omega eps0)) cos(k_x t / 2) exp(-kappa d)."""
    mode = tm0_mode(WAVELENGTH, thickness, n_core)
    d = 100e-9
    expected = (
        mode.kappa_top
        / (OMEGA * sc.epsilon_0)
        * np.cos(mode.k_x * thickness / 2.0)
        * np.exp(-mode.kappa_top * d)
    )
    assert mode.e_z_above(d) == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("thickness, n_core, n_sub", ALL_MODES)
def test_power_per_width_matches_the_integral(thickness, n_core, n_sub):
    """P' = (k_z / (2 omega eps0)) * integral of |H_y|^2 / eps_r dx, by numerical integration."""
    mode = tm0_mode(WAVELENGTH, thickness, n_core, n_sub)
    half = thickness / 2.0
    # quad cannot integrate to infinity here: it would miss a field that decays over 100 nm.
    x_top = half + 40.0 / mode.kappa_top
    x_bottom = -half - 40.0 / mode.kappa_sub
    core, _ = quad(lambda x: mode.h_y(x) ** 2 / n_core**2, -half, half, epsabs=0.0, epsrel=1e-11)
    top, _ = quad(lambda x: mode.h_y(x) ** 2, half, x_top, epsabs=0.0, epsrel=1e-10, limit=200)
    bottom, _ = quad(
        lambda x: mode.h_y(x) ** 2 / n_sub**2, x_bottom, -half, epsabs=0.0, epsrel=1e-10, limit=200
    )
    expected = mode.k_z / (2.0 * OMEGA * sc.epsilon_0) * (core + top + bottom)
    assert mode.power_per_width() == pytest.approx(expected, rel=1e-8, abs=0.0)


def test_group_index_of_thick_and_thin_slabs_approaches_the_limits():
    bulk = group_index(silicon_index, WAVELENGTH)
    thick = tm0_group_index(WAVELENGTH, 100e-6, silicon_index)
    assert thick == pytest.approx(bulk, abs=1e-4)
    thin = tm0_group_index(WAVELENGTH, 1e-9, silicon_index)
    assert thin == pytest.approx(1.0, abs=1e-4)


# Further checks: the fields.


@pytest.mark.parametrize("thickness, n_core, n_sub", ALL_MODES)
def test_field_is_one_at_the_centre_and_continuous_at_the_faces(thickness, n_core, n_sub):
    mode = tm0_mode(WAVELENGTH, thickness, n_core, n_sub)
    half = thickness / 2.0
    small = 1e-13
    assert mode.h_y(0.0) == pytest.approx(1.0, abs=1e-12)
    for face in (half, -half):
        assert mode.h_y(face + small) == pytest.approx(mode.h_y(face - small), abs=1e-5)
        # The tangential E_z is continuous too, although dH_y/dx jumps by n_core^2.
        assert mode.e_z_magnitude(face + small) == pytest.approx(
            mode.e_z_magnitude(face - small), rel=1e-4
        )


@pytest.mark.parametrize("thickness, n_core, n_sub", ALL_MODES)
def test_finite_difference_field_matches_the_analytic_field(thickness, n_core, n_sub):
    mode = tm0_mode(WAVELENGTH, thickness, n_core, n_sub)
    _, x, h_y = tm0_finite_difference(WAVELENGTH, thickness, n_core, n_sub, padding=6e-6)
    assert h_y == pytest.approx(mode.h_y(x), abs=2e-5)


@pytest.mark.parametrize("thickness, n_core, n_sub", ALL_MODES)
def test_e_z_above_matches_the_slope_of_the_finite_difference_field(thickness, n_core, n_sub):
    """E_z in air is (dH_y/dx) / (omega eps0), measured here from the numerical field."""
    mode = tm0_mode(WAVELENGTH, thickness, n_core, n_sub)
    _, x, h_y = tm0_finite_difference(WAVELENGTH, thickness, n_core, n_sub, step=0.25e-9)
    d = 100e-9
    i = np.searchsorted(x, thickness / 2.0 + d)
    slope = (h_y[i + 1] - h_y[i]) / (x[i + 1] - x[i])
    midpoint = (x[i + 1] + x[i]) / 2.0
    assert abs(slope) / (OMEGA * sc.epsilon_0) == pytest.approx(
        mode.e_z_magnitude(midpoint), rel=1e-4
    )


@pytest.mark.parametrize(
    "thickness, n_core, overestimate",
    [
        (220e-9, N_SI, 1.78),
        (300e-9, N_SI, 1.65),
        (400e-9, N_SI, 1.62),
        (400e-9, N_SIN, 1.97),
        (800e-9, N_SIN, 1.85),
    ],
)
def test_air_clad_e_z_formula_overestimates_on_a_substrate(thickness, n_core, overestimate):
    """So the code must use the real value of H_y at the top face."""
    mode = tm0_mode(WAVELENGTH, thickness, n_core, N_SIO2)
    wrong = mode.kappa_top / (OMEGA * sc.epsilon_0) * np.cos(mode.k_x * thickness / 2.0)
    assert wrong / mode.e_z_above(0.0) == pytest.approx(overestimate, abs=0.01)


def test_field_decays_at_kappa_outside():
    mode = tm0_mode(WAVELENGTH, 220e-9, N_SI, N_SIO2)
    d = 100e-9
    assert mode.e_z_above(d) / mode.e_z_above(0.0) == pytest.approx(np.exp(-mode.kappa_top * d))


def test_finite_differences_converge_with_the_square_of_the_step():
    analytic = tm0_mode(WAVELENGTH, 220e-9, N_SI).n_eff
    coarse, _, _ = tm0_finite_difference(WAVELENGTH, 220e-9, N_SI, step=2e-9)
    fine, _, _ = tm0_finite_difference(WAVELENGTH, 220e-9, N_SI, step=1e-9)
    assert (coarse - analytic) / (fine - analytic) == pytest.approx(4.0, abs=0.3)


# Further checks: an asymmetric slab, cut-off and the inverse problem.


def test_equal_claddings_reduce_to_the_symmetric_slab():
    symmetric = tm0_mode(WAVELENGTH, 300e-9, N_SI, 1.2, 1.2)
    assert symmetric.phi_sub == pytest.approx(symmetric.phi_top, abs=1e-14)
    half_phase = symmetric.k_x * 300e-9 / 2.0
    ratio = (N_SI**2 / 1.2**2) * symmetric.kappa_top / symmetric.k_x
    assert np.tan(half_phase) == pytest.approx(ratio, rel=1e-9)


@pytest.mark.parametrize(
    "thickness, core_index", [(300e-9, silicon_index), (800e-9, silicon_nitride_index)]
)
def test_turning_the_slab_upside_down_mirrors_the_mode(thickness, core_index):
    """Silica below and air above is the same slab as air below and silica above, flipped.

    All the other checks have air on top, which would hide any mistake in n_top.
    """
    n_core, n_silica = core_index(WAVELENGTH), silica_index(WAVELENGTH)
    upright = tm0_mode(WAVELENGTH, thickness, n_core, n_silica, 1.0)
    flipped = tm0_mode(WAVELENGTH, thickness, n_core, 1.0, n_silica)
    assert flipped.n_eff == pytest.approx(upright.n_eff, abs=1e-12)
    x = np.array([-2.0, -0.7, -0.2, 0.2, 0.7, 2.0]) * thickness
    assert flipped.h_y(x) == pytest.approx(upright.h_y(-x), rel=1e-9, abs=0.0)
    assert flipped.e_z_magnitude(x) == pytest.approx(upright.e_z_magnitude(-x), rel=1e-9, abs=0.0)
    assert flipped.power_per_width() == pytest.approx(upright.power_per_width(), rel=1e-9, abs=0.0)
    numerical, x_grid, h_y = tm0_finite_difference(WAVELENGTH, thickness, n_core, 1.0, n_silica)
    assert numerical == pytest.approx(flipped.n_eff, abs=5e-5)
    assert h_y == pytest.approx(flipped.h_y(x_grid), abs=1e-3)
    upright_n_g = tm0_group_index(WAVELENGTH, thickness, core_index, silica_index, air_index)
    flipped_n_g = tm0_group_index(WAVELENGTH, thickness, core_index, air_index, silica_index)
    assert flipped_n_g == pytest.approx(upright_n_g, abs=1e-6)


@pytest.mark.parametrize("core_index", [silicon_index, silicon_nitride_index])
def test_the_higher_cladding_sets_the_limits_on_either_side(core_index):
    n_core, n_silica = core_index(WAVELENGTH), silica_index(WAVELENGTH)
    cutoff = tm0_cutoff_thickness(WAVELENGTH, n_core, n_silica, 1.0)
    flipped = tm0_cutoff_thickness(WAVELENGTH, n_core, 1.0, n_silica)
    assert flipped == pytest.approx(cutoff, rel=1e-12, abs=0.0)
    with pytest.raises(ValueError, match="cut-off"):
        tm0_mode(WAVELENGTH, 0.999 * cutoff, n_core, 1.0, n_silica)
    with pytest.raises(ValueError, match="between"):
        tm0_thickness_for_neff(WAVELENGTH, 1.2, n_core, 1.0, n_silica)


def test_doubling_wavelength_and_thickness_changes_only_the_lengths():
    """Maxwell's equations have no length scale. All the other checks run at 1550 nm."""
    small = tm0_mode(WAVELENGTH, 300e-9, N_SI, N_SIO2)
    big = tm0_mode(2.0 * WAVELENGTH, 600e-9, N_SI, N_SIO2)
    assert big.n_eff == pytest.approx(small.n_eff, abs=1e-10)
    assert big.e_z_above(200e-9) == pytest.approx(small.e_z_above(100e-9), rel=1e-9, abs=0.0)
    assert big.power_per_width() == pytest.approx(2.0 * small.power_per_width(), rel=1e-9, abs=0.0)
    found = tm0_thickness_for_neff(2.0 * WAVELENGTH, small.n_eff, N_SI, N_SIO2)
    assert found == pytest.approx(600e-9, rel=1e-9, abs=0.0)
    numerical, _, _ = tm0_finite_difference(2.0 * WAVELENGTH, 600e-9, N_SI, N_SIO2, step=2e-9, padding=6e-6)
    assert numerical == pytest.approx(small.n_eff, abs=5e-5)
    n_g = tm0_group_index(WAVELENGTH, 300e-9, lambda w: N_SI, lambda w: N_SIO2)
    big_n_g = tm0_group_index(2.0 * WAVELENGTH, 600e-9, lambda w: N_SI, lambda w: N_SIO2)
    assert big_n_g == pytest.approx(n_g, abs=1e-6)


def test_symmetric_slab_has_no_cutoff():
    assert tm0_cutoff_thickness(WAVELENGTH, N_SI) == 0.0
    assert tm0_mode(WAVELENGTH, 0.1e-9, N_SI).n_eff > 1.0
    with pytest.raises(ValueError, match="cut-off"):
        tm0_mode(WAVELENGTH, 0.0, N_SI)


@pytest.mark.parametrize(
    "core_index, expected",
    [(silicon_index, 103.30e-9), (silicon_nitride_index, 223.75e-9)],
)
def test_cutoff_on_silica(core_index, expected):
    n_core, n_sub = core_index(WAVELENGTH), silica_index(WAVELENGTH)
    cutoff = tm0_cutoff_thickness(WAVELENGTH, n_core, n_sub)
    assert cutoff == pytest.approx(expected, abs=0.01e-9)
    assert tm0_mode(WAVELENGTH, 1.001 * cutoff, n_core, n_sub).n_eff > n_sub
    with pytest.raises(ValueError, match="cut-off"):
        tm0_mode(WAVELENGTH, 0.999 * cutoff, n_core, n_sub)


def test_core_with_a_low_index_guides_nothing():
    with pytest.raises(ValueError, match="cut-off"):
        tm0_mode(WAVELENGTH, 300e-9, 1.2, N_SIO2)


def test_finite_differences_keep_the_requested_padding():
    _, x, _ = tm0_finite_difference(WAVELENGTH, 300e-9, N_SI, padding=1e-6)
    assert x[-1] == pytest.approx(150e-9 + 1e-6, abs=2e-9)


@pytest.mark.parametrize(
    "thickness, n_sub, padding",
    [
        (100e-9, 1.0, 3e-6),  # Too little padding for this weakly confined mode.
        (30e-9, 1.0, 20e-6),  # Even more weakly confined.
        (100e-9, N_SIO2, 20e-6),  # Below cut-off, so only a mode of the box exists.
    ],
)
def test_finite_differences_refuse_a_mode_they_cannot_trust(thickness, n_sub, padding):
    with pytest.raises(ValueError, match="padding"):
        tm0_finite_difference(WAVELENGTH, thickness, N_SI, n_sub, padding=padding)


def test_finite_differences_work_for_a_weak_mode_given_enough_padding():
    analytic = tm0_mode(WAVELENGTH, 100e-9, N_SI).n_eff
    numerical, _, _ = tm0_finite_difference(WAVELENGTH, 100e-9, N_SI, padding=20e-6)
    assert numerical == pytest.approx(analytic, abs=1e-5)


@pytest.mark.parametrize("thickness, n_core, n_sub", ALL_MODES)
def test_thickness_for_neff_inverts_the_solver(thickness, n_core, n_sub):
    n_eff = tm0_mode(WAVELENGTH, thickness, n_core, n_sub).n_eff
    found = tm0_thickness_for_neff(WAVELENGTH, n_eff, n_core, n_sub)
    assert found == pytest.approx(thickness, rel=1e-9, abs=0.0)


@pytest.mark.parametrize("n_eff", [3.6, N_SI, 1.0, 0.9])
def test_thickness_for_neff_rejects_an_index_out_of_range(n_eff):
    with pytest.raises(ValueError):
        tm0_thickness_for_neff(WAVELENGTH, n_eff, N_SI, N_SIO2)


# Further checks: the group index from the stored energy, a second route.


def energy_route_group_index(wavelength, thickness, core_index, sub_index, top_index):
    """Group index c U / P', with U the energy per unit length per unit width.

    U = (1/4) * integral of [eps0 D (|E_x|^2 + |E_z|^2) + mu0 |H_y|^2] dx, where
    D = n^2 - 2 n lambda dn/dlambda is the dispersive energy factor d(omega eps)/d omega.
    """
    n_core, n_sub, n_top = core_index(wavelength), sub_index(wavelength), top_index(wavelength)
    mode = tm0_mode(wavelength, thickness, n_core, n_sub, n_top)
    omega = angular_frequency(wavelength)
    half = thickness / 2.0

    def layer_energy(index, n, start, end):
        slope = group_index(index, wavelength)  # n_g = n - lambda dn/dlambda.
        d_factor = n**2 + 2.0 * n * (slope - n)

        def density(x):
            e_x = mode.k_z * mode.h_y(x) / (omega * sc.epsilon_0 * n**2)
            e_z = mode.e_z_magnitude(x)
            return sc.epsilon_0 * d_factor * (e_x**2 + e_z**2) + sc.mu_0 * mode.h_y(x) ** 2

        return quad(density, start, end, epsabs=0.0, epsrel=1e-10, limit=200)[0] / 4.0

    energy = (
        layer_energy(core_index, n_core, -half, half)
        + layer_energy(top_index, n_top, half, half + 40.0 / mode.kappa_top)
        + layer_energy(sub_index, n_sub, -half - 40.0 / mode.kappa_sub, -half)
    )
    return sc.c * energy / mode.power_per_width()


@pytest.mark.parametrize(
    "thickness, core_index, sub_index",
    [
        (220e-9, silicon_index, air_index),
        (300e-9, silicon_index, air_index),
        (400e-9, silicon_index, air_index),
        (400e-9, silicon_nitride_index, air_index),
        (800e-9, silicon_nitride_index, air_index),
        (220e-9, silicon_index, silica_index),
        (800e-9, silicon_nitride_index, silica_index),
    ],
)
def test_group_index_agrees_with_stored_energy_over_power(thickness, core_index, sub_index):
    """Also tests the prefactor of P': a wrong factor would show up in this ratio directly."""
    from_derivative = tm0_group_index(WAVELENGTH, thickness, core_index, sub_index)
    from_energy = energy_route_group_index(
        WAVELENGTH, thickness, core_index, sub_index, air_index
    )
    assert from_derivative == pytest.approx(from_energy, abs=1e-6)


def test_group_index_peak_of_the_silicon_membrane():
    thickness = np.arange(200e-9, 300e-9, 1e-9)
    n_g = [tm0_group_index(WAVELENGTH, t, silicon_index) for t in thickness]
    assert np.max(n_g) == pytest.approx(5.18, abs=0.01)
    assert thickness[np.argmax(n_g)] == pytest.approx(246e-9, abs=1.5e-9)


@pytest.mark.parametrize("core_index, expected", [(silicon_index, 232.7e-9), (silicon_nitride_index, 448.6e-9)])
def test_tm1_appears_above_half_a_wavelength_over_root_of_n_squared_minus_1(core_index, expected):
    """TM1 has pi added to the guidance condition, and first appears with kappa near 0."""
    n_core = core_index(WAVELENGTH)
    threshold = WAVELENGTH / (2.0 * np.sqrt(n_core**2 - 1.0))
    assert threshold == pytest.approx(expected, abs=0.1e-9)

    def tm1_margin(thickness):
        k_x, _, _, phi_sub, phi_top = _wavenumbers_and_phases(1.000001, WAVELENGTH, n_core, 1.0, 1.0)
        return k_x * thickness - phi_sub - phi_top - np.pi

    assert tm1_margin(1.01 * threshold) > 0.0
    assert tm1_margin(0.99 * threshold) < 0.0


def test_group_index_includes_material_dispersion():
    with_dispersion = tm0_group_index(WAVELENGTH, 220e-9, silicon_index)
    n_fixed = silicon_index(WAVELENGTH)
    without = tm0_group_index(WAVELENGTH, 220e-9, lambda w: n_fixed)
    assert with_dispersion - without > 0.1
