"""Validation checks for the Sellmeier models (docs/theory.md, section 2.1)."""

import numpy as np
import pytest

from fepc.materials import (
    SILICA_TERMS,
    SILICON_NITRIDE_TERMS,
    SILICON_TERMS,
    group_index,
    sellmeier_index,
    silica_index,
    silicon_index,
    silicon_nitride_index,
)

WAVELENGTH = 1.55e-6  # m


def analytic_group_index(wavelength: float, terms: tuple[tuple[float, float], ...]) -> float:
    """Group index from the exact derivative of the Sellmeier formula."""
    l = wavelength * 1e6
    d_n_squared = sum(
        -2.0 * strength * l * resonance**2 / (l**2 - resonance**2) ** 2
        for strength, resonance in terms
    )
    n = sellmeier_index(wavelength, terms)
    return n - l * d_n_squared / (2.0 * n)


# Checks listed in section B of the project plan.


def test_silicon_index_and_group_index_at_1550_nm():
    assert round(silicon_index(WAVELENGTH), 3) == 3.478
    # The plan says about 3.60. This formula gives 3.6053, which is within 0.01 of it.
    n_g = group_index(silicon_index, WAVELENGTH)
    assert abs(n_g - 3.60) < 0.01
    assert n_g == pytest.approx(3.6053, abs=1e-4)


def test_silicon_nitride_index_and_group_index_at_1550_nm():
    assert round(silicon_nitride_index(WAVELENGTH), 3) == 1.996
    assert round(group_index(silicon_nitride_index, WAVELENGTH), 2) == 2.04


def test_silica_index_at_1550_nm():
    assert round(silica_index(WAVELENGTH), 3) == 1.444


def test_coefficients_are_pinned():
    """Compared with a typed copy, so that a slip of one digit in any coefficient is noticed."""
    assert SILICON_TERMS == (
        (10.6684293, 0.301516485),
        (0.0030434748, 1.13475115),
        (1.54133408, 1104.0),
    )
    assert SILICON_NITRIDE_TERMS == ((3.0249, 0.1353406), (40314.0, 1239.842))
    assert SILICA_TERMS == (
        (0.6961663, 0.0684043),
        (0.4079426, 0.1162414),
        (0.8974794, 9.896161),
    )


# Further checks.


@pytest.mark.parametrize(
    "index, n, n_g",
    [
        (silicon_index, 3.4777, 3.6053),
        (silicon_nitride_index, 1.9963, 2.0396),
        (silica_index, 1.4440, 1.4626),
    ],
)
def test_table_in_the_theory(index, n, n_g):
    assert round(index(WAVELENGTH), 4) == n
    assert round(group_index(index, WAVELENGTH), 4) == n_g


@pytest.mark.parametrize("terms", [SILICON_TERMS, SILICON_NITRIDE_TERMS, SILICA_TERMS])
def test_central_difference_matches_the_exact_derivative(terms):
    numerical = group_index(lambda w: sellmeier_index(w, terms), WAVELENGTH)
    assert numerical == pytest.approx(analytic_group_index(WAVELENGTH, terms), abs=1e-8)


@pytest.mark.parametrize("index", [silicon_index, silicon_nitride_index, silica_index])
def test_index_falls_with_wavelength_across_the_band(index):
    n = index(np.linspace(1.50e-6, 1.63e-6, 27))
    assert np.all(np.diff(n) < 0.0)


@pytest.mark.parametrize("wavelength", [1500e-9, 1630e-9])
def test_group_index_of_a_linear_index_is_its_intercept(wavelength):
    """For n = a + b lambda, n - lambda dn/dlambda is a, at any wavelength."""
    assert group_index(lambda w: 3.476 + 1e5 * w, wavelength) == pytest.approx(3.476, abs=1e-9)


def test_central_difference_error_falls_with_the_square_of_the_step():
    exact = analytic_group_index(WAVELENGTH, SILICON_TERMS)
    coarse = group_index(silicon_index, WAVELENGTH, step=20e-9) - exact
    fine = group_index(silicon_index, WAVELENGTH, step=10e-9) - exact
    assert coarse / fine == pytest.approx(4.0, abs=0.1)
