"""TM0 modes of a three-layer slab waveguide: mode index, fields, power and group index.

The core has thickness t, with a substrate below and a top cladding above. The theory is
in docs/theory.md, section 2.2. Everything is in SI units, and x is measured from the
centre of the core, with the top cladding at x > t/2. The fields use the peak-amplitude
convention of section 1.1, and H_y is normalized to 1 A/m at the centre of the core.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy import constants as sc
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from fepc.constants import FloatOrArray, angular_frequency
from fepc.materials import air_index, group_index


def _wavenumbers_and_phases(
    n_eff: float, wavelength: float, n_core: float, n_sub: float, n_top: float
) -> tuple[float, float, float, float, float]:
    """Return k_x, kappa_sub, kappa_top (rad/m) and the phases phi_sub, phi_top (rad)."""
    k0 = 2.0 * np.pi / wavelength
    k_x = k0 * np.sqrt(n_core**2 - n_eff**2)
    kappa_sub = k0 * np.sqrt(n_eff**2 - n_sub**2)
    kappa_top = k0 * np.sqrt(n_eff**2 - n_top**2)
    phi_sub = np.arctan2(n_core**2 / n_sub**2 * kappa_sub, k_x)
    phi_top = np.arctan2(n_core**2 / n_top**2 * kappa_top, k_x)
    return k_x, kappa_sub, kappa_top, phi_sub, phi_top


@dataclass(frozen=True)
class Tm0Mode:
    """A guided TM0 mode. All lengths are in m, and wavenumbers in rad/m."""

    wavelength: float
    thickness: float
    n_core: float
    n_sub: float
    n_top: float
    n_eff: float
    k_x: float
    kappa_sub: float
    kappa_top: float
    phi_sub: float
    phi_top: float

    @property
    def omega(self) -> float:
        """Angular frequency (rad/s)."""
        return angular_frequency(self.wavelength)

    @property
    def k_z(self) -> float:
        """Propagation constant (rad/m)."""
        return self.n_eff * 2.0 * np.pi / self.wavelength

    @property
    def _scale(self) -> float:
        """Amplitude of the cosine in the core that makes H_y = 1 at the core centre."""
        return 1.0 / np.cos((self.phi_top - self.phi_sub) / 2.0)

    def h_y(self, x: FloatOrArray) -> FloatOrArray:
        """Magnetic field H_y (A/m) at position x (m)."""
        half = self.thickness / 2.0
        core = np.cos(self.k_x * (x + half) - self.phi_sub)
        # The maximum and minimum stop these exponentials overflowing in the wrong layer.
        top = np.cos(self.phi_top) * np.exp(-self.kappa_top * np.maximum(x - half, 0.0))
        bottom = np.cos(self.phi_sub) * np.exp(self.kappa_sub * np.minimum(x + half, 0.0))
        return self._scale * np.where(x > half, top, np.where(x < -half, bottom, core))

    def e_z_magnitude(self, x: FloatOrArray) -> FloatOrArray:
        """|E_z| (V/m) at position x (m), from E_z = i dH_y/dx / (omega eps0 eps_r)."""
        half = self.thickness / 2.0
        h_y = np.abs(self.h_y(x))
        core_slope = self._scale * self.k_x * np.abs(np.sin(self.k_x * (x + half) - self.phi_sub))
        core = core_slope / self.n_core**2
        top = self.kappa_top * h_y / self.n_top**2
        bottom = self.kappa_sub * h_y / self.n_sub**2
        slope_over_eps = np.where(x > half, top, np.where(x < -half, bottom, core))
        return slope_over_eps / (self.omega * sc.epsilon_0)

    def e_z_above(self, d: FloatOrArray) -> FloatOrArray:
        """|E_z| (V/m) at height d (m) above the top surface, where the electron flies."""
        return self.e_z_magnitude(self.thickness / 2.0 + d)

    def power_per_width(self) -> float:
        """Cycle-averaged power per unit width P' (W/m), for H_y = 1 A/m at the core centre."""
        core = self.thickness / 2.0 + (np.sin(2.0 * self.phi_sub) + np.sin(2.0 * self.phi_top)) / (
            4.0 * self.k_x
        )
        bottom = np.cos(self.phi_sub) ** 2 / (2.0 * self.kappa_sub)
        top = np.cos(self.phi_top) ** 2 / (2.0 * self.kappa_top)
        integral = core / self.n_core**2 + bottom / self.n_sub**2 + top / self.n_top**2
        return self.k_z / (2.0 * self.omega * sc.epsilon_0) * self._scale**2 * integral


def tm0_mode(
    wavelength: float, thickness: float, n_core: float, n_sub: float = 1.0, n_top: float = 1.0
) -> Tm0Mode:
    """Find the TM0 mode of a slab. With the default n_sub = n_top = 1 it is air-clad.

    We solve the guidance condition k_x t = phi_sub + phi_top with Brent's method. This
    arctangent form has no poles, unlike tan(k_x t / 2). Its left side minus its right
    side falls steadily with n_eff and changes sign exactly once between the highest
    cladding index and the core index. Brent's method needs only that sign change, and
    it is guaranteed to converge.

    Args:
        wavelength: Vacuum wavelength (m).
        thickness: Core thickness (m).
        n_core, n_sub, n_top: Refractive indices (dimensionless).

    Raises:
        ValueError: If the slab has no guided TM0 mode (below cut-off).
    """

    def mismatch(n_eff: float) -> float:
        k_x, _, _, phi_sub, phi_top = _wavenumbers_and_phases(
            n_eff, wavelength, n_core, n_sub, n_top
        )
        return k_x * thickness - phi_sub - phi_top

    n_low = max(n_sub, n_top)
    if n_core <= n_low or mismatch(n_low) <= 0.0:
        raise ValueError("no guided TM0 mode: the core is below cut-off")
    n_eff = brentq(mismatch, n_low, n_core)
    k_x, kappa_sub, kappa_top, phi_sub, phi_top = _wavenumbers_and_phases(
        n_eff, wavelength, n_core, n_sub, n_top
    )
    return Tm0Mode(
        wavelength, thickness, n_core, n_sub, n_top, n_eff, k_x, kappa_sub, kappa_top, phi_sub, phi_top
    )


def tm0_thickness_for_neff(
    wavelength: float, n_eff: float, n_core: float, n_sub: float = 1.0, n_top: float = 1.0
) -> float:
    """Core thickness (m) whose TM0 mode has index `n_eff`, at fixed indices.

    The guidance condition gives this directly, with no root finding.

    Raises:
        ValueError: If no TM0 mode can have this index.
    """
    if not max(n_sub, n_top) <= n_eff < n_core:
        raise ValueError("n_eff must lie between the highest cladding index and the core index")
    k_x, _, _, phi_sub, phi_top = _wavenumbers_and_phases(n_eff, wavelength, n_core, n_sub, n_top)
    return (phi_sub + phi_top) / k_x


def tm0_cutoff_thickness(
    wavelength: float, n_core: float, n_sub: float = 1.0, n_top: float = 1.0
) -> float:
    """Thinnest core (m) that guides TM0. It is zero for a symmetric slab."""
    return tm0_thickness_for_neff(wavelength, max(n_sub, n_top), n_core, n_sub, n_top)


def tm0_group_index(
    wavelength: float,
    thickness: float,
    core_index: Callable[[float], float],
    sub_index: Callable[[float], float] = air_index,
    top_index: Callable[[float], float] = air_index,
) -> float:
    """Group index of the TM0 mode (dimensionless) at fixed thickness, with material dispersion.

    The index functions give each layer's index at a vacuum wavelength (m), so that the
    change of the materials with wavelength is included along with that of the mode.

    Raises:
        ValueError: If the mode is cut off at wavelength + 0.1 nm, which happens within
            about 0.02 nm of the cut-off thickness.
    """

    def n_eff(w: float) -> float:
        return tm0_mode(w, thickness, core_index(w), sub_index(w), top_index(w)).n_eff

    return group_index(n_eff, wavelength)


def tm0_finite_difference(
    wavelength: float,
    thickness: float,
    n_core: float,
    n_sub: float = 1.0,
    n_top: float = 1.0,
    step: float = 1e-9,
    padding: float = 3e-6,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Independent check of the mode: finite differences, with no guidance condition.

    We discretize d/dx (1/eps dH/dx) + k0^2 H = k_z^2 H / eps on a uniform grid, which
    turns it into an eigenvalue problem. The largest eigenvalue is n_eff^2 of TM0. The
    grid puts both faces of the core halfway between grid points. There 1/eps is replaced
    by 2/(eps_left + eps_right), which keeps dH/dx / eps continuous, so the error falls
    as step^2.

    Args:
        wavelength: Vacuum wavelength (m).
        thickness: Core thickness (m).
        n_core, n_sub, n_top: Refractive indices (dimensionless).
        step: Requested grid step (m). It is adjusted to fit the core exactly.
        padding: Width of cladding kept on each side (m).

    Returns:
        n_eff, the grid positions x (m) from the core centre, and H_y (A/m), normalized
        to 1 at the core centre.

    Raises:
        ValueError: If there is no guided mode, or the padding holds under 5 decay lengths.
    """
    # An odd number of cells in the core puts one grid point exactly at its centre.
    n_cells = 2 * round(thickness / (2.0 * step)) + 1
    h = thickness / n_cells
    n_padding = int(np.ceil(padding / h))
    j = np.arange(-n_padding, n_cells + n_padding)
    x = (j + 0.5) * h - thickness / 2.0
    eps = np.where(x > thickness / 2.0, n_top**2, np.where(x < -thickness / 2.0, n_sub**2, n_core**2))

    # Dividing by k0^2 gives A H = n_eff^2 B H, with these coefficients. The change of
    # variable y = sqrt(b) H makes the matrix symmetric, which needs a tridiagonal solver only.
    a = 2.0 / (eps[:-1] + eps[1:])  # Stands for 1/eps between neighbouring points.
    b = 1.0 / eps  # 1/eps at each point.
    s = (2.0 * np.pi * h / wavelength) ** 2
    # H_y = 0 one step beyond each end of the grid: copying the end values gives those
    # points a second neighbour, whose value then does not appear in the matrix.
    a_left = np.concatenate(([a[0]], a))
    a_right = np.concatenate((a, [a[-1]]))
    diagonal = (1.0 - (a_left + a_right) / s) / b
    off_diagonal = a / (s * np.sqrt(b[:-1] * b[1:]))
    last = len(x) - 1
    eigenvalue, y = eigh_tridiagonal(diagonal, off_diagonal, select="i", select_range=(last, last))
    n_eff = float(np.sqrt(eigenvalue[0]))
    # A mode of the box itself has n_eff below the cladding, making kappa_squared negative.
    kappa_squared = (2.0 * np.pi / wavelength) ** 2 * (n_eff**2 - max(n_sub, n_top) ** 2)
    if kappa_squared * padding**2 < 25.0:
        raise ValueError("no guided mode, or the padding holds under 5 decay lengths")
    h_y = y[:, 0] * np.sqrt(eps)
    return n_eff, x, h_y / h_y[n_padding + (n_cells - 1) // 2]
