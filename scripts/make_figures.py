"""Regenerate every figure of the project as a PDF and a 300-dpi PNG.

Usage: python3 scripts/make_figures.py [--output-dir figures]
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import NullFormatter, ScalarFormatter

from fepc.constants import ev_to_joule, joule_to_ev, photon_energy
from fepc.coupling import phase_matched_kinetic_energy
from fepc.materials import (
    air_index,
    group_index,
    silica_index,
    silicon_index,
    silicon_nitride_index,
)
from fepc.pinem import (
    G_AT_P0_ZERO,
    average_over_g,
    averaged_spectrum,
    pulse_envelope_distribution,
    sideband_probabilities,
    spectrum,
)
from fepc.slab import tm0_cutoff_thickness, tm0_group_index, tm0_mode

# Okabe-Ito colours, chosen to stay distinct with colour-blindness. Line styles differ too.
COLOURS = ["#0072B2", "#E69F00", "#009E73", "#D55E00"]
LINE_STYLES = ["-", "--", "-.", ":"]

WAVELENGTH = 1550e-9  # m
HBAR_OMEGA = photon_energy(WAVELENGTH)  # J


def pinem_probabilities_figure() -> plt.Figure:
    """Sideband probabilities against coupling strength, with the first zero of P_0."""
    g = np.linspace(0.0, 4.0, 801)
    fig, ax = plt.subplots(figsize=(6.0, 3.8), layout="constrained")
    for n, colour, style in zip(range(4), COLOURS, LINE_STYLES):
        ax.plot(g, sideband_probabilities(g, n), color=colour, linestyle=style, label=f"$n={n}$")
    ax.axvline(G_AT_P0_ZERO, color="black", linewidth=0.8)
    ax.annotate(
        f"$P_0=0$ at $|g|={G_AT_P0_ZERO:.4f}$",
        xy=(G_AT_P0_ZERO, 0.8),
        xytext=(1.5, 0.8),
        arrowprops={"arrowstyle": "->", "linewidth": 0.8},
    )
    ax.set_xlabel("Coupling strength $|g|$ (dimensionless)")
    ax.set_ylabel("Sideband probability $P_n$ (dimensionless)")
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(title="Sideband order")
    return fig


def _plot_spectrum(ax: plt.Axes, energy: np.ndarray, s: np.ndarray, index: int, label: str) -> None:
    """Plot a spectrum given in joules and 1/J, on axes in eV and 1/eV."""
    ax.plot(
        joule_to_ev(energy),
        s * ev_to_joule(1.0),
        color=COLOURS[index],
        linestyle=LINE_STYLES[index],
        label=label,
    )


def pinem_spectra_figure() -> plt.Figure:
    """Spectra at 1550 nm for three coupling strengths and three zero-loss peak widths."""
    energy = ev_to_joule(np.linspace(-8.0, 8.0, 3201))
    fig, axes = plt.subplots(3, 1, figsize=(6.0, 6.5), sharex=True, layout="constrained")
    for ax, g in zip(axes, [0.5, G_AT_P0_ZERO, 3.0]):
        for index, fwhm_ev in enumerate([0.3, 0.6, 1.0]):
            s = spectrum(energy, g, HBAR_OMEGA, ev_to_joule(fwhm_ev))
            _plot_spectrum(ax, energy, s, index, f"FWHM {fwhm_ev} eV")
        ax.set_title(f"$|g|={g:.5g}$", loc="left")
        ax.set_ylabel("Spectral density (1/eV)")
    axes[0].legend()
    axes[-1].set_xlabel("Energy change $E$ (eV), positive for gain")
    axes[-1].set_xlim(-8.0, 8.0)
    return fig


def pinem_averaging_figure() -> plt.Figure:
    """How a pulse envelope fills in the zero of P_0 and moves weight towards E = 0."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    ratios = [0.0, 0.3, 1.0]
    envelope_width = 1.0  # Only sigma_tau / T matters, so take T = 1.

    g0_values = np.linspace(0.0, 4.0, 401)
    for index, ratio in enumerate(ratios):
        p0 = []
        for g0 in g0_values:
            g_values, weights = pulse_envelope_distribution(g0, envelope_width, ratio)
            p0.append(average_over_g(g_values, weights, np.array([0]))[0])
        left.plot(
            g0_values,
            p0,
            color=COLOURS[index],
            linestyle=LINE_STYLES[index],
            label=rf"$\sigma_\tau/T={ratio:g}$",
        )
    left.set_xlabel("Peak coupling strength $g_0$ (dimensionless)")
    left.set_ylabel(r"Averaged probability $\bar{P}_0$ (dimensionless)")
    left.set_xlim(0.0, 4.0)
    left.set_ylim(0.0, 1.0)
    left.legend()

    g0 = 3.0
    fwhm = ev_to_joule(0.6)
    energy = ev_to_joule(np.linspace(-8.0, 8.0, 3201))
    for index, ratio in enumerate(ratios):
        g_values, weights = pulse_envelope_distribution(g0, envelope_width, ratio)
        s = averaged_spectrum(energy, g_values, weights, HBAR_OMEGA, fwhm)
        _plot_spectrum(right, energy, s, index, rf"$\sigma_\tau/T={ratio:g}$")
    right.set_xlabel("Energy change $E$ (eV), positive for gain")
    right.set_ylabel("Spectral density (1/eV)")
    right.set_xlim(-8.0, 8.0)
    right.set_ylim(0.0, 0.28)
    right.set_title(f"$g_0={g0:g}$, FWHM {joule_to_ev(fwhm):.1f} eV", loc="left")
    right.legend(loc="upper center", ncols=3)
    return fig


# Each platform at 1550 nm: label, core index and substrate index. The top is always air.
PLATFORMS = [
    ("Si, air-clad", silicon_index, air_index),
    ("Si on SiO$_2$", silicon_index, silica_index),
    ("Si$_3$N$_4$, air-clad", silicon_nitride_index, air_index),
    ("Si$_3$N$_4$ on SiO$_2$", silicon_nitride_index, silica_index),
]


def _thicknesses(
    core_index: Callable[[float], float], sub_index: Callable[[float], float]
) -> np.ndarray:
    """Thicknesses (m) from 50 nm, or just above cut-off, up to 3 um."""
    cutoff = tm0_cutoff_thickness(WAVELENGTH, core_index(WAVELENGTH), sub_index(WAVELENGTH))
    return np.geomspace(max(50e-9, 1.01 * cutoff), 3e-6, 300)


def _use_thickness_axis(ax: plt.Axes) -> None:
    """Label a log axis of film thickness with plain numbers in nm."""
    ax.set_xscale("log")
    ax.set_xlim(50.0, 3000.0)
    ax.set_xticks([50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("Film thickness $t$ (nm)")


def materials_dispersion_figure() -> plt.Figure:
    """How the index and group index of each material change across the telecom band."""
    wavelength = np.linspace(1500e-9, 1630e-9, 131)
    materials = [("Si", silicon_index), ("Si$_3$N$_4$", silicon_nitride_index), ("SiO$_2$", silica_index)]
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    for index, (label, material) in enumerate(materials):
        style = {"color": COLOURS[index], "linestyle": LINE_STYLES[index], "label": label}
        left.plot(wavelength * 1e9, material(wavelength) - material(WAVELENGTH), **style)
        n_g = [group_index(material, w) - group_index(material, WAVELENGTH) for w in wavelength]
        right.plot(wavelength * 1e9, n_g, **style)
    left.set_ylabel("Change in $n$ from 1550 nm (dimensionless)")
    right.set_ylabel("Change in $n_g$ from 1550 nm (dimensionless)")
    for ax in (left, right):
        ax.set_xlabel("Wavelength (nm)")
        ax.set_xlim(1500.0, 1630.0)
        ax.legend()
    return fig


def slab_index_figure() -> plt.Figure:
    """Mode index and group index of TM0 against film thickness, at 1550 nm."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8), layout="constrained")
    for index, (label, core, sub) in enumerate(PLATFORMS):
        thickness = _thicknesses(core, sub)
        n_eff = [tm0_mode(WAVELENGTH, t, core(WAVELENGTH), sub(WAVELENGTH)).n_eff for t in thickness]
        n_g = [tm0_group_index(WAVELENGTH, t, core, sub) for t in thickness]
        style = {"color": COLOURS[index], "linestyle": LINE_STYLES[index], "label": label}
        left.plot(thickness * 1e9, n_eff, **style)
        right.plot(thickness * 1e9, n_g, **style)
    left.set_ylabel(r"Mode index $n_{\rm eff}$ (dimensionless)")
    right.set_ylabel("Group index $n_g$ (dimensionless)")
    for ax in (left, right):
        _use_thickness_axis(ax)
        ax.legend()
    return fig


def slab_mode_figure() -> plt.Figure:
    """H_y and |E_z| across an air-clad Si film and an air-clad Si3N4 film, at 1550 nm."""
    cases = [
        ("Si, 220 nm", 220e-9, silicon_index),
        ("Si$_3$N$_4$, 800 nm", 800e-9, silicon_nitride_index),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True, layout="constrained")
    for ax, (label, thickness, core) in zip(axes, cases):
        mode = tm0_mode(WAVELENGTH, thickness, core(WAVELENGTH))
        x = np.linspace(-1.0e-6, 1.0e-6, 4001)
        e_z = mode.e_z_magnitude(x)
        ax.axvspan(-thickness / 2.0 * 1e9, thickness / 2.0 * 1e9, color="0.9", label="Core")
        ax.plot(x * 1e9, mode.h_y(x), color=COLOURS[0], linestyle=LINE_STYLES[0], label="$H_y$")
        ax.plot(
            x * 1e9,
            e_z / np.max(e_z),
            color=COLOURS[1],
            linestyle=LINE_STYLES[1],
            label="$|E_z|$, scaled to its peak",
        )
        ax.set_title(rf"{label}, $n_{{\rm eff}}={mode.n_eff:.3f}$", loc="left")
        ax.set_xlabel("Position across the film $x$ (nm)")
        ax.set_xlim(-1000.0, 1000.0)
    axes[0].set_ylabel("Field, scaled (dimensionless)")
    axes[0].set_ylim(0.0, 1.3)
    axes[0].legend(loc="upper center", ncols=3)
    return fig


def phase_matching_figure() -> plt.Figure:
    """Electron energy that matches the TM0 mode, against film thickness, at 1550 nm."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2), layout="constrained")
    ax.axhspan(60.0, 300.0, color="0.92", label="60 to 300 keV")
    for index, (label, core, sub) in enumerate(PLATFORMS):
        thickness = _thicknesses(core, sub)
        n_eff = np.array(
            [tm0_mode(WAVELENGTH, t, core(WAVELENGTH), sub(WAVELENGTH)).n_eff for t in thickness]
        )
        energy_kev = joule_to_ev(phase_matched_kinetic_energy(n_eff)) / 1e3
        ax.plot(
            thickness * 1e9,
            energy_kev,
            color=COLOURS[index],
            linestyle=LINE_STYLES[index],
            label=label,
        )
    floor_kev = joule_to_ev(phase_matched_kinetic_energy(silicon_nitride_index(WAVELENGTH))) / 1e3
    cap_kev = joule_to_ev(phase_matched_kinetic_energy(silica_index(WAVELENGTH))) / 1e3
    ax.axhline(floor_kev, color="black", linewidth=0.8)
    ax.axhline(cap_kev, color="black", linewidth=0.8)
    ax.annotate(f"Si$_3$N$_4$ floor, {floor_kev:.1f} keV", xy=(53.0, floor_kev + 4.0))
    ax.annotate(f"SiO$_2$ cap, {cap_kev:.1f} keV", xy=(53.0, cap_kev + 4.0))
    _use_thickness_axis(ax)
    ax.set_ylabel("Phase-matched kinetic energy $E_k$ (keV)")
    ax.set_ylim(0.0, 400.0)
    ax.legend(loc="upper right")
    return fig


FIGURES = {
    "pinem_probabilities": pinem_probabilities_figure,
    "pinem_spectra": pinem_spectra_figure,
    "pinem_averaging": pinem_averaging_figure,
    "materials_dispersion": materials_dispersion_figure,
    "slab_index": slab_index_figure,
    "slab_mode": slab_mode_figure,
    "phase_matching": phase_matching_figure,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parent.parent / "figures"
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, make_figure in FIGURES.items():
        fig = make_figure()
        # No creation date in the PDF, so that regenerating an unchanged figure changes nothing.
        fig.savefig(output_dir / f"{name}.pdf", metadata={"CreationDate": None})
        fig.savefig(output_dir / f"{name}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
