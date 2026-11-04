"""Regenerate every figure of the project as a PDF and a 300-dpi PNG.

Usage: python3 scripts/make_figures.py [--output-dir figures]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from fepc.constants import ev_to_joule, joule_to_ev, photon_energy
from fepc.pinem import (
    G_AT_P0_ZERO,
    average_over_g,
    averaged_spectrum,
    pulse_envelope_distribution,
    sideband_probabilities,
    spectrum,
)

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


FIGURES = {
    "pinem_probabilities": pinem_probabilities_figure,
    "pinem_spectra": pinem_spectra_figure,
    "pinem_averaging": pinem_averaging_figure,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parent.parent / "figures"
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, make_figure in FIGURES.items():
        fig = make_figure()
        fig.savefig(output_dir / f"{name}.pdf")
        fig.savefig(output_dir / f"{name}.png", dpi=300)
        plt.close(fig)


if __name__ == "__main__":
    main()
