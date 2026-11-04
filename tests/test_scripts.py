"""Checks that the figure script and the notebooks run, and that the figures are right."""

import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Draw figures without opening a window.

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.integrate import trapezoid

REPO = Path(__file__).resolve().parent.parent
NOTEBOOKS = sorted((REPO / "notebooks").glob("*.ipynb"))

sys.path.append(str(REPO / "scripts"))
import make_figures


def test_make_figures_regenerates_every_figure(tmp_path):
    script = REPO / "scripts" / "make_figures.py"
    subprocess.run([sys.executable, str(script), "--output-dir", str(tmp_path)], check=True)
    pdf_names = {path.stem for path in tmp_path.glob("*.pdf")}
    png_names = {path.stem for path in tmp_path.glob("*.png")}
    assert pdf_names == set(make_figures.FIGURES)
    assert png_names == pdf_names
    # The set of kept figures must match the set the script makes. Contents are not compared.
    assert {path.stem for path in (REPO / "figures").glob("*.pdf")} == pdf_names
    assert {path.stem for path in (REPO / "figures").glob("*.png")} == pdf_names
    # This figure is 6.0 inches wide, so 300 dpi means 1800 pixels.
    assert plt.imread(tmp_path / "pinem_probabilities.png").shape[1] == 1800


def test_plotted_spectra_have_unit_area_in_ev():
    """Catches a wrong unit on either axis, since the area of a spectrum is dimensionless.

    The tolerance allows for the tiny tails that fall outside the plotted window.
    """
    fig = make_figures.pinem_spectra_figure()
    for ax in fig.axes:
        for line in ax.lines:
            energy_ev, density = line.get_data()
            assert trapezoid(density, energy_ev) == pytest.approx(1.0, abs=1e-3)
    plt.close(fig)


def test_probability_figure_marks_the_zero_of_p0():
    fig = make_figures.pinem_probabilities_figure()
    ax = fig.axes[0]
    g, p0 = ax.lines[0].get_data()
    marked = ax.lines[-1].get_xdata()[0]
    assert np.interp(marked, g, p0) < 1e-4
    plt.close(fig)


def test_there_is_a_notebook_to_run():
    """Without this, an empty folder would silently skip the test below."""
    assert NOTEBOOKS


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_runs_top_to_bottom(notebook, tmp_path):
    # "python -m nbconvert", not "jupyter nbconvert", which can pick another Python's Jupyter.
    command = [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute"]
    subprocess.run([*command, "--output-dir", str(tmp_path), str(notebook)], check=True)
