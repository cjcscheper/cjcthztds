"""cjcthztds.app, headless: run_analysis wiring with the dialogs and windows replaced, and the figures built.

Skipped when the app extra (matplotlib, cjcplot) is not installed, as in CI.
"""

from pathlib import Path

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
pytest.importorskip("cjcplot")
# Draw in memory; CI machines have no screen. Must run before pyplot is imported; plt.show() then returns at once.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from cjcthztds import TimeTrace, analyze, load_time_trace_map  # noqa: E402
from cjcthztds.app import MapSelection, plot_analysis, run, select_map_positions, select_window_regions  # noqa: E402
from cjcthztds.app.dialogs import _file_types  # noqa: E402

DATA = Path(__file__).resolve().parent / "data"

# Without a screen, plt.show() in the app only warns that it cannot show anything; that is expected here.
pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive:UserWarning")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close the figures each test made."""
    yield
    plt.close("all")


def test_file_dialog_filter_comes_from_the_registry():
    """The dialog filter lists every format with an extension, all of them together first, then all files."""
    file_types = _file_types()
    assert file_types[0] == ("THz trace files", "*.pulse.csv *.tim.csv *.dat")
    assert ("Menlo TDS", "*.dat") in file_types and file_types[-1] == ("All files", "*.*")
    assert _file_types(maps_only=True)[1:] == [("THz mapper", "*.tim.csv"), ("All files", "*.*")]


def test_run_analysis_with_files(monkeypatch):
    """Reference and sample from the dialogs, one window region: run_analysis returns the analysis of those files."""
    monkeypatch.setattr(run, "select_reference_file", lambda: DATA / "menlo_dat.dat")
    monkeypatch.setattr(run, "select_sample_files", lambda: [DATA / "toptica_pulse_csv.pulse.csv"])
    monkeypatch.setattr(run, "select_window_regions", lambda *args, **kwargs: [(-2.0, 3.0)])
    analysis = run.run_analysis(show_plots=False, taper=0.25, padding_ps=5.0)
    assert [trace.path.name for trace in analysis.traces] == ["menlo_dat.dat", "toptica_pulse_csv.pulse.csv"]
    assert analysis.window_regions_ps == ((-2.0, 3.0),)
    # The settings reach analyze unchanged, including the band defaults of run_analysis.
    assert (analysis.taper, analysis.padding_ps) == (0.25, 5.0)
    assert (analysis.freq_min_hz, analysis.freq_max_hz) == (0.1e12, 3.5e12)


def test_run_analysis_without_window_regions(monkeypatch):
    """Drawing no region analyses the whole traces."""
    monkeypatch.setattr(run, "select_reference_file", lambda: DATA / "menlo_dat.dat")
    monkeypatch.setattr(run, "select_sample_files", lambda: [DATA / "astrella"])
    monkeypatch.setattr(run, "select_window_regions", lambda *args, **kwargs: [])
    analysis = run.run_analysis(show_plots=False)
    assert analysis.window_regions_ps is None


def test_run_analysis_with_a_map(monkeypatch):
    """A map file as reference opens the mapper window; the picked positions become the traces, reference first."""
    monkeypatch.setattr(run, "select_reference_file", lambda: DATA / "mapper_tim_csv.tim.csv")
    monkeypatch.setattr(run, "select_map_positions", lambda trace_map: MapSelection((0, 0), ((1, 1), (2, 3))))
    monkeypatch.setattr(run, "select_window_regions", lambda *args, **kwargs: [])
    analysis = run.run_analysis(show_plots=False)
    assert [trace.metadata["position"] for trace in analysis.traces] == [(0, 0), (1, 1), (2, 3)]


def test_run_analysis_cancelled(monkeypatch):
    """A cancelled reference dialog stops with a clear RuntimeError."""
    monkeypatch.setattr(run, "select_reference_file", lambda: None)
    with pytest.raises(RuntimeError, match="no reference file selected"):
        run.run_analysis(show_plots=False)


def test_plot_analysis_draws_the_result_figures(monkeypatch):
    """plot_analysis draws traces, amplitude spectra, Re and Im of the field ratio, and the power ratio."""
    time_ps = -10.0 + 0.02 * np.arange(1000)
    pulse = np.exp(-0.5 * (time_ps / 0.2) ** 2) * -time_ps / 0.2
    traces = [TimeTrace(time_ps, pulse), TimeTrace(time_ps, 0.5 * np.roll(pulse, 50))]
    analysis = analyze(traces, window_regions_ps=[(-3.0, 4.0)], freq_min_hz=0.1e12, freq_max_hz=3e12)
    plot_analysis(analysis)
    assert len(plt.get_fignums()) == 5


def test_interactive_windows_build_headless():
    """The mapper window and the window selector build their figures and return an empty selection without a user."""
    selection = select_map_positions(load_time_trace_map(DATA / "mapper_tim_csv.tim.csv"))
    assert selection == MapSelection(reference_position=None, selected_positions=())
    time_ps = np.linspace(-5, 5, 101)
    assert select_window_regions(time_ps, np.array([np.sin(time_ps), np.cos(time_ps)])) == []


def test_run_analysis_shows_peak_fits(monkeypatch):
    """show_peak_fits=True draws one peak-spline figure per trace, also without the result plots."""
    monkeypatch.setattr(run, "select_reference_file", lambda: DATA / "menlo_dat.dat")
    monkeypatch.setattr(run, "select_sample_files", lambda: [DATA / "toptica_pulse_csv.pulse.csv"])
    monkeypatch.setattr(run, "select_window_regions", lambda *args, **kwargs: [])
    run.run_analysis(show_plots=False, show_peak_fits=True)
    assert len(plt.get_fignums()) == 2
