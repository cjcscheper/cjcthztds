"""The interactive THz-TDS analysis from start to end.

Run it with the Run button in PyCharm (this file), with
``python -m cjcthztds.app.run``, or with the console command
``cjcthztds-analyze``. From a script, ``analysis = run_analysis(show_plots=False)``
returns the result for further processing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cjcthztds import (
    Analysis,
    TimeTrace,
    analyze,
    detect_format,
    load_time_trace,
    load_time_trace_map,
    trace_at_position,
)
from cjcthztds.app.dialogs import select_map_file, select_reference_file, select_sample_files
from cjcthztds.app.mapper_window import select_map_positions
from cjcthztds.app.plots import plot_analysis, plot_peak_fit
from cjcthztds.app.window_selector import select_window_regions


def _traces_from_maps(reference_path: Path) -> tuple[list[TimeTrace], list[str]]:
    """Reference and sample traces picked in the mapper window(s), with their legend labels.

    Right-click the reference position and left-click the measurement
    positions. Without measurement positions, a second map file is asked for
    and the measurement positions are picked there (reference and sample
    measured in separate scans).
    """
    # Reference (and usually the samples too) from the first map.
    print("Map file selected. Right-click one point as reference, left-click to select measurements, then close.")
    print("  (Leave measurements unselected to load them from a second map file.)")
    reference_map = load_time_trace_map(reference_path)
    selection = select_map_positions(reference_map)
    if selection.reference_position is None:
        raise RuntimeError("no reference point selected in the mapper window")
    sample_map, positions = reference_map, selection.selected_positions

    # No measurement positions: take them from a second map.
    if not positions:
        print("No measurement positions selected. Select the sample map file...")
        sample_path = select_map_file()
        if sample_path is None:
            raise RuntimeError("no sample map file selected")
        print("Select measurement positions in the sample mapper window, then close.")
        sample_map = load_time_trace_map(sample_path)
        positions = select_map_positions(sample_map).selected_positions
        if not positions:
            raise RuntimeError("no measurement points selected in the sample mapper window")

    traces = [trace_at_position(reference_map, selection.reference_position)]
    traces += [trace_at_position(sample_map, position) for position in positions]
    return traces, [f"position {number}" for number in range(1, len(positions) + 1)]


def run_analysis(
    *,
    taper: float = 0.5,
    center_on_peak: bool = True,
    padding_ps: float = 0.0,
    spline_half_width: int = 5,
    freq_min_hz: float | None = 0.1e12,
    freq_max_hz: float | None = 3.5e12,
    show_plots: bool = True,
    show_legends: bool = True,
    show_window_selector_trace_difference: bool = False,
    field_ratio_separate_figures: bool = False,
    show_peak_fits: bool = False,
) -> Analysis:
    """Start the interactive THz-TDS analysis: pick files, select windows, see the results.

    1. A dialog asks for the reference file. For a map file (raster scan) the
       mapper window opens to pick the reference and measurement positions;
       otherwise a second dialog asks for the sample files.
    2. The window selector shows all traces, each shifted onto the reference
       peak (display only), so one region drawn around the pulse fits all of
       them. Drag one or more regions and close the window; draw none to use
       the whole traces.
    3. :func:`cjcthztds.analyze` computes everything, with the reference first.
    4. The result figures are shown (unless ``show_plots=False``), one after
       another: close one to see the next.

    The settings are passed to ``analyze`` under the same names; see there.

    Parameters
    ----------
    taper, center_on_peak, padding_ps, spline_half_width, freq_min_hz, freq_max_hz :
        As in :func:`cjcthztds.analyze`. Here the band defaults to 0.1–3.5 THz,
        which only limits the plots.
    show_plots : bool
        Show the result figures (default True). False for a script that only
        wants the numbers, e.g. a fit.
    show_legends : bool
        Show legends in the window selector and the result figures.
    show_window_selector_trace_difference : bool
        Also draw ``|reference| - |trace|`` in the window selector.
    field_ratio_separate_figures : bool
        One field-ratio figure per sample (Re and Im) instead of one for all Re
        and one for all Im.
    show_peak_fits : bool
        Also show, per trace, the samples around its peak, the spline through
        them and the peak found (default False):
        a check that the centring found the right peak.

    Returns
    -------
    Analysis
        Every stage of the analysis, reference at index 0.
    """
    # 1. Files: the mapper window for a map, otherwise a second dialog for the samples.
    print("Select reference file...")
    reference_path = select_reference_file()
    if reference_path is None:
        raise RuntimeError("no reference file selected")
    if detect_format(reference_path).IS_MAP:
        traces, sample_labels = _traces_from_maps(reference_path)
    else:
        print("Select sample files...")
        sample_paths = select_sample_files()
        if not sample_paths:
            raise RuntimeError("no sample files selected")
        traces = [load_time_trace(path) for path in [reference_path, *sample_paths]]
        sample_labels = [f"sample {number}" for number in range(1, len(sample_paths) + 1)]
    legend_labels = ["ref", *sample_labels]

    # 2. Window selector. A first analysis without windows gives the centred axis and every peak time;
    #    each trace is drawn shifted onto the reference peak by whole samples (display only), on an axis
    #    long enough that no trace is cut, NaN where a trace has no data.
    centred = analyze(traces, spline_half_width=spline_half_width)
    time_ps, signal = centred.time_ps, centred.signal
    dt_ps = float(time_ps[1] - time_ps[0])
    shifts = [int(round(peak_time_ps / dt_ps)) for peak_time_ps in centred.peak_times_ps]
    first_index, last_index = -max(max(shifts), 0), time_ps.size - 1 - min(min(shifts), 0)
    display_time_ps = time_ps[0] + dt_ps * np.arange(first_index, last_index + 1)
    display_signal = np.full((signal.shape[0], display_time_ps.size), np.nan)
    for row, shift in enumerate(shifts):
        start = -shift - first_index
        display_signal[row, start:start + time_ps.size] = signal[row]
    print("Drag across the plot to define pulse windows. Close the plot when done.")
    window_regions_ps = select_window_regions(
        display_time_ps, display_signal,
        legend_labels=legend_labels, show_magnitude_difference=show_window_selector_trace_difference,
        show_legend=show_legends,
    )
    if not window_regions_ps:
        print("No window regions selected: processing the whole traces without a window.")

    # 3. The analysis itself, with the settings passed on unchanged.
    analysis = analyze(
        traces,
        window_regions_ps=window_regions_ps or None,
        taper=taper,
        center_on_peak=center_on_peak,
        padding_ps=padding_ps,
        spline_half_width=spline_half_width,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
    )

    # A short summary in the terminal: what was loaded and where the pulses are.
    print(f"\nLoaded {len(traces)} traces.")
    for label, trace in zip(legend_labels, traces):
        position = f"  position {trace.metadata['position']}" if "position" in trace.metadata else ""
        print(f"  {label}: {trace.path}{position}")
    print(f"\nReference pulse peak time = {analysis.reference_peak_time_ps:.4f} ps")
    print(f"Peak times relative to the reference = {np.round(analysis.peak_times_ps, 4)} ps")

    # 4. The figures: the peak checks first (if asked for), then the results.
    if show_peak_fits:
        for label, trace_signal in zip(legend_labels, analysis.signal):
            plot_peak_fit(analysis.time_ps, trace_signal, spline_half_width=spline_half_width, title=label)
    if show_plots:
        plot_analysis(
            analysis, legend_labels=legend_labels, show_legends=show_legends,
            field_ratio_separate_figures=field_ratio_separate_figures,
        )
    return analysis


def main() -> None:
    """Console command ``cjcthztds-analyze``: run the analysis and return nothing.

    A console command exits with its function's return value; returning the
    Analysis from here would make the command end with an error status.
    """
    run_analysis()


if __name__ == "__main__":
    run_analysis()
