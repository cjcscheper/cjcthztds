"""Interactive window selection: drag across the time traces to choose the regions to keep."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from cjcplot import plot_1d
from matplotlib.widgets import SpanSelector


def select_window_regions(
    time_ps: np.ndarray,
    signals: np.ndarray,
    *,
    legend_labels: Sequence[str] | None = None,
    reference_index: int = 0,
    show_magnitude_difference: bool = True,
    show_legend: bool = True,
) -> list[tuple[float, float]]:
    """Show the traces and let the user drag out one or more time regions; return them when the window closes.

    Left-drag to add a region (it is shaded grey); add as many as needed and
    close the figure when done.

    Parameters
    ----------
    time_ps : ndarray, shape (N,)
        Time axis in ps.
    signals : ndarray, shape (M, N)
        The traces, one per row (NaN where a trace has no data is fine).
    legend_labels : sequence of str, optional
        One label per trace.
    reference_index : int
        Row of the reference, for the magnitude differences.
    show_magnitude_difference : bool
        Also draw ``|reference| - |trace|`` (dashed) for every other trace, which
        shows where the traces differ.
    show_legend : bool
        Show the legend.

    Returns
    -------
    list of (start_ps, stop_ps)
        The selected regions, in the order they were drawn; empty if none.
    """
    # The traces themselves, as rows.
    signals = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    if signals.shape[-1] != np.size(time_ps):
        raise ValueError(f"signals must have shape (M, {np.size(time_ps)}), got {signals.shape}")
    y_data = list(signals)
    labels = list(legend_labels) if legend_labels is not None else [f"trace {i + 1}" for i in range(len(y_data))]
    line_styles = ["-"] * len(y_data)

    # Optional dashed |ref| - |trace| curves, which highlight where the traces differ.
    if show_magnitude_difference and signals.shape[0] > 1:
        for index, trace in enumerate(signals):
            if index != reference_index:
                y_data.append(np.abs(signals[reference_index]) - np.abs(trace))
                labels.append(f"|ref| - |sample {index}|")
                line_styles.append("--")

    # The plot to drag on.
    fig, ax = plot_1d(
        x_data=[time_ps] * len(y_data),
        y_data=y_data,
        legend_labels=labels,
        line_styles=line_styles,
        xlabel="Time [ps]",
        ylabel="Amplitude [arb. u]",
        fig_show=False,
        first_on_top=False,
        legend_show=show_legend,
    )

    # Each drag adds a region and shades it, so the user sees what was taken.
    selections: list[tuple[float, float]] = []

    def on_select(xmin: float, xmax: float) -> None:
        # A click without a drag spans nothing; ignore it.
        if np.isclose(xmin, xmax):
            return
        start_ps, stop_ps = sorted((float(xmin), float(xmax)))
        selections.append((start_ps, stop_ps))
        ax.axvspan(start_ps, stop_ps, color="gray", alpha=0.3)
        fig.canvas.draw_idle()
        print(f"Added window: {start_ps:.3f} to {stop_ps:.3f} ps")

    # Keep a reference to the selector on the figure: without one it is garbage-collected and stops responding.
    fig.span_selector = SpanSelector(
        ax, on_select, "horizontal", useblit=False, button=1, interactive=True,
        props={"alpha": 0.3, "facecolor": "tab:gray"},
    )
    plt.show(block=True)
    return selections
