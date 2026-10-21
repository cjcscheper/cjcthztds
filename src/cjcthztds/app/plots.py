"""Result figures of an analysis (traces, amplitude spectra, field ratio, power ratio) and the peak-spline check."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from cjcplot import plot_1d, style_colors

from cjcthztds import Analysis, frequency_range_mask, find_peak, peak_spline_curve


def plot_analysis(
    analysis: Analysis,
    *,
    legend_labels: Sequence[str] | None = None,
    show_legends: bool = True,
    field_ratio_separate_figures: bool = False,
) -> None:
    """Plot an analysis: windowed traces, amplitude spectra, field ratio, power ratio.

    The frequency axes are limited to ``analysis.freq_min_hz`` to
    ``analysis.freq_max_hz`` (the full range where those are None).

    Parameters
    ----------
    analysis : Analysis
        The result of ``analyze``.
    legend_labels : sequence of str, optional
        One label per trace; default "ref" and "sample 1", "sample 2", ...
    show_legends : bool
        Show the legends.
    field_ratio_separate_figures : bool
        False (default): one figure with Re of all field ratios and one with Im.
        True: one figure per sample with its Re and Im.
    """
    # Reference first in black, then the style colours; labels default to ref, sample 1, ...
    n_traces = analysis.signal.shape[0]
    reference = analysis.reference_index
    samples = [index for index in range(n_traces) if index != reference]
    if legend_labels is None:
        legend_labels = [
            "ref" if index == reference else f"sample {samples.index(index) + 1}" for index in range(n_traces)
        ]
    colors = ["black", *style_colors()]
    trace_colors = [colors[0] if index == reference else colors[1 + samples.index(index) % (len(colors) - 1)]
                    for index in range(n_traces)]
    sample_colors = [trace_colors[index] for index in samples]
    freq_thz = analysis.freq_hz * 1e-12
    freq_min_hz = analysis.freq_min_hz if analysis.freq_min_hz is not None else 0.0
    freq_max_hz = analysis.freq_max_hz if analysis.freq_max_hz is not None else float(analysis.freq_hz[-1])
    xlim_thz = (freq_min_hz * 1e-12, freq_max_hz * 1e-12)
    in_band = frequency_range_mask(analysis.freq_hz, freq_min_hz, freq_max_hz)

    # 1. The windowed, padded traces that were transformed, with the reference window scaled to the reference peak.
    x_data = [analysis.padded_time_ps] * n_traces
    y_data = list(analysis.padded_signal)
    labels = list(legend_labels)
    line_styles = ["-"] * n_traces
    line_colors = list(trace_colors)
    if analysis.window_regions_ps is not None:
        x_data.append(analysis.time_ps)
        y_data.append(analysis.window[reference] * 1.25 * np.max(np.abs(analysis.signal[reference])))
        labels.append("Tukey window (scaled)")
        line_styles.append("--")
        line_colors.append("grey")
    plot_1d(
        x_data, y_data, colors=line_colors, line_styles=line_styles,
        legend_labels=labels, legend_show=show_legends,
        xlabel="Time [ps]", ylabel="Amplitude [a.u.]", first_on_top=False,
    )

    # 2. Amplitude spectra on a log scale: bandwidth and noise floor.
    amplitude = np.abs(analysis.spectrum)
    band_values = amplitude[:, in_band]
    band_values = band_values[band_values > 0]
    plot_1d(
        [freq_thz] * n_traces, list(amplitude), colors=trace_colors,
        legend_labels=list(legend_labels), legend_show=show_legends,
        xlabel="Frequency [THz]", ylabel="|E(f)| [a.u.]", logy=True,
        xlim=xlim_thz, ylim=(0.9 * band_values.min(), 1.1 * band_values.max()) if band_values.size else None,
        first_on_top=False,
    )

    # 3. Complex field ratio of every sample: Re and Im.
    sample_labels = [legend_labels[index] for index in samples]
    if field_ratio_separate_figures:
        for index, label in zip(samples, sample_labels):
            plot_1d(
                [freq_thz, freq_thz], [analysis.field_ratio[index].real, analysis.field_ratio[index].imag],
                legend_labels=["Re", "Im"], legend_show=show_legends, legend_ncols=2,
                xlabel="Frequency [THz]", ylabel=f"field ratio {label} [-]", xlim=xlim_thz, ylim=(-1.0, 1.2),
            )
    else:
        for part, name in ((np.real, "Re"), (np.imag, "Im")):
            plot_1d(
                [freq_thz] * len(samples), [part(analysis.field_ratio[index]) for index in samples],
                colors=sample_colors, legend_labels=sample_labels, legend_show=show_legends,
                xlabel="Frequency [THz]", ylabel=f"{name}(field ratio) [-]", xlim=xlim_thz, ylim=(-1.0, 1.2),
            )

    # 4. Power ratio |field ratio|² of every sample, y range from the band.
    power_band = analysis.power_ratio[samples][:, in_band]
    power_band = power_band[np.isfinite(power_band) & (power_band > 0)]
    plot_1d(
        [freq_thz] * len(samples), [analysis.power_ratio[index] for index in samples],
        colors=sample_colors, legend_labels=sample_labels, legend_show=show_legends,
        xlabel="Frequency [THz]", ylabel="power ratio [-]", xlim=xlim_thz,
        ylim=(0.9 * power_band.min(), 1.1 * power_band.max()) if power_band.size else None,
    )


def plot_peak_fit(
    time_ps: np.ndarray, signal: np.ndarray, *, spline_half_width: int = 5, title: str = "",
) -> None:
    """Show how :func:`find_peak` found the peak of one trace, to check it by eye.

    The samples the spline goes through (blue), the spline between them (dotted
    yellow) and the peak (red).

    Parameters
    ----------
    time_ps, signal : ndarray, shape (N,)
        The trace.
    spline_half_width : int
        Passed to :func:`find_peak`.
    title : str
        Which trace this is; the title adds the peak time and height.
    """
    # The peak, the spline through the samples around it, and those samples.
    peak = find_peak(time_ps, signal, spline_half_width=spline_half_width)
    curve_time_ps, curve_signal = peak_spline_curve(time_ps, signal, spline_half_width=spline_half_width)
    used = (time_ps >= curve_time_ps[0]) & (time_ps <= curve_time_ps[-1])
    colors = style_colors()

    # The title names the trace and gives the peak.
    peak_text = f"{peak.time_ps:.4f} ps, {peak.amplitude:.4g}"
    plot_1d(
        [time_ps[used], curve_time_ps, [peak.time_ps]],
        [signal[used], curve_signal, [peak.amplitude]],
        colors=[colors[0], colors[1], colors[2]],
        scatter_show=[True, False, True], line_show=[False, True, False],
        scatter_colors=["white", "white", colors[2]], line_styles=["-", ":", "-"],
        legend_labels=["samples", "spline", "peak"], legend_loc="lower center",
        xlabel="Time [ps]", ylabel="Signal [arb. u]",
        title=f"{title}: {peak_text}" if title else peak_text,
    )
