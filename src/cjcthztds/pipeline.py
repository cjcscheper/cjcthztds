"""The processing chain from measured traces to field ratios, keeping every intermediate stage.

``analyze`` runs, in order (THEORY.md explains why each step exists):

1. put all traces on one uniform time axis (``common_time_axis``);
2. centre that axis on the reference peak (``center_on_reference_peak``);
3. multiply each trace by its window (``trace_window``), if regions are given;
4. zero-pad (``zero_pad``);
5. Fourier transform (``fourier_transform``);
6. divide by the reference spectrum (``compute_complex_field_ratio``).

It only computes: no files, no windows, no plots. The interactive version,
which asks for files and windows first, is ``cjcthztds.app.run_analysis``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from cjcthztds.padding import zero_pad
from cjcthztds.peaks import center_on_reference_peak
from cjcthztds.spectra import compute_complex_field_ratio, fourier_transform
from cjcthztds.traces import TimeTrace
from cjcthztds.windows import trace_window


@dataclass(frozen=True)
class Analysis:
    """Every stage of a THz-TDS analysis, from the loaded traces to the field ratios.

    ``M`` is the number of traces, ``N`` the samples on the common time axis,
    ``N_pad`` after zero padding and ``K = N_pad // 2 + 1`` the frequencies.
    Row ``reference_index`` of every array is the reference.

    Attributes
    ----------
    traces : tuple of TimeTrace
        The traces as loaded (own time axes, paths, metadata).
    reference_index : int
        Which trace is the reference.
    time_ps : ndarray, shape (N,)
        The common, uniform time axis, centred so the reference peaks at 0.
    signal : ndarray, shape (M, N)
        The traces on ``time_ps``, before windowing and padding.
    reference_peak_time_ps : float
        The reference peak on the original time axis: the shift applied to it.
    peak_times_ps : ndarray, shape (M,)
        Peak time of every trace on ``time_ps``: the delay relative to the reference.
    window : ndarray, shape (M, N)
        The window applied to each trace (all ones without window regions).
    windowed_signal : ndarray, shape (M, N)
        ``signal * window``.
    padded_time_ps : ndarray, shape (N_pad,)
        ``time_ps`` extended by the zero padding.
    padded_signal : ndarray, shape (M, N_pad)
        The windowed traces with the zeros added: what was Fourier transformed.
    freq_hz : ndarray, shape (K,)
        Frequencies in Hz, from 0 to the Nyquist frequency.
    spectrum : ndarray, complex, shape (M, K)
        Fourier transform of each padded trace (see ``fourier_transform``).
    field_ratio : ndarray, complex, shape (M, K)
        ``spectrum / spectrum[reference_index]``: the complex transmission or
        reflection coefficient of each trace relative to the reference.
    power_ratio : ndarray, shape (M, K)
        ``|field_ratio|²``.
    window_regions_ps : tuple of (start_ps, stop_ps), or None
        Window regions used, on ``time_ps``; None when no window was applied.
    taper, center_on_peak, padding_ps, spline_half_width :
        The settings passed to ``analyze``.
    freq_min_hz, freq_max_hz : float or None
        The usable band, for plots and for whoever uses the result; it does not
        change any array above.
    """

    traces: tuple[TimeTrace, ...]
    reference_index: int
    time_ps: np.ndarray
    signal: np.ndarray
    reference_peak_time_ps: float
    peak_times_ps: np.ndarray
    window: np.ndarray
    windowed_signal: np.ndarray
    padded_time_ps: np.ndarray
    padded_signal: np.ndarray
    freq_hz: np.ndarray
    spectrum: np.ndarray
    field_ratio: np.ndarray
    power_ratio: np.ndarray
    window_regions_ps: tuple[tuple[float, float], ...] | None
    taper: float
    center_on_peak: bool
    padding_ps: float
    spline_half_width: int
    freq_min_hz: float | None
    freq_max_hz: float | None


def common_time_axis(traces: Sequence[TimeTrace], *, reference_index: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Put all traces on one uniform time axis.

    If every trace already has the same uniform axis (the usual case: one
    instrument, one scan setting), it is used as it is and nothing is
    interpolated. Otherwise the traces are linearly interpolated onto an axis
    with the reference's median time step, covering all traces; a trace is 0
    where it was not measured.

    Returns
    -------
    time_ps : ndarray, shape (N,)
        The common time axis in ps.
    signals : ndarray, shape (M, N)
        One trace per row.
    """
    # Every trace needs an increasing time axis of its own length; check before comparing them.
    if not traces:
        raise ValueError("no traces were given")
    if not 0 <= reference_index < len(traces):
        raise ValueError(f"reference_index {reference_index} is out of range for {len(traces)} traces")
    for index, trace in enumerate(traces):
        time_ps, signal = np.asarray(trace.time_ps), np.asarray(trace.signal)
        if time_ps.ndim != 1 or time_ps.shape != signal.shape or time_ps.size < 2:
            raise ValueError(f"trace {index}: time_ps and signal must be 1D, of equal length and at least 2 samples")
        if not (np.all(np.isfinite(time_ps)) and np.all(np.diff(time_ps) > 0)):
            raise ValueError(f"trace {index}: time_ps must be finite and strictly increasing")

    # Same uniform axis everywhere: use it as measured.
    reference_time_ps = np.asarray(traces[reference_index].time_ps, dtype=np.float64)
    steps_ps = np.diff(reference_time_ps)
    uniform = np.max(np.abs(steps_ps - np.mean(steps_ps))) <= 1e-6 * np.mean(steps_ps)
    if uniform and all(np.array_equal(trace.time_ps, reference_time_ps) for trace in traces):
        return reference_time_ps.copy(), np.array([trace.signal for trace in traces], dtype=np.float64)

    # Otherwise interpolate onto a uniform axis spanning every trace, at the reference's step.
    dt_ps = float(np.median(steps_ps))
    start_ps = min(float(trace.time_ps[0]) for trace in traces)
    stop_ps = max(float(trace.time_ps[-1]) for trace in traces)
    time_ps = start_ps + dt_ps * np.arange(int(np.floor((stop_ps - start_ps) / dt_ps + 1e-9)) + 1)
    signals = np.array([np.interp(time_ps, trace.time_ps, trace.signal, left=0.0, right=0.0) for trace in traces])
    return time_ps, signals


def analyze(
    traces: Iterable[TimeTrace],
    *,
    reference_index: int = 0,
    window_regions_ps: Sequence[tuple[float, float]] | None = None,
    taper: float = 0.5,
    center_on_peak: bool = True,
    padding_ps: float = 0.0,
    spline_half_width: int = 5,
    freq_min_hz: float | None = None,
    freq_max_hz: float | None = None,
) -> Analysis:
    """Turn measured traces into spectra and field ratios relative to a reference, keeping every stage.

    Parameters
    ----------
    traces : iterable of TimeTrace
        The reference and the samples, e.g. from :func:`load_time_trace`.
    reference_index : int
        Which trace is the reference (default 0, the first).
    window_regions_ps : sequence of (start_ps, stop_ps), optional
        Time regions to keep, in ps on the *centred* axis (reference peak at
        t = 0), e.g. ``[(-3.0, 8.0)]``. Each trace gets a Tukey window per region
        (see :func:`trace_window`). None (default): no window.
    taper : float
        Fraction of each window in its cosine edges, 0 (rectangle) to 1 (Hann).
        Default 0.5. Only used with window regions.
    center_on_peak : bool
        Move each window onto the trace's own peak inside the region (default
        True), so one region fits a reference and a delayed sample alike.
    padding_ps : float
        Time of zeros added, half before and half after the traces, in ps
        (default 0). Only makes the frequency grid finer; see :func:`zero_pad`.
    spline_half_width : int
        Samples on each side of the largest one in the peak spline (default 5);
        see :func:`find_peak`.
    freq_min_hz, freq_max_hz : float or None
        The usable frequency band, stored in the result for plots. It does not
        limit the computation; every array covers 0 to the Nyquist frequency.

    Returns
    -------
    Analysis
        Every intermediate stage and the settings used.
    """
    # 1. One uniform time axis for all traces; a tuple, so a generator of traces works as well.
    traces = tuple(traces)
    time_ps, signal = common_time_axis(traces, reference_index=reference_index)

    # 2. Reference peak at t = 0; the samples keep their delay relative to it.
    time_ps, peak_times_ps, reference_peak_time_ps = center_on_reference_peak(
        time_ps, signal, reference_index=reference_index, spline_half_width=spline_half_width,
    )

    # 3. Window each trace around its pulse, or leave it whole.
    if window_regions_ps is None:
        regions = None
        window = np.ones_like(signal)
    else:
        regions = tuple((float(start), float(stop)) for start, stop in window_regions_ps)
        window = np.array([
            trace_window(time_ps, trace, regions, taper=taper, center_on_peak=center_on_peak) for trace in signal
        ])
    windowed_signal = signal * window

    # 4.-6. Pad, transform, and divide by the reference spectrum.
    padded_time_ps, padded_signal = zero_pad(time_ps, windowed_signal, padding_ps)
    freq_hz, spectrum = fourier_transform(padded_time_ps, padded_signal)
    field_ratio = compute_complex_field_ratio(spectrum, spectrum[reference_index])

    return Analysis(
        traces=traces,
        reference_index=reference_index,
        time_ps=time_ps,
        signal=signal,
        reference_peak_time_ps=reference_peak_time_ps,
        peak_times_ps=peak_times_ps,
        window=window,
        windowed_signal=windowed_signal,
        padded_time_ps=padded_time_ps,
        padded_signal=padded_signal,
        freq_hz=freq_hz,
        spectrum=spectrum,
        field_ratio=field_ratio,
        power_ratio=np.abs(field_ratio) ** 2,
        window_regions_ps=regions,
        taper=taper,
        center_on_peak=center_on_peak,
        padding_ps=padding_ps,
        spline_half_width=spline_half_width,
        freq_min_hz=freq_min_hz,
        freq_max_hz=freq_max_hz,
    )
