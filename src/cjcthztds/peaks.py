"""Finding the pulse peak between samples, and centring a time axis on the reference peak.

Why sub-sample: a typical THz trace is sampled every 20–50 fs, while the
delay between two traces (a sample, a drift of the setup) can be a fraction
of that. Taking the largest sample would round every peak time to the grid.
A cubic spline through the samples around the largest one follows the pulse
shape between the samples, and its maximum gives the peak time and height.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import CubicSpline

from cjcthztds.traces import TimeTraceMap


@dataclass(frozen=True)
class Peak:
    """The peak of a pulse: the maximum of a cubic spline through the samples around the largest one.

    Attributes
    ----------
    time_ps : float
        Time of the peak in ps.
    amplitude : float
        Signal at the peak, with its sign (a negative pulse has a negative peak).
    """

    time_ps: float
    amplitude: float


def _spline_around_peak(time_ps: np.ndarray, signal: np.ndarray, spline_half_width: int) -> tuple[slice, float]:
    """The samples around the largest ``|signal|`` (as a slice) and the sign of that sample."""
    # Validate the inputs, so a mistake gives a clear message instead of a scipy error.
    if time_ps.ndim != 1 or time_ps.shape != signal.shape:
        raise ValueError(
            f"time_ps and signal must be 1D arrays of equal length, got {time_ps.shape} and {signal.shape}"
        )
    if spline_half_width < 2:
        raise ValueError(f"spline_half_width must be at least 2, got {spline_half_width}")

    # The largest |signal| sample and its neighbours; at the edge of the trace the window is cut off.
    k = int(np.argmax(np.abs(signal)))
    lo, hi = max(0, k - spline_half_width), min(signal.size, k + spline_half_width + 1)
    if hi - lo < 4:
        raise ValueError("fewer than 4 samples around the peak; the peak lies at the edge of the trace")
    return slice(lo, hi), (1.0 if signal[k] >= 0 else -1.0)


def _spline_maximum(time_ps: np.ndarray, signal: np.ndarray, sign: float) -> tuple[float, float]:
    """Time and value of the largest ``sign * spline`` between the first and last of the given samples.

    The maximum is either where the spline's derivative is zero or at a sample,
    so those are the only candidates: the result is exact, with no iterative
    search. The samples themselves are candidates too, so the result never
    lies below the largest sample (e.g. on pure noise, where the spline may
    have no real peak).
    """
    # Zeros of the derivative inside the samples (NaN marks a piece where it is zero throughout), plus the samples.
    spline = CubicSpline(time_ps, signal)
    roots = spline.derivative().roots(extrapolate=False)
    candidates = np.concatenate([roots[np.isfinite(roots)], time_ps])
    best = candidates[np.argmax(sign * spline(candidates))]
    return float(best), float(spline(best))


def find_peak(time_ps: np.ndarray, signal: np.ndarray, *, spline_half_width: int = 5) -> Peak:
    """Find the peak of a pulse between samples.

    The largest ``|signal|`` sample is the rough peak. A cubic spline is drawn
    through it and ``spline_half_width`` samples on each side (scipy's
    ``CubicSpline``, "not-a-knot" ends), and its largest ``|value|`` is the
    peak. A spline passes through every sample and follows any smooth pulse
    shape between them, so it has no shape bias. A sampled polynomial of
    degree 3 or less is reproduced exactly.

    No uncertainty is given: the spline interpolates, it does not fit, so there
    is no fit covariance to propagate, and noise on the samples goes straight
    into the peak.

    Parameters
    ----------
    time_ps : ndarray, shape (N,)
        Time axis in ps.
    signal : ndarray, shape (N,)
        The trace.
    spline_half_width : int
        Samples on each side of the largest one the spline goes through
        (default 5, so 11 samples). At least 2.

    Returns
    -------
    Peak
        Peak time and signed amplitude.
    """
    # The spline goes through the samples around the largest one.
    time_ps = np.asarray(time_ps, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    window, sign = _spline_around_peak(time_ps, signal, spline_half_width)
    peak_time_ps, amplitude = _spline_maximum(time_ps[window], signal[window], sign)
    return Peak(peak_time_ps, amplitude)


def peak_spline_curve(
    time_ps: np.ndarray, signal: np.ndarray, *, spline_half_width: int = 5, n_points: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """The spline :func:`find_peak` uses, evaluated on a fine time grid, for plotting.

    Returns ``(curve_time_ps, curve_signal)``, ``n_points`` values from the first
    to the last sample the spline goes through.
    """
    # The same samples and spline as find_peak, evaluated finely.
    time_ps = np.asarray(time_ps, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    window, _ = _spline_around_peak(time_ps, signal, spline_half_width)
    curve_time_ps = np.linspace(time_ps[window][0], time_ps[window][-1], n_points)
    return curve_time_ps, CubicSpline(time_ps[window], signal[window])(curve_time_ps)


def center_on_reference_peak(
    time_ps: np.ndarray, signals: np.ndarray, *, reference_index: int = 0, spline_half_width: int = 5,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Shift a shared time axis so the reference pulse peaks at t = 0.

    Only the time axis moves; the signals are not changed, so every trace keeps
    its delay relative to the reference. That delay is what a sample adds (its
    optical thickness), so it must not be removed.

    Parameters
    ----------
    time_ps : ndarray, shape (N,)
        Time axis in ps shared by all traces.
    signals : ndarray, shape (M, N)
        The traces, one per row.
    reference_index : int
        Row of the reference trace.
    spline_half_width : int
        Passed to :func:`find_peak`.

    Returns
    -------
    centered_time_ps : ndarray, shape (N,)
        ``time_ps`` minus the reference peak time.
    peak_times_ps : ndarray, shape (M,)
        Peak time of every trace on the centred axis (0 for the reference;
        the delay of each trace relative to the reference).
    reference_peak_time_ps : float
        The reference peak time on the original axis: the shift that was applied.
    """
    # Every trace's own peak, so the delays relative to the reference are known as well.
    signals = np.atleast_2d(np.asarray(signals, dtype=np.float64))
    if not 0 <= reference_index < signals.shape[0]:
        raise ValueError(f"reference_index {reference_index} is out of range for {signals.shape[0]} traces")
    peak_times_ps = np.array([
        find_peak(time_ps, trace, spline_half_width=spline_half_width).time_ps for trace in signals
    ])

    # Move the time origin to the reference peak.
    reference_peak_time_ps = float(peak_times_ps[reference_index])
    centered_time_ps = np.asarray(time_ps, dtype=np.float64) - reference_peak_time_ps
    return centered_time_ps, peak_times_ps - reference_peak_time_ps, reference_peak_time_ps


def peak_map(trace_map: TimeTraceMap, *, spline_half_width: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Peak time and peak amplitude at every position of a :class:`TimeTraceMap`.

    Returns
    -------
    peak_time_ps, peak_amplitude : ndarray, shape (ny, nx)
        The peak time in ps and the signed peak amplitude of each trace.
    """
    # One spline per position; maps are small (hundreds to a few thousand traces).
    ny, nx = trace_map.signal.shape[:2]
    peak_time_ps = np.empty((ny, nx))
    peak_amplitude = np.empty((ny, nx))
    for iy in range(ny):
        for ix in range(nx):
            peak = find_peak(trace_map.time_ps, trace_map.signal[iy, ix], spline_half_width=spline_half_width)
            peak_time_ps[iy, ix], peak_amplitude[iy, ix] = peak.time_ps, peak.amplitude
    return peak_time_ps, peak_amplitude
