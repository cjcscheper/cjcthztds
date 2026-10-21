"""Tukey time windows: keep the pulse, cut off what comes later (echoes, noise).

A window multiplies the trace by 1 where the pulse is and by 0 elsewhere. A
sharp (rectangular) cut puts a step into the trace wherever the signal is
not zero there, and a step rings across the whole spectrum. The Tukey window
goes to zero along a half cosine instead: smoother edges, less ringing, at
the cost of weighting the pulse edges down. THEORY.md discusses the trade-off.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def tukey_window(time_ps: np.ndarray, start_ps: float, stop_ps: float, *, taper: float = 0.5) -> np.ndarray:
    """A Tukey window that is non-zero from ``start_ps`` to ``stop_ps``.

    With ``x = (t - start) / (stop - start)`` running from 0 to 1 across the
    window, and ``taper`` the fraction of the window spent on the two cosine
    edges together::

        w = (1 - cos(2 pi x / taper)) / 2          for 0 <= x < taper / 2
        w = 1                                       in the flat middle
        w = (1 - cos(2 pi (1 - x) / taper)) / 2    for 1 - taper / 2 < x <= 1
        w = 0                                       outside [start, stop]

    ``taper = 0`` is a rectangle, ``taper = 1`` a Hann window (no flat part).

    Parameters
    ----------
    time_ps : ndarray, shape (N,)
        Time axis in ps.
    start_ps, stop_ps : float
        Where the window starts and stops (``start_ps < stop_ps``), in ps.
    taper : float
        Fraction of the window in the cosine edges, from 0 to 1 (default 0.5).

    Returns
    -------
    ndarray, shape (N,)
        The window, between 0 and 1.
    """
    # A reversed or empty window, or a taper outside [0, 1], is a mistake, not a request.
    if not stop_ps > start_ps:
        raise ValueError(f"the window must have start_ps < stop_ps, got {start_ps} and {stop_ps}")
    if not 0.0 <= taper <= 1.0:
        raise ValueError(f"taper must be between 0 and 1, got {taper}")

    # Position inside the window, 0 at the start and 1 at the stop.
    x = (np.asarray(time_ps, dtype=np.float64) - start_ps) / (stop_ps - start_ps)
    window = ((x >= 0.0) & (x <= 1.0)).astype(np.float64)

    # Cosine edges; with taper = 0 they are empty and the window stays a rectangle.
    if taper > 0.0:
        rising = (x >= 0.0) & (x < taper / 2.0)
        falling = (x > 1.0 - taper / 2.0) & (x <= 1.0)
        window[rising] = 0.5 * (1.0 - np.cos(2.0 * np.pi * x[rising] / taper))
        window[falling] = 0.5 * (1.0 - np.cos(2.0 * np.pi * (1.0 - x[falling]) / taper))
    return window


def trace_window(
    time_ps: np.ndarray,
    signal: np.ndarray,
    regions_ps: Iterable[tuple[float, float]],
    *,
    taper: float = 0.5,
    center_on_peak: bool = True,
) -> np.ndarray:
    """The window for one trace: one Tukey window per region, combined by taking the largest value.

    Parameters
    ----------
    time_ps : ndarray, shape (N,)
        Time axis in ps.
    signal : ndarray, shape (N,)
        The trace; only used to find its peak when ``center_on_peak`` is True.
    regions_ps : iterable of (start_ps, stop_ps)
        The selected time regions in ps, in any order; start and stop may be
        swapped.
    taper : float
        Passed to :func:`tukey_window`.
    center_on_peak : bool
        If True (default), each window keeps the width of its region but is
        moved so its centre lies on the largest ``|signal|`` sample inside the
        region. Then one region drawn around the reference pulse also fits a
        sample pulse that arrives a little later.

    Returns
    -------
    ndarray, shape (N,)
        The combined window, between 0 and 1.
    """
    # At least one region is needed; an empty list almost always means the selection went wrong.
    time_ps = np.asarray(time_ps, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    regions = [(min(a, b), max(a, b)) for a, b in regions_ps]
    if not regions:
        raise ValueError("no window regions were given")

    # One window per region; where windows overlap, the larger value wins.
    combined = np.zeros_like(time_ps)
    for start_ps, stop_ps in regions:
        inside = (time_ps >= start_ps) & (time_ps <= stop_ps)
        if center_on_peak and np.any(inside):
            peak_ps = time_ps[inside][np.argmax(np.abs(signal[inside]))]
            half_width_ps = 0.5 * (stop_ps - start_ps)
            start_ps, stop_ps = peak_ps - half_width_ps, peak_ps + half_width_ps
        combined = np.maximum(combined, tukey_window(time_ps, start_ps, stop_ps, taper=taper))
    return combined
