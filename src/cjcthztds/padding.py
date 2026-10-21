"""Zero padding: append zeros before and after a trace to sample its spectrum more finely.

The discrete Fourier transform of N samples spaced dt apart gives the
spectrum at frequencies spaced 1 / (N dt). Adding zeros makes N larger and the
spacing finer, but adds no measured information: the padded spectrum
interpolates the same spectrum between the old frequencies (it passes
exactly through the old values when the length grows by a whole factor).
Frequency resolution, the ability to separate two close features, is set by
the measured time span alone. See THEORY.md.
"""

from __future__ import annotations

import numpy as np


def zero_pad(time_ps: np.ndarray, signals: np.ndarray, padding_ps: float) -> tuple[np.ndarray, np.ndarray]:
    """Add ``padding_ps / 2`` of zeros before and after the traces, keeping the time step.

    Parameters
    ----------
    time_ps : ndarray, shape (N,)
        Uniform time axis in ps.
    signals : ndarray, shape (N,) or (M, N)
        One trace, or one trace per row.
    padding_ps : float
        Total time of zeros to add, in ps; half goes before the trace and half
        after, rounded to whole time steps. 0 returns the input unchanged.

    Returns
    -------
    padded_time_ps : ndarray, shape (N + 2 p,)
        The time axis extended by ``p`` steps on each side.
    padded_signals : ndarray, shape (N + 2 p,) or (M, N + 2 p)
        The traces with ``p`` zeros on each side.
    """
    # Negative padding would cut the trace; that is never what padding is for.
    time_ps = np.asarray(time_ps, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    if padding_ps < 0:
        raise ValueError(f"padding_ps must be 0 or positive, got {padding_ps}")
    if time_ps.ndim != 1 or time_ps.size < 2 or signals.shape[-1] != time_ps.size:
        raise ValueError(f"time_ps must be 1D with at least 2 samples and match the last axis of signals, "
                         f"got {time_ps.shape} and {signals.shape}")

    # Whole steps of the existing grid, so the padded samples continue it exactly.
    dt_ps = (time_ps[-1] - time_ps[0]) / (time_ps.size - 1)
    steps_per_side = int(round(0.5 * padding_ps / dt_ps))
    if steps_per_side == 0:
        return time_ps.copy(), signals.copy()

    # Extend the axis on both sides and put the traces in the middle.
    padded_time_ps = time_ps[0] + dt_ps * np.arange(-steps_per_side, time_ps.size + steps_per_side)
    widths = [(0, 0)] * (signals.ndim - 1) + [(steps_per_side, steps_per_side)]
    return padded_time_ps, np.pad(signals, widths)
