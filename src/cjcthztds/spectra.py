"""From time traces to spectra: Fourier transform, complex field ratio, band limits, phase.

Sign convention (THEORY.md derives it and compares it with cjctmm): the
spectrum is ``E(f) = sum_n E(t_n) exp(-i 2 pi f t_n) dt``, the same kernel as
numpy's forward FFT. In this convention a pulse that arrives a time ``Δt``
later has its spectrum multiplied by ``exp(-i 2 pi f Δt)``, so the phase of
the field ratio of a delayed sample *decreases* with frequency, with slope
``-Δt`` against ``ω = 2 pi f``.
"""

from __future__ import annotations

import numpy as np

from cjcthztds.traces import TimeTrace

# ps -> s, so the frequencies come out in Hz.
_SECONDS_PER_PS = 1e-12


def fourier_transform(time_ps: np.ndarray, signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fourier transform of real time traces, at the non-negative frequencies.

    Computes ``E(f_k) = sum_n E(t_n) exp(-i 2 pi f_k t_n) dt`` with
    ``f_k = k / (N dt)``. This is numpy's ``rfft`` times two factors that make it
    an approximation of the continuous Fourier integral
    ``∫ E(t) exp(-i 2 pi f t) dt``:

    - ``dt``, so the amplitude does not depend on how finely the trace was
      sampled or how much it was zero-padded;
    - ``exp(-i 2 pi f t_0)``, the phase of the first sample's time ``t_0``, so
      the phase refers to t = 0 of the time axis and not to wherever the
      trace happens to start. A pulse at t = 0 then has a flat phase.

    Field ratios (sample over reference on the same time axis) do not depend
    on either factor; spectra of single traces and padded spectra do.

    Parameters
    ----------
    time_ps : ndarray, shape (N,)
        Uniform time axis in ps.
    signals : ndarray, shape (N,) or (M, N)
        Real traces, one per row.

    Returns
    -------
    freq_hz : ndarray, shape (N // 2 + 1,)
        Frequencies in Hz, from 0 to the Nyquist frequency ``1 / (2 dt)``.
    spectrum : ndarray, complex, shape (N // 2 + 1,) or (M, N // 2 + 1)
        The spectrum of each trace, in signal units times seconds.
    """
    # The transform assumes equal time steps; refuse an axis that does not have them.
    time_ps = np.asarray(time_ps, dtype=np.float64)
    signals = np.asarray(signals, dtype=np.float64)
    if time_ps.ndim != 1 or time_ps.size < 2 or signals.shape[-1] != time_ps.size:
        raise ValueError(f"time_ps must be 1D with at least 2 samples and match the last axis of signals, "
                         f"got {time_ps.shape} and {signals.shape}")
    steps_ps = np.diff(time_ps)
    dt_ps = (time_ps[-1] - time_ps[0]) / (time_ps.size - 1)
    if not dt_ps > 0 or np.max(np.abs(steps_ps - dt_ps)) > 1e-3 * dt_ps:
        raise ValueError("time_ps must increase in equal steps (within 0.1 %); resample the traces first")

    # Only the non-negative frequencies: a real trace has E(-f) = E(f)*, so the rest is redundant.
    dt = dt_ps * _SECONDS_PER_PS
    freq_hz = np.fft.rfftfreq(time_ps.size, d=dt)
    spectrum = np.fft.rfft(signals, axis=-1) * dt * np.exp(-2j * np.pi * freq_hz * time_ps[0] * _SECONDS_PER_PS)
    return freq_hz, spectrum


def compute_complex_field_ratio(spectrum: np.ndarray, reference_spectrum: np.ndarray) -> np.ndarray:
    """The complex field ratio ``E_sample(f) / E_reference(f)``.

    Valid for transmission and reflection alike: with the reference measured
    without the sample (transmission) or on a mirror (reflection), the ratio is
    the sample's complex transmission or reflection coefficient, up to known
    factors of the reference path. ``|ratio|²`` is the corresponding power ratio.

    Where the reference spectrum is exactly zero the ratio is NaN. Outside the
    bandwidth of the setup the reference is tiny but not zero, and the ratio is
    noise divided by noise: use :func:`frequency_range_mask` to keep to the usable band.

    Parameters
    ----------
    spectrum : ndarray, complex, shape (K,) or (M, K)
        Spectra of the samples (the reference row itself gives exactly 1).
    reference_spectrum : ndarray, complex, shape (K,)
        Spectrum of the reference.
    """
    # Divide where the reference is non-zero; NaN marks the frequencies where that is impossible.
    spectrum = np.asarray(spectrum, dtype=np.complex128)
    reference_spectrum = np.asarray(reference_spectrum, dtype=np.complex128)
    ratio = np.full(np.broadcast_shapes(spectrum.shape, reference_spectrum.shape), np.nan + 0j)
    np.divide(spectrum, reference_spectrum, out=ratio, where=reference_spectrum != 0)
    return ratio


def frequency_range_mask(
    freq_hz: np.ndarray, freq_min_hz: float | None = None, freq_max_hz: float | None = None,
) -> np.ndarray:
    """True for the frequencies inside ``[freq_min_hz, freq_max_hz]``; None means no limit on that side."""
    # Start from every frequency and remove the ones outside each given limit.
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    mask = np.ones(freq_hz.shape, dtype=bool)
    if freq_min_hz is not None:
        mask &= freq_hz >= freq_min_hz
    if freq_max_hz is not None:
        mask &= freq_hz <= freq_max_hz
    return mask


def unwrapped_phase(
    freq_hz: np.ndarray, field_ratio: np.ndarray, freq_min_hz: float | None = None, freq_max_hz: float | None = None,
) -> np.ndarray:
    """The continuous phase of a field ratio inside a frequency band, NaN outside it.

    ``np.angle`` returns the phase modulo 2π, so a phase that keeps decreasing
    (a delayed sample) jumps by 2π every time it passes -π. Unwrapping removes
    those jumps, but only where the steps between frequencies are smaller than
    π, which fails in the noise outside the band. So the phase is unwrapped
    only inside ``[freq_min_hz, freq_max_hz]``. Unwrapping fixes the phase only up
    to a whole multiple of 2π; that multiple is chosen so a straight-line fit
    of the phase against frequency passes within π of zero at f = 0, as the
    phase of a real (causal) response must.

    Parameters
    ----------
    freq_hz : ndarray, shape (K,)
        Frequencies in Hz.
    field_ratio : ndarray, complex, shape (K,) or (M, K)
        Field ratio(s), one per row.
    freq_min_hz, freq_max_hz : float or None
        The band to unwrap in; None means no limit on that side.

    Returns
    -------
    ndarray, shape like ``field_ratio``
        Phase in rad inside the band, NaN outside.
    """
    # Unwrap only inside the band, starting at its lowest frequency.
    freq_hz = np.asarray(freq_hz, dtype=np.float64)
    mask = frequency_range_mask(freq_hz, freq_min_hz, freq_max_hz)
    if mask.sum() < 2:
        raise ValueError("the frequency band holds fewer than 2 frequencies")
    band_phase = np.unwrap(np.angle(np.asarray(field_ratio)[..., mask]), axis=-1)

    # Remove the whole multiple of 2 pi that a line fit puts at f = 0.
    _, intercept = np.polyfit(freq_hz[mask], np.atleast_2d(band_phase).T, deg=1)
    band_phase = band_phase - 2.0 * np.pi * np.round(intercept / (2.0 * np.pi)).reshape(band_phase.shape[:-1] + (1,))

    # Same shape as the input, NaN outside the band.
    phase = np.full(np.shape(field_ratio), np.nan)
    phase[..., mask] = band_phase
    return phase


def amplitude_spectrum(trace: TimeTrace) -> tuple[np.ndarray, np.ndarray]:
    """``|E(f)|`` of one trace, for a quick look at bandwidth and noise floor.

    Returns ``(freq_hz, amplitude)``, with the amplitude in signal units times
    seconds (see :func:`fourier_transform`). The trace must be uniformly sampled.
    """
    # No windowing or padding: the spectrum of the trace as measured.
    freq_hz, spectrum = fourier_transform(trace.time_ps, trace.signal)
    return freq_hz, np.abs(spectrum)
