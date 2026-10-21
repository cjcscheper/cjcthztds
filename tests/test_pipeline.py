"""analyze(): every stage kept with consistent shapes, the settings stored, and simple sample models recovered."""

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from cjcthztds import (
    Analysis,
    TimeTrace,
    analyze,
    frequency_range_mask,
    common_time_axis,
    load_time_trace,
    unwrapped_phase,
)

DATA = Path(__file__).resolve().parent / "data"
TIME_PS = -10.0 + 0.02 * np.arange(1500)


def single_cycle_pulse(time_ps, center_ps=0.0):
    """Derivative of a Gaussian with peaks of +-1, 0.2 ps from its centre."""
    x = (time_ps - center_ps) / 0.2
    return -x * np.exp(0.5 * (1.0 - x**2))


def slab_traces(echo=True):
    """Reference at 0.3 ps; sample delayed 2 ps and scaled 0.6, with an echo 0.1 x the pulse a further 6 ps later."""
    reference = single_cycle_pulse(TIME_PS, 0.3)
    sample = 0.6 * single_cycle_pulse(TIME_PS, 2.3) + (0.06 * single_cycle_pulse(TIME_PS, 8.3) if echo else 0.0)
    return [TimeTrace(TIME_PS, reference), TimeTrace(TIME_PS, sample)]


def test_analyze_keeps_every_stage_with_consistent_shapes():
    """Every array has the documented shape: (N,), (M, N), (N_pad,), (M, N_pad), (K,), (M, K)."""
    traces = slab_traces()
    analysis = analyze(traces, window_regions_ps=[(-2.0, 4.0)], padding_ps=10.0)
    m, n = 2, TIME_PS.size
    n_pad = n + 2 * 250  # 10 ps of padding at 0.02 ps steps: 250 zeros on each side
    k = n_pad // 2 + 1
    assert isinstance(analysis, Analysis)
    assert analysis.time_ps.shape == (n,)
    for name in ("signal", "window", "windowed_signal"):
        assert getattr(analysis, name).shape == (m, n), name
    assert analysis.padded_time_ps.shape == (n_pad,) and analysis.padded_signal.shape == (m, n_pad)
    assert analysis.freq_hz.shape == (k,)
    for name in ("spectrum", "field_ratio", "power_ratio"):
        assert getattr(analysis, name).shape == (m, k), name
    assert analysis.peak_times_ps.shape == (m,)
    assert analysis.traces == tuple(traces)
    # The stages fit together: windowed = signal * window, and the padded traces hold the windowed ones.
    assert np.array_equal(analysis.windowed_signal, analysis.signal * analysis.window)
    assert np.array_equal(analysis.padded_signal[:, 250:-250], analysis.windowed_signal)
    assert np.allclose(analysis.power_ratio, np.abs(analysis.field_ratio) ** 2, equal_nan=True)


def test_analyze_stores_its_settings():
    """The settings used are stored in the result, and the result is frozen."""
    analysis = analyze(slab_traces(), window_regions_ps=[[-2, 4]], taper=0.25, center_on_peak=False,
                       padding_ps=1.0, spline_half_width=3, freq_min_hz=0.2e12, freq_max_hz=2e12)
    assert analysis.window_regions_ps == ((-2.0, 4.0),)
    assert analysis.taper == 0.25 and analysis.center_on_peak is False
    assert analysis.padding_ps == 1.0 and analysis.spline_half_width == 3
    assert (analysis.freq_min_hz, analysis.freq_max_hz) == (0.2e12, 2e12)
    # setattr instead of analysis.taper = ...: the write is meant to fail, and this way the IDE does not flag it.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(analysis, "taper", 0.5)


def test_centring_and_delays():
    """The reference peaks at t = 0 on the centred axis; the sample's peak time is its 2 ps delay."""
    analysis = analyze(slab_traces())
    # The pulse's + lobe (largest |signal|, first of two equal lobes) is at centre - 0.2 ps = 0.1 ps.
    assert analysis.reference_peak_time_ps == pytest.approx(0.1, abs=0.01)
    assert analysis.peak_times_ps[0] == 0.0
    assert analysis.peak_times_ps[1] == pytest.approx(2.0, abs=1e-3)
    assert np.allclose(analysis.time_ps, TIME_PS - analysis.reference_peak_time_ps)


def test_without_window_regions_nothing_is_windowed():
    """Without regions the window is all ones and the traces go to the FFT unchanged."""
    analysis = analyze(slab_traces())
    assert analysis.window_regions_ps is None
    assert np.all(analysis.window == 1.0)
    assert np.array_equal(analysis.windowed_signal, analysis.signal)


def test_frequency_band_does_not_change_the_computation():
    """freq_min_hz / freq_max_hz are stored for plots; every computed array is the same with and without them."""
    full = analyze(slab_traces())
    limited = analyze(slab_traces(), freq_min_hz=0.2e12, freq_max_hz=2e12)
    for field in ("freq_hz", "spectrum", "field_ratio", "power_ratio", "padded_signal"):
        assert np.array_equal(getattr(full, field), getattr(limited, field), equal_nan=True), field


def test_field_ratio_of_an_attenuated_delayed_sample():
    """Sample = 0.6 x reference delayed 2 ps: |t| = 0.6 and phase -omega x 2 ps in the band; reference row = 1."""
    analysis = analyze(slab_traces(echo=False))
    band = frequency_range_mask(analysis.freq_hz, 0.1e12, 3e12)
    assert np.allclose(analysis.field_ratio[0][band], 1.0, atol=1e-14)
    assert np.allclose(np.abs(analysis.field_ratio[1][band]), 0.6, atol=1e-9)
    phase = unwrapped_phase(analysis.freq_hz, analysis.field_ratio[1], 0.1e12, 3e12)
    assert np.allclose(phase[band], -2 * np.pi * analysis.freq_hz[band] * 2e-12, atol=1e-8)


def test_window_removes_echo_fringes():
    """The echo makes |t| oscillate around 0.6 (Fabry-Perot fringes); a window around the main pulse removes them."""
    band_low, band_high = 0.2e12, 2.0e12
    whole = analyze(slab_traces())
    windowed = analyze(slab_traces(), window_regions_ps=[(-3.0, 3.0)], taper=0.5)
    band = frequency_range_mask(whole.freq_hz, band_low, band_high)
    # Echo amplitude 0.06 / 0.6 = 0.1 relative to the main pulse: |t| swings between about 0.54 and 0.66.
    assert np.ptp(np.abs(whole.field_ratio[1][band])) > 0.1
    # The window (centred on each trace's own peak) keeps only the main pulse: |t| is 0.6 to within 1 %.
    assert np.allclose(np.abs(windowed.field_ratio[1][band]), 0.6, rtol=0.01)


def test_other_reference_index():
    """With reference_index=1 the second trace defines t = 0 and divides the spectra."""
    analysis = analyze(slab_traces(echo=False), reference_index=1)
    band = frequency_range_mask(analysis.freq_hz, 0.1e12, 3e12)
    assert analysis.peak_times_ps[1] == 0.0 and analysis.peak_times_ps[0] == pytest.approx(-2.0, abs=1e-3)
    assert np.allclose(np.abs(analysis.field_ratio[0][band]), 1 / 0.6, rtol=1e-9)


def test_traces_on_different_axes_are_resampled():
    """Traces with different time axes are interpolated onto one uniform axis at the reference step."""
    reference = TimeTrace(TIME_PS, single_cycle_pulse(TIME_PS))
    other_time_ps = -5.0 + 0.01 * np.arange(1500)
    sample = TimeTrace(other_time_ps, single_cycle_pulse(other_time_ps, 1.0))
    time_ps, signals = common_time_axis([reference, sample])
    assert np.allclose(np.diff(time_ps), 0.02)
    assert time_ps[0] == pytest.approx(-10.0) and time_ps[-1] == pytest.approx(TIME_PS[-1])
    assert signals.shape == (2, time_ps.size)
    # Outside the sample's own range (before -5 ps) it is 0.
    assert np.all(signals[1][time_ps < -5.0] == 0.0)
    assert analyze([reference, sample]).peak_times_ps[1] == pytest.approx(1.0, abs=2e-3)


def test_analyze_files_from_two_instruments():
    """A Menlo and a Toptica test file (same time grid) analyse together; identical pulses give |t| = 1 in band."""
    traces = [load_time_trace(DATA / "menlo_dat.dat"), load_time_trace(DATA / "toptica_pulse_csv.pulse.csv")]
    analysis = analyze(traces, freq_min_hz=0.2e12, freq_max_hz=1.5e12)
    band = frequency_range_mask(analysis.freq_hz, 0.2e12, 1.5e12)
    # Both files hold the same pulse with independent 1e-3 noise.
    assert np.allclose(np.abs(analysis.field_ratio[1][band]), 1.0, atol=0.02)


def test_analyze_validation():
    """No traces, a bad reference index or a broken time axis raise clear ValueErrors."""
    with pytest.raises(ValueError, match="no traces were given"):
        analyze([])
    with pytest.raises(ValueError, match="reference_index 2 is out of range for 2 traces"):
        analyze(slab_traces(), reference_index=2)
    with pytest.raises(ValueError, match="trace 1: time_ps must be finite and strictly increasing"):
        analyze([TimeTrace(TIME_PS, TIME_PS), TimeTrace(TIME_PS[::-1], TIME_PS)])
