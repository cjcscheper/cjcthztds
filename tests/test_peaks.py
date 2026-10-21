"""Peak finding between samples, and centring a time axis on the reference peak."""

import dataclasses

import numpy as np
import pytest

from cjcthztds import Peak, TimeTraceMap, center_on_reference_peak, find_peak, peak_map, peak_spline_curve

# 50 fs steps, as in a typical TDS scan.
TIME_PS = np.arange(-5.0, 5.0, 0.05)


def gaussian(time_ps, center_ps, width_ps=0.3):
    """A Gaussian pulse of height 1."""
    return np.exp(-0.5 * ((time_ps - center_ps) / width_ps) ** 2)


def test_cubic_is_found_exactly():
    """The spline reproduces a sampled cubic exactly: its maximum between samples, to rounding."""
    # p(t) = 2 - 3 (t - 0.123)^2 + 0.5 (t - 0.123)^3 has its maximum at t = 0.123 with p = 2; a short axis keeps
    # it the largest |signal|.
    time_ps = np.arange(-0.5, 0.5, 0.05)
    peak = find_peak(time_ps, 2.0 - 3.0 * (time_ps - 0.123) ** 2 + 0.5 * (time_ps - 0.123) ** 3)
    assert peak.time_ps == pytest.approx(0.123, abs=1e-12)
    assert peak.amplitude == pytest.approx(2.0, abs=1e-12)


def test_negative_pulse_keeps_its_sign():
    """A negative pulse has a negative peak amplitude at the right time."""
    peak = find_peak(TIME_PS, -1.5 * gaussian(TIME_PS, 0.4))
    assert peak.amplitude == pytest.approx(-1.5, rel=1e-5)
    assert peak.time_ps == pytest.approx(0.4, abs=1e-4)


@pytest.mark.parametrize("center_ps", [0.0, 0.011, 0.0237, -1.0391])
def test_shifted_pulse_is_found_between_samples(center_ps):
    """A Gaussian shifted by a fraction of a step is found to well below the 50 fs step."""
    peak = find_peak(TIME_PS, gaussian(TIME_PS, center_ps))
    # Within 0.1 fs (0.2 % of a step): the spline through 11 samples follows the 0.3 ps wide Gaussian closely.
    assert peak.time_ps == pytest.approx(center_ps, abs=1e-4)
    assert peak.amplitude == pytest.approx(1.0, abs=1e-5)


def test_noise_only_traces_give_the_largest_sample_at_most():
    """On pure noise the peak stays within the spline's samples and close to the largest sample."""
    rng = np.random.default_rng(3)
    for _ in range(500):
        noise = 2e-3 * rng.standard_normal(TIME_PS.size)
        k = np.argmax(np.abs(noise))
        peak = find_peak(TIME_PS, noise)
        # The maximum is searched only between the 11 samples, and a spline does not overshoot them by much.
        assert abs(peak.time_ps - TIME_PS[k]) <= 5 * 0.05 + 1e-12
        # Never below the largest sample (up to rounding of the spline evaluated at that sample).
        assert np.abs(noise).max() * (1 - 1e-12) <= abs(peak.amplitude) <= 2 * np.abs(noise).max()


def test_rising_edge_gives_the_largest_sample():
    """Samples that rise steadily to the end of the trace have no maximum in between: the last sample is the peak."""
    time_ps = np.arange(5) * 1.0
    peak = find_peak(time_ps, np.array([0.0, 0.1, 0.5, 0.8, 1.0]))
    assert (peak.time_ps, peak.amplitude) == (4.0, 1.0)


def test_peak_spline_curve_goes_through_the_samples():
    """peak_spline_curve spans the 11 samples around the peak and passes through them."""
    signal = gaussian(TIME_PS, 0.011)
    curve_time_ps, curve_signal = peak_spline_curve(TIME_PS, signal, n_points=1001)
    k = int(np.argmax(signal))
    assert curve_time_ps[0] == TIME_PS[k - 5] and curve_time_ps[-1] == TIME_PS[k + 5]
    assert np.isclose(np.interp(TIME_PS[k], curve_time_ps, curve_signal), signal[k], atol=1e-4)
    assert curve_signal.max() == pytest.approx(find_peak(TIME_PS, signal).amplitude, abs=1e-6)


def test_peak_is_frozen_and_validated():
    """Peak is a frozen dataclass; wrong shapes and spline widths raise clear errors."""
    peak = find_peak(TIME_PS, gaussian(TIME_PS, 0.0))
    assert isinstance(peak, Peak)
    # setattr instead of peak.time_ps = ...: the write is meant to fail, and this way the IDE does not flag it.
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(peak, "time_ps", 1.0)
    with pytest.raises(ValueError, match="1D arrays of equal length"):
        find_peak(TIME_PS, TIME_PS[:-1])
    with pytest.raises(ValueError, match="spline_half_width must be at least 2"):
        find_peak(TIME_PS, gaussian(TIME_PS, 0.0), spline_half_width=1)


def test_centring_puts_reference_peak_at_zero():
    """After centring, the reference peaks at t = 0 and a sample keeps its delay relative to it."""
    signals = np.array([gaussian(TIME_PS, 0.537), 0.5 * gaussian(TIME_PS, 2.037)])
    centered_ps, peak_times_ps, reference_peak_ps = center_on_reference_peak(TIME_PS, signals)
    assert reference_peak_ps == pytest.approx(0.537, abs=1e-3)
    assert find_peak(centered_ps, signals[0]).time_ps == pytest.approx(0.0, abs=1e-12)
    assert peak_times_ps[0] == 0.0
    assert peak_times_ps[1] == pytest.approx(1.5, abs=1e-3)
    # Only the axis moves; the signals are untouched.
    assert np.allclose(np.diff(centered_ps), 0.05)


def test_centring_on_another_reference():
    """reference_index picks which trace defines t = 0."""
    signals = np.array([gaussian(TIME_PS, 0.5), gaussian(TIME_PS, -1.0)])
    _, peak_times_ps, reference_peak_ps = center_on_reference_peak(TIME_PS, signals, reference_index=1)
    assert reference_peak_ps == pytest.approx(-1.0, abs=1e-3)
    assert peak_times_ps == pytest.approx([1.5, 0.0], abs=1e-3)
    with pytest.raises(ValueError, match="reference_index 2 is out of range for 2 traces"):
        center_on_reference_peak(TIME_PS, signals, reference_index=2)


def test_peak_map_of_a_scan():
    """peak_map gives the peak time and amplitude of every position, in (ny, nx)."""
    delays_ps = np.array([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5]])
    signal = 2.0 * gaussian(TIME_PS, delays_ps[..., None])
    trace_map = TimeTraceMap(x_mm=np.arange(3.0), y_mm=np.arange(2.0), time_ps=TIME_PS, signal=signal)
    peak_time_ps, peak_amplitude = peak_map(trace_map)
    assert peak_time_ps.shape == peak_amplitude.shape == (2, 3)
    assert np.allclose(peak_time_ps, delays_ps, atol=1e-3)
    assert np.allclose(peak_amplitude, 2.0, atol=5e-3)
