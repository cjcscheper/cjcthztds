"""Tukey windows and zero padding."""

import numpy as np
import pytest

from cjcthztds import fourier_transform, trace_window, tukey_window, zero_pad

# 0.01 ps steps from -2 to 2 ps, so window edges at whole hundredths fall on samples.
TIME_PS = np.round(np.arange(-200, 201) * 0.01, 10)


def test_tukey_window_is_one_in_the_flat_part_and_zero_outside():
    """A window from -1 to 1 ps with taper 0.5: flat from -0.5 to 0.5 ps, cosine edges, 0 outside."""
    window = tukey_window(TIME_PS, -1.0, 1.0, taper=0.5)
    flat = np.abs(TIME_PS) <= 0.5
    outside = np.abs(TIME_PS) > 1.0
    assert np.all(window[flat] == 1.0)
    assert np.all(window[outside] == 0.0)
    # Halfway up the rising edge (x = taper / 4 = 0.125, i.e. t = -0.75 ps) the cosine is at 0.5.
    assert window[np.isclose(TIME_PS, -0.75)][0] == pytest.approx(0.5)
    assert np.allclose(window, window[::-1])


def test_tukey_window_limits():
    """taper 0 is a rectangle over [start, stop]; taper 1 is a Hann window (0.5 at a quarter, 1 at the centre)."""
    rectangle = tukey_window(TIME_PS, -1.0, 1.0, taper=0.0)
    assert np.array_equal(rectangle, (np.abs(TIME_PS) <= 1.0).astype(float))
    hann = tukey_window(TIME_PS, -1.0, 1.0, taper=1.0)
    assert hann[np.isclose(TIME_PS, 0.0)][0] == pytest.approx(1.0)
    assert hann[np.isclose(TIME_PS, -0.5)][0] == pytest.approx(0.5)
    assert np.allclose(hann[np.abs(TIME_PS) <= 1.0], 0.5 * (1 + np.cos(np.pi * TIME_PS[np.abs(TIME_PS) <= 1.0])))


def test_tukey_window_validation():
    """A reversed window or a taper outside [0, 1] raises a clear ValueError."""
    with pytest.raises(ValueError, match="start_ps < stop_ps"):
        tukey_window(TIME_PS, 1.0, -1.0)
    with pytest.raises(ValueError, match="taper must be between 0 and 1"):
        tukey_window(TIME_PS, -1.0, 1.0, taper=1.5)


def test_trace_window_centres_on_the_peak():
    """With center_on_peak the window keeps the region's width but is centred on the trace's peak."""
    signal = np.exp(-0.5 * ((TIME_PS - 0.3) / 0.05) ** 2)
    window = trace_window(TIME_PS, signal, [(-0.5, 0.5)], taper=0.0)
    assert np.array_equal(window, ((TIME_PS >= -0.2 - 1e-9) & (TIME_PS <= 0.8 + 1e-9)).astype(float))
    fixed = trace_window(TIME_PS, signal, [(0.5, -0.5)], taper=0.0, center_on_peak=False)
    assert np.array_equal(fixed, (np.abs(TIME_PS) <= 0.5).astype(float))


def test_trace_window_combines_regions():
    """Two regions give two windows; where they overlap the larger value wins."""
    window = trace_window(TIME_PS, np.zeros_like(TIME_PS), [(-1.5, -0.5), (0.0, 1.0)], taper=0.0, center_on_peak=False)
    inside = ((TIME_PS >= -1.5) & (TIME_PS <= -0.5)) | ((TIME_PS >= 0.0) & (TIME_PS <= 1.0))
    assert np.array_equal(window, inside.astype(float))
    with pytest.raises(ValueError, match="no window regions were given"):
        trace_window(TIME_PS, np.zeros_like(TIME_PS), [])


def test_zero_pad_adds_zeros_and_continues_the_axis():
    """padding_ps = 1 ps at 0.01 ps steps adds 50 zeros on each side and extends the time axis evenly."""
    signals = np.ones((2, TIME_PS.size))
    padded_time_ps, padded = zero_pad(TIME_PS, signals, 1.0)
    assert padded.shape == (2, TIME_PS.size + 100)
    assert np.all(padded[:, :50] == 0) and np.all(padded[:, -50:] == 0) and np.all(padded[:, 50:-50] == 1)
    assert padded_time_ps[0] == pytest.approx(-2.5) and padded_time_ps[-1] == pytest.approx(2.5)
    assert np.allclose(np.diff(padded_time_ps), 0.01)


def test_zero_pad_nothing_and_validation():
    """padding_ps = 0 returns the traces unchanged; negative padding raises."""
    time_ps, signal = zero_pad(TIME_PS, np.sin(TIME_PS), 0.0)
    assert np.array_equal(time_ps, TIME_PS) and np.array_equal(signal, np.sin(TIME_PS))
    with pytest.raises(ValueError, match="padding_ps must be 0 or positive"):
        zero_pad(TIME_PS, np.sin(TIME_PS), -1.0)


def test_zero_padding_leaves_the_spectrum_unchanged():
    """Doubling the length with zeros adds frequencies in between; at the old frequencies the spectrum is the same."""
    # 400 samples; padding of 4 ps = 2 x 200 samples doubles the length, so every second new frequency is an old one.
    time_ps = -2.0 + 0.01 * np.arange(400)
    signal = np.exp(-0.5 * (time_ps / 0.2) ** 2) * np.sin(2 * np.pi * time_ps / 0.5)
    freq_hz, spectrum = fourier_transform(time_ps, signal)
    padded_freq_hz, padded_spectrum = fourier_transform(*zero_pad(time_ps, signal, 4.0))
    assert padded_freq_hz.size == 2 * freq_hz.size - 1
    assert np.allclose(padded_freq_hz[::2], freq_hz)
    assert np.allclose(padded_spectrum[::2], spectrum, rtol=0, atol=1e-12 * np.abs(spectrum).max())
