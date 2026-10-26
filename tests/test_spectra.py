"""Fourier transform against analytic spectra, and the complex field ratio of simple sample models.

The delayed-copy tests pin the sign convention (THEORY.md, "Time convention"):
a sample that arrives Δt later has the field ratio exp(-i ω Δt), so its phase
decreases with frequency with slope -Δt. Never change these tests to match code.
"""

import numpy as np
import pytest

from cjcthztds import (
    TimeTrace,
    amplitude_spectrum,
    frequency_range_mask,
    compute_complex_field_ratio,
    fourier_transform,
    unwrapped_phase,
)

# 20 fs steps over 40 ps: Nyquist frequency 25 THz, frequency step 25 GHz.
TIME_PS = -20.0 + 0.02 * np.arange(2000)
SIGMA_PS = 0.3


def single_cycle_pulse(time_ps, center_ps=0.0):
    """Derivative of a Gaussian (zero mean, like a real THz pulse), peaks of +-1."""
    x = (time_ps - center_ps) / 0.2
    return -x * np.exp(0.5 * (1.0 - x**2))


def test_fourier_transform_of_a_gaussian():
    """A Gaussian of width sigma, t0 after the first sample: |E_k| = (sigma/dt) sqrt(2 pi) exp(-2 pi^2 sigma^2 f^2),
    phase -2 pi f t0."""
    center_ps = 1.0
    freq_hz, spectrum = fourier_transform(TIME_PS, np.exp(-0.5 * ((TIME_PS - center_ps) / SIGMA_PS) ** 2))
    sigma, dt = SIGMA_PS * 1e-12, 0.02e-12
    # The sum over samples is the integral divided by dt: sqrt(2 pi) x 0.3 ps / 0.02 ps = 37.6 at f = 0, falling
    # to 1/e at f = 1 / (sqrt(2) pi sigma) = 0.75 THz.
    expected = sigma / dt * np.sqrt(2 * np.pi) * np.exp(-2 * np.pi**2 * sigma**2 * freq_hz**2)
    assert np.abs(spectrum[0]) == pytest.approx(SIGMA_PS / 0.02 * np.sqrt(2 * np.pi), rel=1e-10)
    assert np.allclose(np.abs(spectrum), expected, rtol=0, atol=1e-10 * expected[0])
    # Phase: the discrete transform counts time from the first sample (-20 ps), so the pulse centre is
    # t0 = 21 ps later and gives exp(-i 2 pi f t0); checked where the amplitude is well above rounding.
    strong = expected > 1e-6 * expected[0]
    t0 = (center_ps - TIME_PS[0]) * 1e-12
    expected_phase = np.exp(-2j * np.pi * freq_hz[strong] * t0)
    assert np.allclose(spectrum[strong], expected[strong] * expected_phase, atol=1e-10 * expected[0])


def test_frequency_axis():
    """K = N // 2 + 1 frequencies from 0 to the Nyquist frequency 1 / (2 dt), in Hz."""
    freq_hz, spectrum = fourier_transform(TIME_PS, np.zeros((3, TIME_PS.size)))
    assert freq_hz.shape == (1001,) and spectrum.shape == (3, 1001)
    assert freq_hz[1] == pytest.approx(1 / 40e-12) and freq_hz[-1] == pytest.approx(25e12)


def test_fourier_transform_needs_a_uniform_axis():
    """A time axis with unequal steps is refused, with a hint to resample."""
    bad_time_ps = TIME_PS.copy()
    bad_time_ps[10] += 0.005
    with pytest.raises(ValueError, match="equal steps"):
        fourier_transform(bad_time_ps, np.zeros_like(TIME_PS))


def test_field_ratio_of_identical_traces_is_one():
    """A trace divided by itself is 1 at every frequency, to the rounding of one complex division."""
    _, spectrum = fourier_transform(TIME_PS, single_cycle_pulse(TIME_PS))
    ratio = compute_complex_field_ratio(spectrum, spectrum)
    assert np.max(np.abs(ratio - 1.0)) < 1e-15


def test_field_ratio_of_a_scaled_copy():
    """A copy scaled by 0.5 (an attenuating sample) gives |ratio| = 0.5 and phase 0 at every frequency."""
    _, spectra = fourier_transform(TIME_PS, np.array([single_cycle_pulse(TIME_PS), 0.5 * single_cycle_pulse(TIME_PS)]))
    ratio = compute_complex_field_ratio(spectra[1], spectra[0])
    assert np.allclose(ratio, 0.5, rtol=1e-12)


def test_field_ratio_of_a_delayed_copy_is_a_phase_ramp():
    """A copy delayed by 1 ps (50 samples) gives ratio = exp(-i omega dt): |ratio| = 1, phase slope -dt."""
    delay_ps = 1.0
    signals = np.array([single_cycle_pulse(TIME_PS), single_cycle_pulse(TIME_PS, delay_ps)])
    freq_hz, spectra = fourier_transform(TIME_PS, signals)
    ratio = compute_complex_field_ratio(spectra[1], spectra[0])
    omega = 2 * np.pi * freq_hz
    # Where the reference is above rounding noise the ratio is the exact phase factor (shift by whole samples).
    strong = np.abs(spectra[0]) > 1e-8 * np.abs(spectra[0]).max()
    assert np.allclose(ratio[strong], np.exp(-1j * omega[strong] * delay_ps * 1e-12), atol=1e-6)
    # Unwrapped in the 0.2-3 THz band, the phase is the straight line -omega dt (many turns of 2 pi).
    phase = unwrapped_phase(freq_hz, ratio, 0.2e12, 3e12)
    band = frequency_range_mask(freq_hz, 0.2e12, 3e12)
    assert np.allclose(phase[band], -omega[band] * delay_ps * 1e-12, atol=1e-6)
    assert np.all(np.isnan(phase[~band]))


def test_field_ratio_of_a_sub_sample_delay():
    """A delay that is not a whole number of samples (0.737 ps) gives the same phase ramp within the band."""
    delay_ps = 0.737
    signals = np.array([single_cycle_pulse(TIME_PS), single_cycle_pulse(TIME_PS, delay_ps)])
    freq_hz, spectra = fourier_transform(TIME_PS, signals)
    ratio = compute_complex_field_ratio(spectra[1], spectra[0])
    band = frequency_range_mask(freq_hz, 0.1e12, 3e12)
    phase = unwrapped_phase(freq_hz, ratio, 0.1e12, 3e12)
    # The sampled pulse is band-limited well below the 25 THz Nyquist frequency, so the shift is exact to ~1e-9.
    assert np.allclose(np.abs(ratio[band]), 1.0, atol=1e-8)
    assert np.allclose(phase[band], -2 * np.pi * freq_hz[band] * delay_ps * 1e-12, atol=1e-8)


def test_unwrapped_phase_of_several_traces():
    """unwrapped_phase works row by row on (M, K) field ratios."""
    freq_hz = np.linspace(0, 3e12, 301)
    delays = np.array([[0.0], [2e-12], [-1e-12]])
    phase = unwrapped_phase(freq_hz, np.exp(-2j * np.pi * freq_hz * delays), 0.1e12, 2.5e12)
    band = frequency_range_mask(freq_hz, 0.1e12, 2.5e12)
    assert phase.shape == (3, 301)
    assert np.allclose(phase[:, band], -2 * np.pi * freq_hz[band] * delays)


def test_field_ratio_is_nan_where_the_reference_is_zero():
    """Where the reference spectrum is exactly 0 the ratio is NaN, not inf or an error."""
    ratio = compute_complex_field_ratio(np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 2.0]))
    assert ratio[0] == 1.0 and np.isnan(ratio[1]) and ratio[2] == 1.5


def test_frequency_range_mask():
    """frequency_range_mask keeps [freq_min_hz, freq_max_hz], inclusive; None leaves that side open."""
    freq_hz = np.array([0.0, 1e12, 2e12, 3e12])
    assert np.array_equal(frequency_range_mask(freq_hz, 1e12, 2e12), [False, True, True, False])
    assert np.array_equal(frequency_range_mask(freq_hz, None, 1e12), [True, True, False, False])
    assert np.array_equal(frequency_range_mask(freq_hz), [True] * 4)


def test_amplitude_spectrum_of_a_trace():
    """amplitude_spectrum(trace) is |fourier_transform| of the trace as measured."""
    trace = TimeTrace(time_ps=TIME_PS, signal=single_cycle_pulse(TIME_PS))
    freq_hz, amplitude = amplitude_spectrum(trace)
    assert np.array_equal(amplitude, np.abs(fourier_transform(TIME_PS, trace.signal)[1]))
    # Zero-mean pulse: no DC component; the spectrum peaks at 1 / (2 pi * 0.2 ps) = 0.80 THz.
    assert amplitude[0] < 1e-6 * amplitude.max()
    assert freq_hz[np.argmax(amplitude)] == pytest.approx(0.796e12, abs=0.025e12)
