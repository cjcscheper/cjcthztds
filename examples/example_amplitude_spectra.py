"""Amplitude spectra |E(f)| on a log scale: the bandwidth of the pulse and the noise floor it sinks into."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors
from synthetic_data import synthetic_data_reference_and_sample

from cjcthztds import fourier_transform, zero_pad

# 160 ps of zeros around the 40 ps traces samples the spectra 5 times more finely (5 GHz instead of 25 GHz),
# so the echo ripple is drawn smoothly; it interpolates and adds no information. Without padding,
# amplitude_spectrum(trace) gives the same curve at the coarser spacing.
reference, sample = synthetic_data_reference_and_sample()
freq_hz, reference_spectrum = fourier_transform(*zero_pad(reference.time_ps, reference.signal, 160.0))
_, sample_spectrum = fourier_transform(*zero_pad(sample.time_ps, sample.signal, 160.0))
reference_amplitude, sample_amplitude = np.abs(reference_spectrum), np.abs(sample_spectrum)

# The pulse spectrum peaks near 1 THz and falls off as a Gaussian; above about 4 THz only the flat noise
# floor is left, so a field ratio is meaningful only below that. The sample's ripple is the echo interfering.
plot_1d(
    [freq_hz * 1e-12, freq_hz * 1e-12],
    [reference_amplitude / reference_amplitude.max(), sample_amplitude / reference_amplitude.max()],
    colors=["black", style_colors()[0]],
    legend_labels=["reference", "sample"],
    xlabel="Frequency [THz]", ylabel=r"$|E(f)|$ [arb. u]",
    xlim=(0.0, 6.0), ylim=(1e-3, 2.0),
    logy=True,
    legend_loc="upper right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=str(Path(__file__).with_suffix(".png")),
    fig_show=False,
)
