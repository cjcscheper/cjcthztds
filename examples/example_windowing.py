"""Why windowing: the echo makes the field ratio oscillate; a Tukey window around the direct pulse removes it."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors
from synthetic_data import TRANSMISSION, synthetic_data_reference_and_sample

from cjcthztds import analyze, frequency_range_mask

output_stem = Path(__file__).with_suffix("")
traces = synthetic_data_reference_and_sample()

# The same analysis with and without a window from -3 to 3 ps (taper 0.5), moved onto each trace's own peak.
# 160 ps of zero padding samples the spectra 5 times more finely (5 GHz instead of 25 GHz), only so the
# fringes below are drawn smoothly; it adds no information.
windowed = analyze(traces, window_regions_ps=[(-3.0, 3.0)], taper=0.5, padding_ps=160.0)
whole = analyze(traces, padding_ps=160.0)

# The window (scaled to the pulse height) over the sample trace: it keeps the direct pass and drops the echo.
sample_index = 1
plot_1d(
    [windowed.time_ps, windowed.time_ps],
    [windowed.signal[sample_index], TRANSMISSION * windowed.window[sample_index]],
    colors=[style_colors()[0], style_colors()[2]], line_styles=["-", "--"],
    legend_labels=["", "window"],
    xlabel="Time [ps]", ylabel="Signal [arb. u]",
    xlim=(-5.0, 15.0), ylim=(-1.0, 1.0),
    xticks_pos=[-5.0, 0.0, 5.0, 10.0, 15.0],
    legend_loc="lower right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_trace.png",
    fig_show=False,
)

# Without the window the echo interferes with the direct pass: fringes 1 / 6.84 ps = 146 GHz apart, swinging
# between 0.70 x (1 - 0.30) = 0.49 and 0.70 x (1 + 0.30) = 0.91. With the window only the flat 0.70 is left.
band = frequency_range_mask(whole.freq_hz, 0.1e12, 3.0e12)
freq_thz = whole.freq_hz[band] * 1e-12
plot_1d(
    [freq_thz, freq_thz],
    [np.abs(windowed.field_ratio[sample_index][band]), np.abs(whole.field_ratio[sample_index][band])],
    colors=[style_colors()[0], style_colors()[2]],
    legend_labels=["windowed", "raw"],
    xlabel="Frequency [THz]", ylabel=r"$|t(f)|$",
    xlim=(0.0, 3.0), ylim=(0.2, 1.0),
    legend_loc="lower right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_field_ratio.png",
    fig_show=False,
)
