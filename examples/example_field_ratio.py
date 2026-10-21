"""Field ratio of the silicon wafer: |t(f)| and the unwrapped phase, within the usable band, without a window."""

from pathlib import Path

import numpy as np
from cjcplot import plot_1d, style_colors
from synthetic_data import DELAY_PS, TRANSMISSION, synthetic_data_reference_and_sample

from cjcthztds import analyze, frequency_range_mask, unwrapped_phase

output_stem = Path(__file__).with_suffix("")
FREQ_MIN_HZ, FREQ_MAX_HZ = 0.1e12, 3.0e12

# The whole traces, no window: the echo stays in (example_windowing.py shows what a window changes).
# 160 ps of zero padding samples the spectra 5 times more finely (5 GHz instead of 25 GHz), so the
# fringes are drawn smoothly; it interpolates and adds no information.
analysis = analyze(
    synthetic_data_reference_and_sample(), padding_ps=160.0, freq_min_hz=FREQ_MIN_HZ, freq_max_hz=FREQ_MAX_HZ,
)
band = frequency_range_mask(analysis.freq_hz, FREQ_MIN_HZ, FREQ_MAX_HZ)
freq_thz = analysis.freq_hz[band] * 1e-12
field_ratio = analysis.field_ratio[1][band]

# |t| oscillates around the two interface transmissions, 4n/(n+1)^2 = 0.70 for n = 3.42: the echo,
# 6.84 ps after the direct pass, interferes with it and gives fringes 146 GHz apart, between 0.49 and 0.91.
plot_1d(
    freq_thz, np.abs(field_ratio),
    colors=[style_colors()[0]],
    hlines_pos=[TRANSMISSION], hlines_colors=["black"], hlines_styles=["--"],
    hlines_labels=[r"$4n/(n+1)^2$"],
    legend_labels=["measured"],
    xlabel="Frequency [THz]", ylabel=r"$|t(f)|$",
    xlim=(0.0, 3.0), ylim=(0.2, 1.0),
    legend_loc="lower right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_amplitude.png",
    fig_show=False,
)

# The phase falls linearly, -omega * 2.42 ps, the delay (n - 1) d / c: numpy's exp(-i omega t) convention
# (THEORY.md). Its slope gives the refractive index. The expected line is drawn dashed on top.
phase = unwrapped_phase(analysis.freq_hz, analysis.field_ratio[1], FREQ_MIN_HZ, FREQ_MAX_HZ)[band]
plot_1d(
    [freq_thz, freq_thz], [phase, -2 * np.pi * freq_thz * DELAY_PS],
    colors=[style_colors()[0], "black"], line_styles=["-", "--"],
    legend_labels=["measured", r"$-\omega (n-1) d / c$"],
    xlabel="Frequency [THz]", ylabel=r"Phase of $t(f)$ [rad]",
    xlim=(0.0, 3.0), ylim=(-50.0, 0.0),
    legend_loc="upper right",
    first_on_top=False,
    fig_aspect_ratio=(0.5, 0.5),
    save_path=f"{output_stem}_phase.png",
    fig_show=False,
)
