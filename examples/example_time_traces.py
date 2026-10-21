"""Time traces: the reference pulse and the pulse after a silicon wafer, delayed, weaker, with an echo."""

from pathlib import Path

from cjcplot import plot_1d, style_colors
from synthetic_data import synthetic_data_reference_and_sample

reference, sample = synthetic_data_reference_and_sample()

# The sample pulse arrives 2.4 ps later (the extra optical path in silicon) at 0.70 of the height,
# and its first echo 6.8 ps after that (a round trip through the wafer).
plot_1d(
    [reference.time_ps, sample.time_ps],
    [reference.signal, sample.signal],
    colors=["black", style_colors()[0]],
    legend_labels=["reference", "sample"],
    xlabel="Time [ps]", ylabel="Signal [arb. u]",
    xlim=(-5.0, 15.0), ylim=(-1.2, 1.2),
    xticks_pos=[-5.0, 0.0, 5.0, 10.0, 15.0],
    yticks_pos=[-1.2, -0.8, -0.4, 0.0, 0.4, 0.8, 1.2],
    legend_loc="upper right",
    fig_aspect_ratio=(0.5, 0.5),
    save_path=str(Path(__file__).with_suffix(".png")),
    fig_show=False,
)
