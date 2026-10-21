"""Mapper scan: the peak amplitude at every position of a synthetic .tim.csv reflection scan of a 100 mm wafer."""

import tempfile
from pathlib import Path

import numpy as np
from cjcplot import plot_2d
from synthetic_data import single_cycle_pulse

from cjcthztds import load_time_trace_map, peak_map

# A 49 x 49 reflection scan over 120 x 120 mm in 2.5 mm steps, 15 ps per trace in 40 fs steps, over a 100 mm
# silicon wafer with a 32.5 mm primary flat (bottom) and an 18 mm secondary flat 90 degrees from it (left), the
# SEMI flats of a p-type (100) wafer. Off the wafer nothing is reflected, so the trace is noise only; on the
# wafer the pulse comes back with a peak of about 0.70 (as the sample in example_time_traces.py), varying by
# 2 % from position to position (Gaussian), as surface and thickness variations would make it.
x_mm = -60.0 + 2.5 * np.arange(49)
y_mm = -60.0 + 2.5 * np.arange(49)
time_ps = -4.0 + 0.04 * np.arange(375)
xx, yy = np.meshgrid(x_mm, y_mm)
radius_mm = 50.0
primary_flat_y_mm = -np.sqrt(radius_mm**2 - (32.5 / 2) ** 2)
secondary_flat_x_mm = -np.sqrt(radius_mm**2 - (18.0 / 2) ** 2)
on_wafer = (np.hypot(xx, yy) < radius_mm) & (yy > primary_flat_y_mm) & (xx > secondary_flat_x_mm)
rng = np.random.default_rng(0)
reflection = np.where(on_wafer, 0.70 * (1.0 + 0.02 * rng.standard_normal(xx.shape)), 0.0)
signal = reflection[..., None] * single_cycle_pulse(time_ps)
signal += 2e-3 * rng.standard_normal(signal.shape)

# Write it in the mapper's .tim.csv layout (key row, value row, one trace per row, x fastest) and load it
# back with the package's loader, as for a real scan.
with tempfile.TemporaryDirectory() as folder:
    path = Path(folder) / "wafer_scan.tim.csv"
    rows = [
        "No_Y,No_X,dy,dx,No_values,dt/ps,Begin/ps,YO,XO,Scan",
        f"{y_mm.size},{x_mm.size},2.5,2.5,{time_ps.size},0.04,-4.0,{y_mm[0]},{x_mm[0]},S",
    ]
    rows += [",".join(f"{value:.4f}" for value in trace) for trace in signal.reshape(-1, time_ps.size)]
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    trace_map = load_time_trace_map(path)

# The peak amplitude (maximum of a spline through the samples around the largest one) shows the wafer with its flats and
# the 2 % variation across it; off the wafer only the noise is left (about 0.005).
_, peak_amplitude = peak_map(trace_map)
plot_2d(
    trace_map.x_mm, trace_map.y_mm, np.abs(peak_amplitude),
    xlabel="x [mm]", ylabel="y [mm]", cbar_label="Peak amplitude [arb. u]",
    clim=(0.0, 0.8),
    xlim=(-60.0, 60.0), ylim=(-60.0, 60.0),
    xticks_pos=[-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0],
    yticks_pos=[-60.0, -40.0, -20.0, 0.0, 20.0, 40.0, 60.0],
    aspect="equal",
    fig_aspect_ratio=(0.6, 0.5),
    save_path=str(Path(__file__).with_suffix(".png")),
    fig_show=False,
)
