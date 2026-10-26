"""Write the small synthetic instrument files in tests/data/, one per supported format.

Run it from the repository root after changing a writer (the files are kept in
the repository, so the tests do not depend on this script):

    python tests/generate_test_data.py

Every file holds the same synthetic pulse, the derivative of a Gaussian, with
a little noise from a fixed seed, so the output is identical on every run. A
file is named after the format module that reads it (``menlo_dat.dat`` for
``formats/menlo_dat.py``); ``test_every_format_has_a_test_file`` relies on that.

The layouts follow the loaders, which were written against real exports; each
format module's docstring describes its layout. Numbers the tests check:

- every trace: the pulse peaks at +1 at t = -0.2 ps and at -1 at t = +0.2 ps
  (width 0.2 ps), noise 1e-3, sampled every 0.05 ps from -5 ps;
- Menlo: 400 samples; Toptica: 400 samples, signal 2 = 0.5 * signal 1;
- mapper: 3 x 4 grid (y from 10 mm, x from 20 mm, steps 0.5 mm), 200 samples,
  trace (iy, ix) scaled by 1 + 0.1 iy + 0.01 ix;
- Astrella: 401 delay steps from -15 to 5 ps (THz time -5 to 15 ps), with
  stale readbacks, two 0.0 glitches and one far outlier in the delay column.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA = Path(__file__).resolve().parent / "data"
NOISE = 1e-3


def single_cycle_pulse(time_ps: np.ndarray, center_ps: float = 0.0, width_ps: float = 0.2) -> np.ndarray:
    """Derivative of a Gaussian, scaled to peaks of +1 at ``center - width`` and -1 at ``center + width``."""
    x = (time_ps - center_ps) / width_ps
    return -x * np.exp(0.5 * (1.0 - x**2))


def write_menlo_dat(rng: np.random.Generator) -> None:
    """Menlo .dat: 4 '#' header lines, then 'time<TAB>signal' rows, CRLF line endings."""
    time_ps = -5.0 + 0.05 * np.arange(400)
    signal = single_cycle_pulse(time_ps) + NOISE * rng.standard_normal(time_ps.size)
    header = [
        "# Menlo Systems Time Domain Spectrometer",
        "# Program Version 1.4.6 ",
        "#  Start: -5 ps, Average over 100 waveforms. Date: 2026-10-21",
        "# Time [ps] \t THz Signal [arb. u]",
    ]
    rows = [f"{t:.8f}\t{s:.8f}" for t, s in zip(time_ps, signal)]
    (DATA / "menlo_dat.dat").write_bytes(("\r\n".join(header + rows) + "\r\n").encode("utf-8"))


def write_toptica_pulse_csv(rng: np.random.Generator) -> None:
    """Toptica .pulse.csv: one header row, then 'time,signal 1,signal 2' rows."""
    time_ps = -5.0 + 0.05 * np.arange(400)
    signal = single_cycle_pulse(time_ps) + NOISE * rng.standard_normal(time_ps.size)
    rows = ["Time (ps),Signal 1 (nA),Signal 2 (nA)"]
    rows += [f"{t:.3f},{s:.6f},{0.5 * s:.6f}" for t, s in zip(time_ps, signal)]
    (DATA / "toptica_pulse_csv.pulse.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_mapper_tim_csv(rng: np.random.Generator) -> None:
    """Mapper .tim.csv: key row, value row, then one trace row per position, x fastest."""
    ny, nx, nt = 3, 4, 200
    time_ps = -5.0 + 0.05 * np.arange(nt)
    rows = ["No_Y,No_X,dy,dx,No_values,dt/ps,Begin/ps,YO,XO,Scan", f"{ny},{nx},0.5,0.5,{nt},0.05,-5.0,10.0,20.0,S"]
    for iy in range(ny):
        for ix in range(nx):
            trace = (1.0 + 0.1 * iy + 0.01 * ix) * single_cycle_pulse(time_ps) + NOISE * rng.standard_normal(nt)
            rows.append(",".join(f"{value:.6f}" for value in trace))
    (DATA / "mapper_tim_csv.tim.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")


def write_astrella(rng: np.random.Generator) -> None:
    """Astrella (no extension): 'Key: value' header, then clock-time rows with a glitchy delay readback."""
    # Commanded grid: probe delay -15 to 5 ps; the THz time is the negated delay, -5 to 15 ps.
    start_ps, step_ps, n_steps = -15.0, 0.05, 401
    delay_ps = start_ps + step_ps * np.arange(n_steps)
    x = single_cycle_pulse(-delay_ps) + NOISE * rng.standard_normal(n_steps)
    y = 0.05 * single_cycle_pulse(-delay_ps) + NOISE * rng.standard_normal(n_steps)

    # The stage readback: about a fifth of the rows repeat the previous position (stale register),
    # two rows read exactly 0.0 and one reads far outside the scan.
    readback_ps = delay_ps.copy()
    stale = rng.random(n_steps) < 0.2
    stale[0] = False
    readback_ps[stale] = delay_ps[np.flatnonzero(stale) - 1]
    readback_ps[[57, 233]] = 0.0
    readback_ps[150] = 4321.0

    lines = [
        f"ProbeDelay: {start_ps}, {start_ps + step_ps * (n_steps - 1)}, {n_steps}, {step_ps}",
        "PumpDelay: 0.0, 0.0, 1, 0.0",
        "Comment: synthetic test file",
    ]
    for row in range(n_steps):
        seconds = 11.0 + 0.25 * row
        clock = f"14:{2 + int(seconds // 60):02d}:{seconds % 60:06.3f}"
        lines.append(f"{clock}\t{readback_ps[row]:.4f}\t{x[row]:.6e}\t{y[row]:.6e}\t{1050 + row % 7}\t0\t0")
    (DATA / "astrella").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Write every test file, each with its own seeded generator so changing one leaves the others unchanged."""
    DATA.mkdir(exist_ok=True)
    for seed, writer in enumerate((write_menlo_dat, write_toptica_pulse_csv, write_mapper_tim_csv, write_astrella)):
        writer(np.random.default_rng(seed))
        print(f"wrote {writer.__name__.removeprefix('write_')}")


if __name__ == "__main__":
    main()
