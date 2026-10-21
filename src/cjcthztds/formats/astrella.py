"""Astrella far-field THz-TDS export (no file extension).

Layout (text; header lines ``Key: values``, then whitespace-separated data rows)::

    ProbeDelay: -15.0, 5.0, 401, 0.05
    ...other "Key: value" header lines...
    14:02:11.250	-15.0000	1.2e-06	3.0e-07	1052	0	0
    14:02:11.500	-15.0000	1.3e-06	2.9e-07	1049	0	0
    ...

``ProbeDelay: start, end, number of steps, step`` is the commanded delay grid
in ps. Each data row starts with the clock time (``HH:MM:SS.fff``), then:

- column 1: probe delay in ps as read back from the stage;
- column 2: lock-in X (in phase): the THz signal;
- column 3: lock-in Y (quadrature);
- column 4: intensity monitor counter, then unused zeros.

The file has no extension, so it is recognised by its content: the first
non-empty line starts with ``ProbeDelay:``.

**The delay column cannot be trusted.** It is a readback of the stage
position register: in a typical scan about a fifth of the rows repeat the
previous position (a stale register), and a few rows hold garbage (exactly
0 ps, or values far outside the scan). The commanded grid in the header is
exact, and the scan writes one row per commanded step, so by default the
delay axis is rebuilt from the grid by row number; the readback only serves
to find the offset between row number and grid step. If the number of rows
does not match the number of steps, the loader falls back to filtering the
readback (see ``delay_axis``).

**Time direction.** A longer probe delay samples an earlier part of the THz
pulse, so the THz time is the negated delay. The rows are reversed so the
returned time axis increases.
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from typing import Literal

import numpy as np

from cjcthztds.traces import TimeTrace

DESCRIPTION = "Astrella far-field TDS (no extension)"
# No extension to filter on; the file dialogs show these files under "All files".
FILE_PATTERNS: tuple[str, ...] = ()
IS_MAP = False

# A data row starts with the clock time, e.g. 14:02:11.250.
_CLOCK_TIME = re.compile(r"^\d{1,2}:\d{2}:\d{2}(?:\.\d+)?$")


def detect(path: Path) -> bool:
    """An Astrella export is recognised by a first non-empty line starting with ``ProbeDelay:``."""
    # Read only the start of the file: this runs on every file a user picks, including large binary ones.
    try:
        with path.open("r", encoding="utf-8", errors="replace") as file:
            for line in file.read(4096).splitlines():
                if line.strip():
                    return line.strip().lower().startswith("probedelay:")
    except (OSError, UnicodeError):
        return False
    return False


def _delay_axis_from_grid(
    readback_ps: np.ndarray, start_ps: float, step_ps: float, n_steps: int,
) -> np.ndarray | None:
    """Rebuild the delay axis from the commanded grid by row number, or return None.

    Row ``k`` sits at grid step ``k + offset``; the offset is the most common
    difference between the grid step a readback lands on and the row number,
    counting only readbacks that land on a grid node (stale and garbage rows
    are off by whole steps or far off the grid, so they do not win the vote).

    Returns None when the row count is not one row per step, or when too few
    readbacks agree, so the caller can fall back to the readback itself.
    """
    # One row per commanded step is the precondition for indexing by row number (allow 2 % slack).
    n_rows = readback_ps.size
    if n_steps <= 0 or abs(n_rows - n_steps) > max(2, int(0.02 * n_steps)):
        return None

    # Only readbacks on a grid node can vote; the rest are garbage.
    grid_index = np.round((readback_ps - start_ps) / step_ps)
    on_grid = np.abs(readback_ps - (start_ps + step_ps * grid_index)) <= 0.25 * abs(step_ps)
    if on_grid.sum() < 0.5 * n_rows:
        return None

    # The healthy rows share one offset between grid step and row number; stale rows are off by one or more.
    offsets = (grid_index[on_grid] - np.arange(n_rows)[on_grid]).astype(np.int64)
    unique_offsets, counts = np.unique(offsets, return_counts=True)
    if counts.max() < 0.5 * on_grid.sum():
        return None
    offset = int(unique_offsets[np.argmax(counts)])

    return start_ps + step_ps * (np.arange(n_rows, dtype=np.float64) + offset)


def _delay_axis_from_readback(
    readback_ps: np.ndarray, signal: np.ndarray, start_ps: float, step_ps: float,
    n_steps: int, max_step_multiplier: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Reject glitched readbacks, then put the remaining rows on the uniform commanded grid.

    Glitches are found by detrending: the stage advances by a nearly constant
    amount per row, so ``readback - slope * row`` is the same for every healthy
    row and far off for a glitch. The slope is the median step between rows,
    which stale rows (step 0) and rare glitches do not move. The test is global, so one bad row cannot drag
    the acceptance window with it (as a filter on the distance to the previous
    accepted row would: one 0 ps glitch mid-scan would reject every later row).
    Rows on the same grid node are averaged, and nodes without any row are
    filled by linear interpolation, with a warning.
    """
    # Keep rows close to the scan trend; max_step_multiplier steps is the tolerance.
    n_rows = readback_ps.size
    slope_ps = float(np.median(np.diff(readback_ps)))
    detrended_ps = readback_ps - slope_ps * np.arange(n_rows)
    keep = np.abs(detrended_ps - np.median(detrended_ps)) <= max_step_multiplier * abs(step_ps)
    if keep.sum() < 2:
        raise ValueError("every Astrella row was rejected as a stage glitch")

    # Snap the kept rows to the nearest grid node and average repeated nodes.
    node = np.clip(np.round((readback_ps[keep] - start_ps) / step_ps).astype(np.int64), 0, n_steps - 1)
    unique_node, inverse = np.unique(node, return_inverse=True)
    node_signal = np.bincount(inverse, weights=signal[keep]) / np.bincount(inverse)

    # Fill nodes the stage never reported, so the FFT gets a gap-free uniform axis.
    all_nodes = np.arange(unique_node[0], unique_node[-1] + 1)
    if all_nodes.size > unique_node.size:
        warnings.warn(
            f"Astrella readback covered {unique_node.size} of {all_nodes.size} delay steps; "
            f"{1.0 - unique_node.size / all_nodes.size:.0%} of the trace was interpolated.",
            RuntimeWarning, stacklevel=4,
        )
    return start_ps + step_ps * all_nodes, np.interp(all_nodes, unique_node, node_signal)


def load(
    path: Path,
    *,
    signal_channel: Literal["X", "Y", "R"] = "X",
    delay_axis: Literal["grid", "readback"] = "grid",
    max_step_multiplier: float = 5.0,
) -> TimeTrace:
    """Load an Astrella export.

    Parameters
    ----------
    signal_channel : {"X", "Y", "R"}
        Lock-in output to return: ``"X"`` (in phase, default; the THz signal),
        ``"Y"`` (quadrature, for checking the lock-in phase) or ``"R"``
        (magnitude sqrt(X² + Y²), always positive, so the sign is lost).
    delay_axis : {"grid", "readback"}
        ``"grid"`` (default): the commanded grid indexed by row number; every
        row is kept and nothing is interpolated. Falls back to ``"readback"``
        with a warning when the row count does not match the step count.
        ``"readback"``: the stage readback with glitches rejected, repeats
        averaged and missing steps interpolated.
    max_step_multiplier : float
        ``"readback"`` only: how far a row may lie from the scan trend, in grid
        steps, before it is rejected as a glitch (default 5).
    """
    # Check the choices first, so a typo does not silently give the default.
    if signal_channel not in ("X", "Y", "R"):
        raise ValueError(f"signal_channel must be 'X', 'Y' or 'R', got {signal_channel!r}")
    if delay_axis not in ("grid", "readback"):
        raise ValueError(f"delay_axis must be 'grid' or 'readback', got {delay_axis!r}")

    # One pass: "Key: value" header lines into metadata, clock-time rows into data.
    metadata: dict[str, str | float | int] = {}
    rows: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="replace") as file:
        for line in file:
            parts = line.split()
            if len(parts) >= 4 and _CLOCK_TIME.match(parts[0]):
                try:
                    rows.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    continue
            elif ":" in line and not rows:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip()
    if len(rows) < 2:
        raise ValueError(f"{path.name}: fewer than 2 data rows (rows start with a clock time HH:MM:SS)")
    readback_ps, x, y = np.asarray(rows, dtype=np.float64).T

    # The commanded grid: ProbeDelay: start, end, number of steps, step (ps).
    try:
        start_ps, end_ps, n_steps_float, step_ps = (float(value) for value in str(metadata["ProbeDelay"]).split(","))
    except (KeyError, ValueError) as error:
        raise ValueError(f"{path.name}: no valid 'ProbeDelay: start, end, steps, step' header line") from error
    n_steps = int(round(n_steps_float))
    if step_ps == 0.0:
        raise ValueError(f"{path.name}: ProbeDelay step is 0")
    metadata.update({"probe_delay_start_ps": start_ps, "probe_delay_end_ps": end_ps,
                     "probe_delay_steps": n_steps, "probe_delay_step_ps": step_ps, "signal_channel": signal_channel})

    # The requested lock-in output.
    signal = {"X": x, "Y": y, "R": np.hypot(x, y)}[signal_channel]

    # Delay axis from the grid where possible, else from the filtered readback.
    delay_ps = _delay_axis_from_grid(readback_ps, start_ps, step_ps, n_steps) if delay_axis == "grid" else None
    if delay_axis == "grid" and delay_ps is None:
        warnings.warn(
            f"{path.name}: {readback_ps.size} rows for {n_steps} commanded steps, so the delay axis "
            "cannot be indexed by row number; using the filtered stage readback instead.",
            RuntimeWarning, stacklevel=3,
        )
    metadata["delay_axis"] = "grid" if delay_ps is not None else "readback"
    if delay_ps is None:
        delay_ps, signal = _delay_axis_from_readback(
            readback_ps, signal, start_ps, step_ps, n_steps, max_step_multiplier,
        )

    # THz time is the negated probe delay; reverse when needed so the time axis increases.
    time_ps = -delay_ps
    if time_ps[0] > time_ps[-1]:
        time_ps, signal = time_ps[::-1], signal[::-1]
    return TimeTrace(time_ps=time_ps.copy(), signal=signal.copy(), path=path, metadata=metadata)
