"""Measured time-domain data: one trace (TimeTrace) or a scanned map of traces (TimeTraceMap).

These hold only what the instrument recorded (time axis, signal, where it came
from, header values). Everything derived from them, such as spectra and field
ratios, lives in :class:`cjcthztds.Analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TimeTrace:
    """One measured THz time trace.

    Attributes
    ----------
    time_ps : ndarray, shape (N,)
        Time axis in ps, increasing.
    signal : ndarray, shape (N,)
        Measured signal at each time, in the instrument's units (arbitrary units
        for most detectors).
    path : Path or None
        File the trace was loaded from; None for traces made in a script.
    metadata : dict
        Values read from the file besides the trace itself, e.g. the Menlo
        header lines, the Astrella scan grid, or the position of a mapper trace.
    """

    time_ps: np.ndarray
    signal: np.ndarray
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TimeTraceMap:
    """A raster scan: one time trace at every (x, y) position of a 2D grid.

    Attributes
    ----------
    x_mm : ndarray, shape (nx,)
        x positions of the grid columns in mm.
    y_mm : ndarray, shape (ny,)
        y positions of the grid rows in mm.
    time_ps : ndarray, shape (nt,)
        Time axis in ps, shared by every position.
    signal : ndarray, shape (ny, nx, nt)
        ``signal[iy, ix]`` is the trace measured at ``(x_mm[ix], y_mm[iy])``.
    path : Path or None
        File the map was loaded from; None for maps made in a script.
    metadata : dict
        Header values of the file (grid size, steps, start positions, ...).
    """

    x_mm: np.ndarray
    y_mm: np.ndarray
    time_ps: np.ndarray
    signal: np.ndarray
    path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def trace_at_position(trace_map: TimeTraceMap, position: tuple[int, int]) -> TimeTrace:
    """Take the trace at grid position ``position = (iy, ix)`` out of a map.

    The returned trace keeps the map's path and metadata and adds the grid
    index and the position in mm, so it can still be told apart from the other
    traces of the same file.
    """
    # Check the index before using it, so a wrong position names the grid size instead of a numpy IndexError.
    iy, ix = (int(index) for index in position)
    ny, nx = trace_map.signal.shape[:2]
    if not (0 <= iy < ny and 0 <= ix < nx):
        raise ValueError(f"position (iy, ix) = {(iy, ix)} is outside the map grid of shape (ny, nx) = {(ny, nx)}")

    # Record where the trace came from next to the map's own header values.
    metadata = dict(trace_map.metadata)
    metadata.update({
        "position": (iy, ix),
        "x_mm": float(trace_map.x_mm[ix]),
        "y_mm": float(trace_map.y_mm[iy]),
    })
    return TimeTrace(
        time_ps=np.asarray(trace_map.time_ps, dtype=np.float64),
        signal=np.asarray(trace_map.signal[iy, ix], dtype=np.float64),
        path=trace_map.path,
        metadata=metadata,
    )
