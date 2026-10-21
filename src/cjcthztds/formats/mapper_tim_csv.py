"""2D THz mapper export (``.tim.csv``): one time trace per position of a raster scan.

Layout (comma-separated text)::

    No_Y,No_X,dy,dx,No_values,dt/ps,Begin/ps,YO,XO,Scan
    3,4,0.5,0.5,200,0.05,-5.0,10.0,20.0,S
    <trace at (iy=0, ix=0): No_values numbers>
    <trace at (iy=0, ix=1)>
    ...
    <trace at (iy=No_Y-1, ix=No_X-1)>

Row 1 holds the metadata keys, row 2 their values, then ``No_Y * No_X`` trace
rows, x running fastest (row index ``iy * No_X + ix``). Positions are
``YO + dy * iy`` and ``XO + dx * ix`` in mm, times ``Begin/ps + dt/ps * k``
in ps. Keys are matched ignoring case, spaces and underscores. Textual values
(such as ``Scan``) are kept in the metadata but not used.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from cjcthztds.traces import TimeTraceMap

DESCRIPTION = "THz mapper"
FILE_PATTERNS = ("*.tim.csv",)
IS_MAP = True


def detect(path: Path) -> bool:
    """A mapper export is recognised by its double extension ``.tim.csv``."""
    return path.name.lower().endswith(".tim.csv")


def _normalized_key(key: str) -> str:
    """Write a header key without case, spaces or underscores, so ``No_Y``, ``no y`` and ``NoY`` match."""
    return key.strip().lower().replace(" ", "").replace("_", "")


def load(path: Path) -> TimeTraceMap:
    """Load a ``.tim.csv`` mapper file into a :class:`TimeTraceMap`."""
    # Read every non-empty row; the file is small enough (one row per position) to hold in memory.
    with path.open("r", newline="", encoding="utf-8", errors="replace") as file:
        rows = [row for row in csv.reader(file) if row]
    if len(rows) < 3:
        raise ValueError(f"{path.name}: expected a key row, a value row and trace rows, got {len(rows)} rows")

    # Header: keep every key-value pair as written, numbers as floats and text as strings.
    metadata: dict[str, float | str] = {}
    numeric: dict[str, float] = {}
    for key, value in zip(rows[0], rows[1]):
        if not key.strip() or not value.strip():
            continue
        try:
            metadata[key.strip()] = numeric[_normalized_key(key)] = float(value)
        except ValueError:
            metadata[key.strip()] = value.strip()

    # The grid and the time axis need these keys; name the missing ones as they appear in the file.
    required = {"noy": "No_Y", "nox": "No_X", "dy": "dy", "dx": "dx", "novalues": "No_values",
                "dt/ps": "dt/ps", "begin/ps": "Begin/ps", "yo": "YO", "xo": "XO"}
    missing = [name for key, name in required.items() if key not in numeric]
    if missing:
        raise ValueError(f"{path.name}: header is missing the numeric fields {missing}")
    ny, nx, nt = (int(round(numeric[key])) for key in ("noy", "nox", "novalues"))
    if ny <= 0 or nx <= 0 or nt <= 1:
        raise ValueError(f"{path.name}: No_Y, No_X must be positive and No_values at least 2, got {ny}, {nx}, {nt}")

    # One trace row per position, x fastest; rows may carry extra trailing values, which are cut off.
    trace_rows = rows[2:]
    if len(trace_rows) < ny * nx:
        raise ValueError(f"{path.name}: {len(trace_rows)} trace rows for a {ny} x {nx} grid (need {ny * nx})")
    signal = np.empty((ny, nx, nt), dtype=np.float64)
    for flat_index in range(ny * nx):
        values = np.asarray([value for value in trace_rows[flat_index] if value.strip()], dtype=np.float64)
        if values.size < nt:
            raise ValueError(f"{path.name}: trace row {flat_index + 1} has {values.size} values, expected {nt}")
        signal[flat_index // nx, flat_index % nx] = values[:nt]

    # Positions and times follow from the start values and steps in the header.
    return TimeTraceMap(
        x_mm=numeric["xo"] + numeric["dx"] * np.arange(nx, dtype=np.float64),
        y_mm=numeric["yo"] + numeric["dy"] * np.arange(ny, dtype=np.float64),
        time_ps=numeric["begin/ps"] + numeric["dt/ps"] * np.arange(nt, dtype=np.float64),
        signal=signal,
        path=path,
        metadata=metadata,
    )
