"""Toptica TeraFlash export (``.pulse.csv``), e.g. the TeraFlash in Lab C FLX 8.100.

Layout (comma-separated text)::

    Time (ps),Signal 1 (nA),Signal 2 (nA)
    -5.000,0.0012,0.0010
    -4.950,0.0013,0.0011
    ...

One header row, then one row per time step: time in ps and two signal columns
(the two detection channels). Some exports end every row with empty columns
(``...,0.0010,,,``); those are ignored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cjcthztds.traces import TimeTrace

DESCRIPTION = "Toptica TeraFlash"
FILE_PATTERNS = ("*.pulse.csv",)
IS_MAP = False


def detect(path: Path) -> bool:
    """A TeraFlash export is recognised by its double extension ``.pulse.csv``."""
    return path.name.lower().endswith(".pulse.csv")


def load(path: Path, *, signal_column: int = 1) -> TimeTrace:
    """Load a ``.pulse.csv`` file.

    Parameters
    ----------
    signal_column : {1, 2}
        Which of the two signal columns to return (default 1, the first after
        the time column).
    """
    # Only two signal columns exist; anything else is a typo, not a request for column 3.
    if signal_column not in (1, 2):
        raise ValueError(f"signal_column must be 1 or 2 for a .pulse.csv file, got {signal_column!r}")

    # Read only the time and the chosen column: the empty trailing columns of some exports
    # would make a full read fail.
    with path.open("r", encoding="utf-8", errors="replace") as file:
        header = file.readline().strip()
    data = np.atleast_2d(np.genfromtxt(
        path, delimiter=",", skip_header=1, usecols=(0, signal_column), dtype=np.float64, autostrip=True,
    ))

    # Drop rows that are not two numbers (e.g. a truncated last line).
    data = data[np.isfinite(data).all(axis=1)]
    if data.shape[0] < 2:
        raise ValueError(f"{path.name}: fewer than 2 numeric rows of [time, signal {signal_column}]")

    return TimeTrace(
        time_ps=data[:, 0], signal=data[:, 1], path=path,
        metadata={"header": header, "signal_column": signal_column},
    )
