"""Menlo Systems TDS export (``.dat``), e.g. the Menlo spectrometer in Lab C FLX 8.100.

Layout (tab-separated text, CRLF line endings)::

    # Menlo Systems Time Domain Spectrometer
    # Program Version 1.4.6
    #  Start: -160 ps, Average over 500 waveforms. Date: 2023-11-24
    # Time [ps] 	 THz Signal [arb. u]
    -160.04254343	0.00000000
    -160.00918702	0.00000002
    ...

Four header lines starting with ``#``, then one row per time step:
time in ps, signal.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cjcthztds.traces import TimeTrace

DESCRIPTION = "Menlo TDS"
FILE_PATTERNS = ("*.dat",)
IS_MAP = False


def detect(path: Path) -> bool:
    """A Menlo export is recognised by its ``.dat`` extension."""
    return path.suffix.lower() == ".dat"


def load(path: Path) -> TimeTrace:
    """Load a Menlo ``.dat`` file; the ``#`` header lines are kept as ``metadata["header"]``."""
    # Keep the header lines (instrument, averaging, date) so a script can still report them.
    with path.open("r", encoding="utf-8", errors="replace") as file:
        header = [line.lstrip("#").strip() for line in file if line.startswith("#")]

    # The data rows are plain numbers; loadtxt skips the '#' header lines itself.
    data = np.atleast_2d(np.loadtxt(path, delimiter="\t", comments="#", dtype=np.float64))
    if data.shape[1] < 2 or data.shape[0] < 2:
        raise ValueError(f"{path.name}: expected at least 2 rows of [time, signal], got shape {data.shape}")

    return TimeTrace(time_ps=data[:, 0], signal=data[:, 1], path=path, metadata={"header": header})
