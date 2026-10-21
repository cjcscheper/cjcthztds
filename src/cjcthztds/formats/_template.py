"""Template for a new file format. It is not registered and never runs.

To add a format:

1. Copy this file to ``formats/<instrument>_<extension>.py`` (e.g. ``teraview_txt.py``)
   and fill in the four parts below. Describe the file layout in the module
   docstring, with a few example lines, the way the other format modules do:
   that is the only documentation of the format.
2. Add the module to ``FORMATS`` in ``formats/__init__.py``. The position in
   that list is the detection order: put a format recognised by its content
   before any format recognised by an extension it shares.
3. Add a small test file ``tests/data/<module name>.<extension>``, made by
   ``tests/generate_test_data.py`` (add a writer there), and a test in
   ``tests/test_formats.py`` that checks the loaded values.
   ``test_every_format_has_a_test_file`` fails until the file exists.
4. Mention the format in README.md (supported formats) and CHANGELOG.md.

Layout (example)::

    Time (ps)	Signal (V)
    -5.000	0.0012
    ...
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cjcthztds.traces import TimeTrace

# Shown in the file dialogs and in "unsupported format" errors.
DESCRIPTION = "My instrument"
# Glob patterns for the file dialogs; () for a format without an extension.
FILE_PATTERNS = ("*.myext",)
# True when one file holds a raster scan: load() then returns a TimeTraceMap.
IS_MAP = False


def detect(path: Path) -> bool:
    """Return True if ``path`` is a file in this format.

    Prefer the extension (``path.name.lower().endswith(".myext")``). If the
    extension is shared with another format, or missing, read only the first
    few lines and look for something unique, and return False instead of
    raising on files you cannot read.
    """
    return path.name.lower().endswith(".myext")


def load(path: Path) -> TimeTrace:
    """Read the file into a TimeTrace: time in ps (increasing), the signal, and any header values.

    Add keyword-only options (``*, signal_column: int = 1``) only where the
    file really offers a choice, check them, and document them here.
    """
    # Read the numbers; convert the time to ps if the file stores another unit.
    data = np.loadtxt(path, skiprows=1, dtype=np.float64)
    return TimeTrace(time_ps=data[:, 0], signal=data[:, 1], path=path, metadata={})
