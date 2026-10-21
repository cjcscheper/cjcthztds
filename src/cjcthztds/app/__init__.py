"""Everything in cjcthztds that needs a screen: file dialogs, interactive windows, result plots.

``run_analysis()`` is the interactive analysis from start to end: pick files,
select windows, see the results, get the :class:`cjcthztds.Analysis` back.
The parts are available on their own as well.

Needs the ``app`` extra (matplotlib and cjcplot): ``pip install "cjcthztds[app]"``.
The core package (``import cjcthztds``) works without it.
"""

# Check the optional dependencies first, so a missing one gives an install hint instead of a bare ImportError.
try:
    import cjcplot  # noqa: F401
    import matplotlib  # noqa: F401
except ImportError as error:
    raise ImportError(
        f"cjcthztds.app needs matplotlib and cjcplot ({error.name} is missing). "
        'Install them with: pip install "cjcthztds[app] @ git+https://github.com/cjcscheper/cjcthztds.git" '
        '(or pip install -e ".[app]" in a clone).'
    ) from error

from cjcthztds.app.dialogs import select_files, select_map_file, select_reference_file, select_sample_files
from cjcthztds.app.mapper_window import MapSelection, select_map_positions
from cjcthztds.app.plots import plot_analysis, plot_peak_fit
from cjcthztds.app.run import run_analysis
from cjcthztds.app.window_selector import select_window_regions

__all__ = [
    "MapSelection",
    "plot_analysis",
    "plot_peak_fit",
    "run_analysis",
    "select_files",
    "select_map_file",
    "select_map_positions",
    "select_reference_file",
    "select_sample_files",
    "select_window_regions",
]
