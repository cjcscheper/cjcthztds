"""Loading instrument files: one module per file format, one registry list.

Every format module defines

- ``DESCRIPTION``: a short name for file dialogs and error messages;
- ``FILE_PATTERNS``: glob patterns for the file dialogs (empty if the format has
  no extension);
- ``IS_MAP``: True if a file holds a whole raster scan (a :class:`TimeTraceMap`),
  False if it holds one trace (a :class:`TimeTrace`);
- ``detect(path) -> bool``: whether a file is in this format;
- ``load(path, **options)``: read it.

``FORMATS`` below lists them in the order they are tried: the first module
whose ``detect`` says yes loads the file. Formats recognised by their content
come first, so a file without a telling extension is not claimed by an
extension check. To add a format, copy ``_template.py`` and add one line here.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

from cjcthztds.formats import astrella, mapper_tim_csv, menlo_dat, toptica_pulse_csv
from cjcthztds.traces import TimeTrace, TimeTraceMap, trace_at_position

# Detection order: content-based first (Astrella has no extension), then by extension.
FORMATS: tuple[ModuleType, ...] = (
    astrella,
    toptica_pulse_csv,
    mapper_tim_csv,
    menlo_dat,
)


def supported_formats() -> list[tuple[str, tuple[str, ...]]]:
    """Return ``(description, file patterns)`` of every supported format, in detection order.

    This is what the file dialogs show; a format without an extension has no
    patterns and is found under "All files".
    """
    return [(module.DESCRIPTION, tuple(module.FILE_PATTERNS)) for module in FORMATS]


def detect_format(path: str | Path) -> ModuleType:
    """Return the format module that reads ``path``; raise ``ValueError`` if no format does."""
    # A missing file would otherwise be reported as "unsupported", which sends the user looking in the wrong place.
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no such file: {path}")

    # The first format in FORMATS that recognises the file wins.
    for module in FORMATS:
        if module.detect(path):
            return module
    supported = ", ".join(
        f"{module.DESCRIPTION} ({' '.join(module.FILE_PATTERNS)})" if module.FILE_PATTERNS else module.DESCRIPTION
        for module in FORMATS
    )
    raise ValueError(f"unsupported file format: {path.name}. Supported: {supported}")


def load_time_trace(path: str | Path, *, position: tuple[int, int] | None = None, **options) -> TimeTrace:
    """Load one time trace from any supported file.

    Parameters
    ----------
    path : str or Path
        The file. Its format is detected (see ``supported_formats()``).
    position : (iy, ix), optional
        Only for map files (a raster scan): the grid position of the trace to
        take. Required there, rejected for single-trace files.
    **options
        Options of the format's ``load``, e.g. ``signal_column=2`` for a
        Toptica ``.pulse.csv`` or ``signal_channel="Y"`` for an Astrella file.
    """
    # A map file holds many traces, so it needs a position; a single-trace file must not get one.
    module = detect_format(path)
    if module.IS_MAP:
        if position is None:
            raise ValueError(
                f"{Path(path).name} is a {module.DESCRIPTION} map; pass position=(iy, ix), "
                "or use load_time_trace_map() to load the whole scan"
            )
        return trace_at_position(module.load(Path(path), **options), position)
    if position is not None:
        raise ValueError(f"position is only for map files; {Path(path).name} holds a single {module.DESCRIPTION} trace")
    return module.load(Path(path), **options)


def load_time_trace_map(path: str | Path, **options) -> TimeTraceMap:
    """Load a whole raster scan (one trace per position) from a supported map file."""
    # Only map formats return a TimeTraceMap; say which loader to use instead.
    module = detect_format(path)
    if not module.IS_MAP:
        raise ValueError(f"{Path(path).name} is a single {module.DESCRIPTION} trace, not a map; use load_time_trace()")
    return module.load(Path(path), **options)
