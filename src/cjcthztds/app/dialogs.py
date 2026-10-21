"""File dialogs (tkinter) whose filters list every supported format."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog

from cjcthztds.formats import FORMATS


def _file_types(*, maps_only: bool = False) -> list[tuple[str, str]]:
    """The dialog filter: all supported extensions together, one entry per format, then all files.

    Built from the format registry, so a newly added format shows up without
    touching this file. A format without an extension (Astrella) is reached
    through "All files".
    """
    # One entry per format that has an extension; map formats only when asked for maps.
    formats = [module for module in FORMATS if module.IS_MAP or not maps_only]
    entries = [(module.DESCRIPTION, " ".join(module.FILE_PATTERNS)) for module in formats if module.FILE_PATTERNS]

    # The combined entry comes first, so every supported file is visible at once.
    all_patterns = " ".join(patterns for _, patterns in entries)
    return [("THz trace files", all_patterns), *entries, ("All files", "*.*")]


def _ask(title: str, *, multiple: bool, maps_only: bool = False) -> list[Path]:
    """Show an open-file dialog without an empty Tk main window next to it."""
    # A hidden root window, destroyed again even if the dialog raises.
    root = tk.Tk()
    root.withdraw()
    try:
        if multiple:
            chosen = filedialog.askopenfilenames(title=title, filetypes=_file_types(maps_only=maps_only))
        else:
            chosen = filedialog.askopenfilename(title=title, filetypes=_file_types(maps_only=maps_only))
            chosen = (chosen,) if chosen else ()
    finally:
        root.destroy()
    return [Path(path) for path in chosen]


def select_reference_file() -> Path | None:
    """Ask for the reference file; None if the dialog was cancelled."""
    chosen = _ask("Select reference file", multiple=False)
    return chosen[0] if chosen else None


def select_sample_files() -> list[Path]:
    """Ask for one or more sample files; an empty list if the dialog was cancelled."""
    return _ask("Select sample files", multiple=True)


def select_map_file() -> Path | None:
    """Ask for a map file (a raster scan); None if the dialog was cancelled."""
    chosen = _ask("Select map file", multiple=False, maps_only=True)
    return chosen[0] if chosen else None


def select_files() -> list[Path]:
    """Ask for a reference file, then for the sample files; return them with the reference first.

    The list goes straight into the loaders and ``analyze`` (reference index 0)::

        paths = select_files()
        analysis = analyze([load_time_trace(path) for path in paths])

    Raises ``RuntimeError`` if either dialog is cancelled. Map files need a
    position per trace; pick those with ``run_analysis()`` or
    ``select_map_positions()``.
    """
    # Both dialogs must give something: an analysis without a reference or without samples is meaningless.
    reference_path = select_reference_file()
    if reference_path is None:
        raise RuntimeError("no reference file selected")
    sample_paths = select_sample_files()
    if not sample_paths:
        raise RuntimeError("no sample files selected")
    return [reference_path, *sample_paths]
