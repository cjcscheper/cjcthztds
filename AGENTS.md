# AGENTS.md — cjcthztds

Guidance for AI coding agents (and contributors) working in this repository.

This file covers much of the same ground as `README.md`, but for a different
reader: `README.md` is for people who *use* cjcthztds (install, the three ways
to start, supported files, what `analyze` returns); this file is for agents
who *work on* it (layout, design rules, conventions, tests, releases).
`THEORY.md` holds the processing chain, its reasons and the sign convention
for both. When something changes that more than one of them describes, such
as the public API, a file format or a convention, update all of them.

cjcthztds processes THz time-domain spectroscopy measurements: instrument
files in, time traces, spectra and complex field ratios out, every stage
kept in a frozen `Analysis`. It is measurement processing only: fitting,
material models and layers live in a separate materials package, which
consumes `Analysis.freq_hz` and `Analysis.field_ratio` and uses cjctmm for
the optics. Do not add fitting, transfer matrices or materials here.

## Commands

```bash
pip install -e ".[test]"                       # editable install with pytest (core only, as in CI)
pytest                                         # run all tests; app tests skip without the app extra
pip install -e ".[app]"                        # matplotlib + cjcplot (pinned) for cjcthztds.app and examples
python examples/example_field_ratio.py         # each example saves its PNG(s) next to itself
python tests/generate_test_data.py             # rewrite the synthetic instrument files in tests/data/
python -m cjcthztds.app.run                    # the interactive analysis (also: cjcthztds-analyze)
```

## Public API

Scripts import only from the package roots, never from deeper submodules:
`cjcthztds` for everything that computes, `cjcthztds.app` for everything that
needs a screen.

```python
from cjcthztds import load_time_trace, load_time_trace_map, supported_formats, detect_format
from cjcthztds import TimeTrace, TimeTraceMap, trace_at_position
from cjcthztds import analyze, Analysis, common_time_axis
from cjcthztds import find_peak, Peak, peak_spline_curve, center_on_reference_peak, peak_map
from cjcthztds import tukey_window, trace_window, zero_pad
from cjcthztds import fourier_transform, compute_complex_field_ratio, frequency_range_mask, unwrapped_phase, amplitude_spectrum
from cjcthztds.app import run_analysis, select_files, select_reference_file, select_sample_files, select_map_file
from cjcthztds.app import select_window_regions, select_map_positions, MapSelection, plot_analysis, plot_peak_fit
```

## Package layout

```
src/cjcthztds/
  __init__.py        core public API and __version__ (its only source); never imports app/
  traces.py          TimeTrace, TimeTraceMap (measured data only), trace_at_position
  formats/
    __init__.py      FORMATS (the one registry list, = detection order), load_time_trace,
                     load_time_trace_map, detect_format, supported_formats
    _template.py     commented template for a new format (not registered)
    astrella.py  toptica_pulse_csv.py  mapper_tim_csv.py  menlo_dat.py
  peaks.py           find_peak (cubic spline), Peak, peak_spline_curve, center_on_reference_peak, peak_map
  windows.py         tukey_window, trace_window (computation only)
  padding.py         zero_pad
  spectra.py         fourier_transform, compute_complex_field_ratio, frequency_range_mask, unwrapped_phase,
                     amplitude_spectrum
  pipeline.py        common_time_axis, analyze, Analysis
  app/               needs matplotlib + cjcplot; raises an install hint without them
    __init__.py      dependency check, app public API
    dialogs.py       tkinter file dialogs, filters built from formats.FORMATS
    window_selector.py   select_window_regions (SpanSelector)
    mapper_window.py     select_map_positions, MapSelection (the 2D mapper window)
    plots.py         plot_analysis (the result figures)
    run.py           run_analysis, main (console command); runnable as a file
tests/
  generate_test_data.py   writes tests/data/<format module>.<ext> from a synthetic pulse, fixed seeds
  data/              the generated files, kept in the repository (byte-exact, see .gitattributes)
  test_formats.py  test_peaks.py  test_windows_padding.py  test_spectra.py  test_pipeline.py
  test_dependencies.py    core imports no matplotlib or cjcplot; app without the extra gives the install hint
  test_app.py        headless app tests (dialogs/windows replaced); skipped without the app extra
examples/            synthetic_data.py (shared pulses) + one script per topic, PNG next to it
THEORY.md            processing chain, why each step, time convention, comparison with cjctmm
```

## Processing rules

- **One place per step.** Loading in `formats/`, peaks in `peaks.py`, windows in
  `windows.py`, padding in `padding.py`, the transform and the ratio in
  `spectra.py`. `pipeline.py` and `app/` call them; the mapper window uses the
  same loader, `peak_map`, `find_peak` and `fourier_transform` as the core.
  Never write a second copy of a loader, an FFT or a peak finder.
- **`analyze` only computes**: no dialogs, no plots, no prints, no files.
  Everything interactive is in `app/`, which calls `analyze`.
- **`Analysis` keeps every stage** and the settings used. A `TimeTrace` holds
  only measured time-domain data; anything derived belongs in `Analysis`.
- **The frequency band never limits the computation**: `freq_min_hz`,
  `freq_max_hz` are stored for plots and fits; every array covers 0 to Nyquist.
- **Sign convention is fixed** (THEORY.md, section 8): spectra use numpy's
  kernel `exp(-i omega t)`, with the `dt` and `exp(-i omega t0)` factors of
  `fourier_transform`; a delayed sample has a falling field-ratio phase.
  `test_field_ratio_of_a_delayed_copy_is_a_phase_ramp` pins it; never change
  that test to match code.
- **No user-editable settings in the package**: no module-level constants that
  users are expected to change. Every setting is a keyword argument with a
  documented default; `run_analysis` passes its settings to `analyze` under the
  same names. Format modules define `DESCRIPTION`, `FILE_PATTERNS`, `IS_MAP`
  (format facts, not settings).
- **File formats:** one module per format with `detect(path)` and
  `load(path, **options)`, its layout documented with example lines in the
  module docstring, and one line in `FORMATS`. Content-based detection comes
  before extension checks. Every registered format has a generated test file
  `tests/data/<module name>.<ext>`.
- The interactive windows (window selector, mapper window, result plots) keep
  their current look and behaviour until the maintainer redesigns them; change
  their internals only to fix bugs or to reuse core functions.

## Coding conventions

- Functions first: processing is written as plain functions (inputs in,
  outputs out). Classes are used ONLY as frozen dataclasses for returned data
  (`TimeTrace`, `TimeTraceMap`, `Peak`, `Analysis`, `MapSelection`): no
  methods, no inheritance, no objects that change themselves, no classes that
  only group functions (a module does that). The interactive windows keep the
  closures they need, but no class hierarchies.
- Every logical block inside a function gets a one-line `#` comment saying why
  it exists. Docstrings explain why and how (where a formula comes from, which
  sign and why).
- SI units need no suffix (`freq_hz` is the one exception, kept explicit as
  the package's frequency name; `omega` in rad/s); non-SI or scaled values carry
  one (`time_ps`, `padding_ps`, `x_mm`, `freq_thz`).
- Naming: lowercase `t`, `r` for complex field ratios, uppercase `T`, `R` for
  power. Geometry-agnostic names (`compute_complex_field_ratio`, `field_ratio`),
  never "transmission" for something that is valid in reflection too.
  Physicist-first names (`find_peak`, `peak_times_ps`), no programmer idioms.
- Shared code goes in one module and is imported, never copied.
- Remove unused code instead of leaving it commented out.
- Python 3.10 compatible (no syntax newer than 3.10).

## Dependencies

- All dependencies live in `pyproject.toml`, the only list: `[project]
  dependencies` (numpy, scipy), the `test` extra (pytest) and the `app` extra
  (matplotlib, cjcplot pinned to a release tag). There is no `requirements.txt`.
- The core (everything outside `app/`) and `tests/` never import matplotlib or
  cjcplot (`test_core_does_not_import_matplotlib_or_cjcplot` checks it); only
  `app/` and `examples/` may, via the pinned `app` extra. When upgrading the
  pin, change the tag in pyproject.toml (and in the README) and regenerate all
  example PNGs. Examples use cjcplot's public API only, save their PNG with
  `save_path=..., fig_show=False`, size 1D figures `fig_aspect_ratio=(0.5, 0.5)`,
  and draw the reference black and the sample blue
  (`colors=["black", style_colors()[0]]`).

## Tests

- Tests check values that can be verified by hand: the Gaussian's analytic
  spectrum, a delayed copy's phase ramp `-omega dt`, a scaled copy's `|t| = 0.5`,
  an exactly reproduced cubic's maximum, window values at known points, the generated
  files' known numbers (lobes of +-1 at -+0.2 ps, grid start and step).
- Wrong inputs raise `ValueError`s (or `FileNotFoundError`) with clear
  messages, and those messages are tested.
- New or changed functionality comes with new or updated tests in the same
  change, and `pytest` passes before committing. When the public API changes,
  update the README, this file, THEORY.md, CHANGELOG.md and the affected
  examples (and their PNGs) too.
- Every test has a one-line docstring saying what it checks; non-obvious lines
  (the reason behind a value or tolerance) get a short comment.
- App tests run headless (Agg, dialogs and windows replaced with
  `monkeypatch`) and skip when the app extra is missing.
- GitHub Actions (`.github/workflows/tests.yml`) runs `pytest` on Python 3.10
  and 3.14 for every push and pull request, core only (without the app extra).

## Changelog and releases

- Every change that users notice (new or changed functionality, fixes, docs
  people rely on) gets a line in `CHANGELOG.md` under `## [Unreleased]`, in the
  same change, sorted under `### Added`, `### Changed`, `### Fixed` or `### Removed`.
- Before finishing a change, check that `CHANGELOG.md` describes it.
- Versions follow MAJOR.MINOR.PATCH: patch for fixes, minor for new features,
  major for changes that break existing scripts (below 1.0 the API may still change).
  A change of a convention, a file format's reading or of returned values counts as breaking.
- A release:
  1. rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD` and start a new empty
     `## [Unreleased]` above it;
  2. set the new version in `src/cjcthztds/__init__.py` (`__version__`), its only
     source: `pyproject.toml` reads it from there; also update the pinned
     version in the README's install command;
  3. commit, tag `vX.Y.Z`, push the commit and the tag, and publish the GitHub
     release with the changelog section as its notes.

## Git workflow

- Do not commit, push, tag, merge or force-push without the maintainer's approval.
- Keep each change focused: no unrelated reformatting or renaming mixed in.
- Keep generated files in the same change as the code that produces them:
  regenerate an example's PNG when the example or the processing changes, and
  the files in `tests/data/` when `tests/generate_test_data.py` changes.
- In the commit message (or pull request), summarise what changes for users and
  how it was checked (tests run, examples regenerated).
