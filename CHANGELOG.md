# Changelog

All notable changes to cjcthztds. Versions follow MAJOR.MINOR.PATCH: patch for
fixes, minor for new features, major for changes that break existing scripts.

## [Unreleased]

### Changed
- README: the example figures are grouped per example, each with a short
  explanation of what it shows.
- **Breaking:** `fourier_transform` (and so `Analysis.spectrum` and
  `amplitude_spectrum`) is now numpy's plain discrete transform `rfft`, no
  longer multiplied by `dt` and `exp(-i 2 pi f t0)`. Spectra are in the
  signal's units and their phase counts from the first sample. Field ratios
  and power ratios are unchanged.
- The amplitude-spectrum figure of `plot_analysis` and the raw spectra in the
  mapper window are shown relative to the reference's maximum (the largest
  selected spectrum in the mapper window), so the scale no longer depends on
  the sampling. Axis labels say `[arb. u]` instead of `[a.u.]`, and the
  mapper's field ratio `[-]`.

## [0.1.0] - 2026-10-21

First release.

### Added
- Loaders with automatic format detection: `load_time_trace`,
  `load_time_trace_map`, `detect_format`, `supported_formats`, for Menlo `.dat`,
  Toptica TeraFlash `.pulse.csv`, THz mapper `.tim.csv` raster scans and
  Astrella far-field exports (delay axis rebuilt from the commanded grid; the
  stage readback is unreliable). One module per format and one registry list
  (`formats.FORMATS`); `formats/_template.py` for adding a format.
- Frozen dataclasses `TimeTrace` and `TimeTraceMap` for measured data, and
  `trace_at_position` to take one trace out of a map.
- `analyze(traces, ...)` returning a frozen `Analysis` with every stage: traces
  as loaded, the common centred time axis and signals, windows, windowed and
  padded traces, frequencies, spectra, complex field ratio, power ratio, peak
  times and the settings used.
- The steps on their own: `common_time_axis`, `find_peak` (maximum of a cubic
  spline through the samples around the largest one, returns a `Peak`), `peak_spline_curve`,
  `center_on_reference_peak`, `peak_map`, `tukey_window`, `trace_window`,
  `zero_pad`, `fourier_transform` (scaled by `dt` and referred to t = 0),
  `compute_complex_field_ratio`, `frequency_range_mask`, `unwrapped_phase`,
  `amplitude_spectrum`.
- `cjcthztds.app` (needs the `app` extra): `run_analysis` (interactive analysis
  that returns the `Analysis`; also `python -m cjcthztds.app.run` and the console
  command `cjcthztds-analyze`), file dialogs with filters from the format
  registry (`select_files`, `select_reference_file`, `select_sample_files`,
  `select_map_file`), the window selector (`select_window_regions`), the 2D
  mapper window (`select_map_positions`, `MapSelection`) and the result figures
  (`plot_analysis`); `plot_peak_fit` and `run_analysis(show_peak_fits=True)` show
  the samples, spline and peak found for each trace.
- THEORY.md (processing chain, windowing trade-offs, zero padding, the sign
  convention and how to compare with cjctmm), examples with synthetic data,
  synthetic test files from `tests/generate_test_data.py`, tests and a GitHub
  Actions workflow.
