"""cjcthztds: THz time-domain spectroscopy measurement processing.

Load time traces from instrument files (:func:`load_time_trace`,
:func:`load_time_trace_map`), and turn them into spectra and complex field
ratios relative to a reference with :func:`analyze`, which keeps every
intermediate stage in a frozen :class:`Analysis`. The steps are also
available on their own: peak finding, Tukey windows, zero padding, Fourier
transform. numpy and scipy are the only dependencies.

Everything that needs a screen (file dialogs, the window selector, the
mapper window, result plots, :func:`cjcthztds.app.run_analysis`) lives in
:mod:`cjcthztds.app` and needs the ``app`` extra.
"""

from cjcthztds.formats import detect_format, load_time_trace, load_time_trace_map, supported_formats
from cjcthztds.padding import zero_pad
from cjcthztds.peaks import Peak, center_on_reference_peak, find_peak, peak_map, peak_spline_curve
from cjcthztds.pipeline import Analysis, analyze, common_time_axis
from cjcthztds.spectra import (
    amplitude_spectrum,
    frequency_range_mask,
    compute_complex_field_ratio,
    fourier_transform,
    unwrapped_phase,
)
from cjcthztds.traces import TimeTrace, TimeTraceMap, trace_at_position
from cjcthztds.windows import trace_window, tukey_window

__version__ = "0.1.0"

__all__ = [
    "Analysis",
    "Peak",
    "TimeTrace",
    "TimeTraceMap",
    "amplitude_spectrum",
    "analyze",
    "frequency_range_mask",
    "center_on_reference_peak",
    "common_time_axis",
    "compute_complex_field_ratio",
    "detect_format",
    "find_peak",
    "fourier_transform",
    "load_time_trace",
    "load_time_trace_map",
    "peak_map",
    "peak_spline_curve",
    "supported_formats",
    "trace_at_position",
    "trace_window",
    "tukey_window",
    "unwrapped_phase",
    "zero_pad",
]
