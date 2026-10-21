"""Loading instrument files: every format reads its test file into the right values; detection and errors."""

from pathlib import Path

import numpy as np
import pytest

from cjcthztds import (
    TimeTrace,
    TimeTraceMap,
    detect_format,
    find_peak,
    load_time_trace,
    load_time_trace_map,
    supported_formats,
    trace_at_position,
)
from cjcthztds.formats import FORMATS

DATA = Path(__file__).resolve().parent / "data"


def _module_name(module) -> str:
    """'cjcthztds.formats.menlo_dat' -> 'menlo_dat', the stem of its test file."""
    return module.__name__.rsplit(".", 1)[-1]


@pytest.mark.parametrize("module", FORMATS, ids=_module_name)
def test_every_format_has_a_test_file(module):
    """Every registered format has exactly one test file named after it, detected as that format and loadable."""
    matches = [path for path in DATA.iterdir() if path.name.split(".")[0] == _module_name(module)]
    assert len(matches) == 1, f"add tests/data/{_module_name(module)}.<ext> (see tests/generate_test_data.py)"
    path = matches[0]
    assert detect_format(path) is module
    loaded = load_time_trace_map(path) if module.IS_MAP else load_time_trace(path)
    assert isinstance(loaded, TimeTraceMap if module.IS_MAP else TimeTrace)


def test_format_modules_define_the_registry_interface():
    """Every format module has DESCRIPTION, FILE_PATTERNS, IS_MAP, detect and load."""
    for module in FORMATS:
        assert isinstance(module.DESCRIPTION, str) and module.DESCRIPTION
        assert isinstance(module.FILE_PATTERNS, tuple)
        assert isinstance(module.IS_MAP, bool)
        assert callable(module.detect) and callable(module.load)


def test_supported_formats_lists_the_registry_in_order():
    """supported_formats() gives (description, patterns) per format, Astrella (content-detected) first."""
    formats = supported_formats()
    assert [description for description, _ in formats] == [module.DESCRIPTION for module in FORMATS]
    assert formats[0] == ("Astrella far-field TDS (no extension)", ())
    assert ("Menlo TDS", ("*.dat",)) in formats


def _check_pulse(trace: TimeTrace, n_samples: int, scale: float = 1.0):
    """The generated pulse: time from -5 ps in 0.05 ps steps, +scale at -0.2 ps and -scale at +0.2 ps."""
    assert trace.time_ps.shape == trace.signal.shape == (n_samples,)
    assert trace.time_ps[0] == pytest.approx(-5.0)
    assert np.allclose(np.diff(trace.time_ps), 0.05)
    # Both lobes are sampled exactly (-0.2 and 0.2 lie on the grid); the noise is 1e-3. Checking the sign of
    # each lobe also catches a reversed time axis, which would swap them.
    assert np.interp(-0.2, trace.time_ps, trace.signal) == pytest.approx(scale, abs=0.005)
    assert np.interp(0.2, trace.time_ps, trace.signal) == pytest.approx(-scale, abs=0.005)


def test_menlo_dat():
    """Menlo .dat: 400 samples of [time, signal], the 4 header lines kept as metadata."""
    trace = load_time_trace(DATA / "menlo_dat.dat")
    _check_pulse(trace, 400)
    assert trace.path == DATA / "menlo_dat.dat"
    assert trace.metadata["header"][0] == "Menlo Systems Time Domain Spectrometer"
    assert len(trace.metadata["header"]) == 4


def test_toptica_pulse_csv_both_columns():
    """Toptica .pulse.csv: signal_column picks one of the two signals (signal 2 = 0.5 * signal 1 in the test file)."""
    first = load_time_trace(DATA / "toptica_pulse_csv.pulse.csv")
    second = load_time_trace(DATA / "toptica_pulse_csv.pulse.csv", signal_column=2)
    _check_pulse(first, 400)
    # Rounded to 6 decimals in the file, so equal to within 1e-6.
    assert np.allclose(second.signal, 0.5 * first.signal, atol=1e-6)
    assert second.metadata["signal_column"] == 2
    with pytest.raises(ValueError, match="signal_column must be 1 or 2"):
        load_time_trace(DATA / "toptica_pulse_csv.pulse.csv", signal_column=3)


def test_toptica_pulse_csv_ignores_empty_trailing_columns(tmp_path):
    """Some TeraFlash exports end every row with empty columns; they do not break the loader."""
    path = tmp_path / "trace.pulse.csv"
    path.write_text("time,sig1,sig2,unused1,unused2\n0.0,1.0,10.0,,\n1.0,2.0,20.0,,\n", encoding="utf-8")
    trace = load_time_trace(path, signal_column=2)
    assert np.array_equal(trace.time_ps, [0.0, 1.0]) and np.array_equal(trace.signal, [10.0, 20.0])


def test_mapper_tim_csv_grid_and_positions():
    """Mapper .tim.csv: a 3 x 4 grid in mm, row iy * No_X + ix at signal[iy, ix] (scaled 1 + 0.1 iy + 0.01 ix)."""
    trace_map = load_time_trace_map(DATA / "mapper_tim_csv.tim.csv")
    assert trace_map.signal.shape == (3, 4, 200)
    assert np.allclose(trace_map.x_mm, [20.0, 20.5, 21.0, 21.5])
    assert np.allclose(trace_map.y_mm, [10.0, 10.5, 11.0])
    assert np.allclose(trace_map.time_ps, -5.0 + 0.05 * np.arange(200))
    assert trace_map.metadata["Scan"] == "S" and trace_map.metadata["No_X"] == 4.0
    # The scale factor identifies the row; it would be wrong if x and y were swapped. Sample 96 is t = -0.2 ps.
    for iy, ix in [(0, 0), (2, 1), (1, 3)]:
        assert trace_map.signal[iy, ix, 96] == pytest.approx(1.0 + 0.1 * iy + 0.01 * ix, abs=0.005)


def test_mapper_trace_at_position():
    """A single trace out of a map keeps the map's path and records its grid index and position in mm."""
    trace = load_time_trace(DATA / "mapper_tim_csv.tim.csv", position=(2, 1))
    _check_pulse(trace, 200, scale=1.21)
    assert trace.metadata["position"] == (2, 1)
    assert trace.metadata["x_mm"] == 20.5 and trace.metadata["y_mm"] == 11.0
    assert trace.path == DATA / "mapper_tim_csv.tim.csv"


def test_map_files_need_a_position_and_single_files_refuse_one():
    """load_time_trace on a map needs position=(iy, ix); a single-trace file rejects it; a map loader rejects traces."""
    with pytest.raises(ValueError, match=r"pass position=\(iy, ix\)"):
        load_time_trace(DATA / "mapper_tim_csv.tim.csv")
    with pytest.raises(ValueError, match="position is only for map files"):
        load_time_trace(DATA / "menlo_dat.dat", position=(0, 0))
    with pytest.raises(ValueError, match="not a map"):
        load_time_trace_map(DATA / "menlo_dat.dat")
    with pytest.raises(ValueError, match=r"outside the map grid of shape \(ny, nx\) = \(3, 4\)"):
        trace_at_position(load_time_trace_map(DATA / "mapper_tim_csv.tim.csv"), (3, 0))


def test_mapper_missing_header_field(tmp_path):
    """A mapper file without a required header field names it."""
    path = tmp_path / "scan.tim.csv"
    header = "No_Y,No_X,dy,dx,No_values,dt/ps,Begin/ps,YO\n1,1,0.5,0.5,2,0.05,-5.0,10.0\n"
    path.write_text(header + "1.0,2.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing the numeric fields \['XO'\]"):
        load_time_trace_map(path)


def test_astrella_axis_rebuilt_from_grid_despite_glitches():
    """Astrella: stale, 0.0 and far-off readbacks do not matter; the axis is the commanded grid, negated, increasing."""
    trace = load_time_trace(DATA / "astrella")
    # ProbeDelay -15 to 5 ps in 401 steps of 0.05 ps -> THz time -5 to 15 ps, every row kept.
    assert trace.time_ps.size == 401
    assert np.allclose(trace.time_ps, -5.0 + 0.05 * np.arange(401))
    assert trace.metadata["delay_axis"] == "grid"
    assert trace.metadata["probe_delay_steps"] == 401 and trace.metadata["Comment"] == "synthetic test file"
    # The pulse sits where it was written (+1 at THz time -0.2 ps, -1 at +0.2 ps); a misassigned or
    # non-negated axis would move or swap the lobes.
    _check_pulse(trace, 401)


def test_astrella_channels():
    """signal_channel picks lock-in X (default), Y (0.05 x the pulse in the test file) or R = hypot(X, Y)."""
    x = load_time_trace(DATA / "astrella")
    y = load_time_trace(DATA / "astrella", signal_channel="Y")
    r = load_time_trace(DATA / "astrella", signal_channel="R")
    assert np.interp(-0.2, y.time_ps, y.signal) == pytest.approx(0.05, abs=0.005)
    assert np.allclose(r.signal, np.hypot(x.signal, y.signal))
    with pytest.raises(ValueError, match="signal_channel must be 'X', 'Y' or 'R'"):
        load_time_trace(DATA / "astrella", signal_channel=1)


def test_astrella_readback_axis_rejects_glitches():
    """delay_axis='readback' drops the 0.0 and far-off rows, averages stale repeats and fills the gaps on the grid."""
    with pytest.warns(RuntimeWarning, match="interpolated"):
        trace = load_time_trace(DATA / "astrella", delay_axis="readback")
    assert trace.metadata["delay_axis"] == "readback"
    # Still on the uniform grid, from -5 to 15 ps up to a stale end row; stale rows leave gaps that are interpolated.
    assert np.allclose(np.diff(trace.time_ps), 0.05)
    assert trace.time_ps[0] == pytest.approx(-5.0, abs=0.06) and trace.time_ps[-1] == pytest.approx(15.0, abs=0.06)
    assert find_peak(trace.time_ps, trace.signal).time_ps == pytest.approx(-0.2, abs=0.03)


def test_astrella_falls_back_to_readback_when_rows_do_not_match_steps(tmp_path):
    """With far fewer rows than commanded steps, the grid cannot be indexed by row: warn and use the readback."""
    lines = (DATA / "astrella").read_text(encoding="utf-8").splitlines()
    path = tmp_path / "short_scan"
    path.write_text("\n".join(lines[:200]) + "\n", encoding="utf-8")
    # Rows 0-196 of the scan: probe delay -15 to -5.2 ps, so THz time 5.2 to 15 ps, all at the grid step.
    # Two warnings: the fallback itself, and the stale rows' gaps that were interpolated.
    with (
        pytest.warns(RuntimeWarning, match="cannot be indexed by row number"),
        pytest.warns(RuntimeWarning, match="interpolated"),
    ):
        trace = load_time_trace(path)
    assert trace.metadata["delay_axis"] == "readback"
    assert np.allclose(np.diff(trace.time_ps), 0.05)
    assert trace.time_ps[0] == pytest.approx(5.2, abs=0.06) and trace.time_ps[-1] == pytest.approx(15.0, abs=0.06)


def test_unsupported_and_missing_files(tmp_path):
    """An unknown file gives a ValueError listing the supported formats; a missing file a FileNotFoundError."""
    path = tmp_path / "notes.txt"
    path.write_text("hello\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"unsupported file format: notes.txt. Supported: Astrella"):
        load_time_trace(path)
    with pytest.raises(FileNotFoundError, match="no such file"):
        load_time_trace(tmp_path / "missing.dat")


def test_astrella_detection_is_by_content(tmp_path):
    """A file without extension is Astrella only if its first non-empty line starts with ProbeDelay:."""
    from cjcthztds.formats import astrella

    good, bad = tmp_path / "scan_a", tmp_path / "scan_b"
    good.write_text("\n  ProbeDelay: -1, 1, 3, 1\n", encoding="utf-8")
    bad.write_text("Comment: x\nProbeDelay: -1, 1, 3, 1\n", encoding="utf-8")
    assert astrella.detect(good) and not astrella.detect(bad)
