"""Interactive window for a raster scan (TimeTraceMap): pick a reference position and measurement positions.

What the window shows and does:

- a map of the peak amplitude (or, with the "Map" button, the peak time relative
  to the reference) of every position;
- right-click a cell: make it the reference (black frame); right-click it again
  to clear it;
- left-click a cell: add or remove it; left-drag: add every cell on the line;
  "Select 4-inch wafer" selects the cells inside the wafer edge found in the
  peak-amplitude map; "Reset" clears everything;
- on the right, for the selected cells in selection order (colours): the time
  traces, the real and imaginary part of their spectra (or of their field ratio
  to the reference, once one is chosen), and their peak amplitudes and peak
  times.

Close the window to return the selection.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from cjcplot import plot_2d, style_colors
from matplotlib.backend_bases import MouseEvent
from matplotlib.patches import Rectangle
from matplotlib.ticker import AutoMinorLocator
from matplotlib.widgets import Button

from cjcthztds import TimeTraceMap, compute_complex_field_ratio, find_peak, fourier_transform, peak_map


@dataclass(frozen=True)
class MapSelection:
    """The positions picked in the mapper window, as grid indices ``(iy, ix)``.

    Attributes
    ----------
    reference_position : (iy, ix) or None
        The reference cell (right-click), or None if none was chosen.
    selected_positions : tuple of (iy, ix)
        The measurement cells (left-click / drag / wafer button), in selection order.
    """

    reference_position: tuple[int, int] | None
    selected_positions: tuple[tuple[int, int], ...]


def _minor_ticks(ax) -> None:
    """One minor tick between major ticks, as cjcplot draws them."""
    ax.xaxis.set_minor_locator(AutoMinorLocator(n=2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(n=2))


def _cell_edges(centers: np.ndarray) -> np.ndarray:
    """Edges of the grid cells around the given cell centres (half a step beyond the first and last)."""
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])
    mids = 0.5 * (centers[:-1] + centers[1:])
    return np.concatenate([[centers[0] - 0.5 * (centers[1] - centers[0])], mids,
                           [centers[-1] + 0.5 * (centers[-1] - centers[-2])]])


def _line_cells(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    """Grid cells on the straight line from ``start`` to ``end`` (Bresenham's algorithm)."""
    # Step along the longer direction and accumulate the error in the other one.
    y0, x0 = start
    y1, x1 = end
    dy, dx = abs(y1 - y0), abs(x1 - x0)
    sy, sx = (1 if y0 < y1 else -1), (1 if x0 < x1 else -1)
    err = dx - dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((y0, x0))
        if y0 == y1 and x0 == x1:
            return cells
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def _fit_circle(x_vals: np.ndarray, y_vals: np.ndarray) -> tuple[float, float, float] | None:
    """Least-squares circle ``(cx, cy, radius)`` through scattered points, or None if there is none."""
    # x² + y² + a x + b y + c = 0 is linear in a, b, c.
    if x_vals.size < 8:
        return None
    design = np.column_stack((x_vals, y_vals, np.ones_like(x_vals)))
    (a, b, c), *_ = np.linalg.lstsq(design, -(x_vals**2 + y_vals**2), rcond=None)
    cx, cy = -0.5 * a, -0.5 * b
    radius_sq = cx**2 + cy**2 - c
    if not np.isfinite(radius_sq) or radius_sq <= 0.0:
        return None
    return float(cx), float(cy), float(np.sqrt(radius_sq))


def _erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    """Shrink a boolean mask by one cell per iteration (a cell stays only if all 8 neighbours are set)."""
    eroded = mask.astype(bool, copy=True)
    for _ in range(iterations):
        core = np.zeros_like(eroded)
        core[1:-1, 1:-1] = (
            eroded[1:-1, 1:-1] & eroded[:-2, :-2] & eroded[:-2, 1:-1] & eroded[:-2, 2:]
            & eroded[1:-1, :-2] & eroded[1:-1, 2:] & eroded[2:, :-2] & eroded[2:, 1:-1] & eroded[2:, 2:]
        )
        eroded = core
        if not np.any(eroded):
            break
    return eroded


def _wafer_mask(x_mm: np.ndarray, y_mm: np.ndarray, amplitude_map: np.ndarray) -> np.ndarray | None:
    """Cells inside a wafer, found from the sharp edge it leaves in the peak-amplitude map; None if not found.

    The strongest amplitude gradients mark the wafer edge. A circle fitted
    through them gives the centre; the edge radius is then taken per angle
    (median, smoothed, kept within 0.78–1.08 of the circle radius) so a flat
    or a chipped edge is followed. Four cells are eroded from the result to stay
    clear of the edge.
    """
    # Fill NaNs with the median, so the gradient does not spread them.
    finite = np.isfinite(amplitude_map)
    if np.count_nonzero(finite) < 12:
        return None
    filled = np.where(finite, amplitude_map, float(np.nanmedian(amplitude_map[finite])))

    # The 8 % steepest cells are the edge candidates.
    grad_y, grad_x = np.gradient(filled)
    grad = np.hypot(grad_x, grad_y)
    edge = grad >= float(np.nanpercentile(grad, 92.0))
    if np.count_nonzero(edge) < 24:
        return None
    x_grid, y_grid = np.meshgrid(x_mm, y_mm)
    circle = _fit_circle(x_grid[edge], y_grid[edge])
    if circle is None:
        return None
    cx, cy, radius = circle

    # Edge radius per angular bin (180 bins), gaps filled with the median, smoothed around the circle.
    n_bins = 180
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    edge_theta = np.arctan2(y_grid[edge] - cy, x_grid[edge] - cx)
    edge_rho = np.hypot(x_grid[edge] - cx, y_grid[edge] - cy)
    edge_bin = np.clip(np.digitize(edge_theta, bin_edges) - 1, 0, n_bins - 1)
    boundary = np.full(n_bins, np.nan)
    for index in range(n_bins):
        radii = edge_rho[edge_bin == index]
        if radii.size >= 2:
            boundary[index] = float(np.median(radii))
    if np.all(np.isnan(boundary)):
        return None
    boundary = np.where(np.isnan(boundary), float(np.nanmedian(boundary)), boundary)
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0]) / 9.0
    boundary = np.convolve(np.pad(boundary, 2, mode="wrap"), kernel, mode="valid")
    boundary = np.clip(boundary, 0.78 * radius, 1.08 * radius)

    # A cell is inside if it lies within the edge radius of its own angle.
    point_bin = np.clip(np.digitize(np.arctan2(y_grid - cy, x_grid - cx), bin_edges) - 1, 0, n_bins - 1)
    inside = (np.hypot(x_grid - cx, y_grid - cy) <= boundary[point_bin]) & finite
    return _erode(inside, iterations=4)


def select_map_positions(trace_map: TimeTraceMap, *, show_analysis_panels: bool = True) -> MapSelection:
    """Open the mapper window for ``trace_map`` and return the positions picked when it is closed.

    Parameters
    ----------
    trace_map : TimeTraceMap
        The scan, e.g. from ``load_time_trace_map``.
    show_analysis_panels : bool
        Show the trace, spectrum and peak panels next to the map (default True).
    """
    # Peak maps once, with the same spline peak finding as the rest of the package.
    peak_time_map, peak_amplitude_map = peak_map(trace_map)
    peak_amplitude_map = np.abs(peak_amplitude_map)
    x_mm, y_mm, time_ps = trace_map.x_mm, trace_map.y_mm, trace_map.time_ps
    colors = style_colors()

    # Layout in text widths (497 pt): map block 1.4 wide, colorbar 0.12, analysis column 1.0.
    fig_textwidth_pt = 497.0
    fig = plt.figure(figsize=(2.52 * fig_textwidth_pt / 72.27, 1.5 * fig_textwidth_pt / 72.27), layout="constrained")
    if show_analysis_panels:
        outer = fig.add_gridspec(1, 3, width_ratios=[1.4, 0.12, 1.0])
        right = outer[0, 2].subgridspec(3, 1, height_ratios=[0.5, 0.5, 0.5])
        ax_td = fig.add_subplot(right[0, 0])
        fft_split = right[1, 0].subgridspec(1, 2)
        ax_fft_real = fig.add_subplot(fft_split[0, 0])
        ax_fft_imag = fig.add_subplot(fft_split[0, 1])
        bottom = right[2, 0].subgridspec(1, 2)
        ax_amp = fig.add_subplot(bottom[0, 0])
        ax_tpeak = fig.add_subplot(bottom[0, 1])
        ax_td.set_box_aspect(0.5)
        for ax in (ax_fft_real, ax_fft_imag, ax_amp, ax_tpeak):
            ax.set_box_aspect(1.0)
    else:
        outer = fig.add_gridspec(1, 2, width_ratios=[1.4, 0.12])
    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1.4, 0.1])
    cbar_col = outer[0, 1].subgridspec(2, 1, height_ratios=[1.4, 0.1])
    ax_map = fig.add_subplot(left[0, 0])
    ax_map_selector = fig.add_subplot(left[1, 0])
    ax_cbar = fig.add_subplot(cbar_col[0, 0])
    fig.add_subplot(cbar_col[1, 0]).set_axis_off()
    ax_map.set_box_aspect(1.0)

    # The two map quantities the "Map" button cycles through.
    map_options = {"Peak amplitude": peak_amplitude_map, "Peak time difference": peak_time_map}
    cbar_labels = {
        "Peak amplitude": "Peak amplitude (spline)",
        "Peak time difference": "Peak time difference (ps)",
    }
    selected_map_key = "Peak amplitude"
    plot_2d(
        x_mm, y_mm, map_options[selected_map_key],
        xlabel="x position", ylabel="y position", cbar_label=cbar_labels[selected_map_key],
        cmap="inferno", alpha=0.92, ax=ax_map, cax=ax_cbar,
    )
    mesh = ax_map.collections[0]
    cbar = mesh.colorbar

    # Buttons below the map.
    ax_map_selector.set_axis_off()
    reset_button = Button(ax_map_selector.inset_axes((0.0, 0.0, 1.0 / 3.0, 1.0)), "Reset")
    selector_button = Button(ax_map_selector.inset_axes((1.0 / 3.0, 0.0, 1.0 / 3.0, 1.0)), f"Map: {selected_map_key}")
    wafer_button = Button(ax_map_selector.inset_axes((2.0 / 3.0, 0.0, 1.0 / 3.0, 1.0)), "Select 4-inch wafer")

    # Selection state, changed by the mouse and button callbacks below.
    selected_order: list[tuple[int, int]] = []
    selection_patches: list[Rectangle] = []
    reference_key: tuple[int, int] | None = None
    drag_start_key: tuple[int, int] | None = None

    # Thin grey grid lines on the cell edges.
    x_edges, y_edges = _cell_edges(x_mm), _cell_edges(y_mm)
    ax_map.set_xticks(x_edges, minor=True)
    ax_map.set_yticks(y_edges, minor=True)
    ax_map.tick_params(which="minor", length=0)
    x_min, x_max = float(np.min(x_edges)), float(np.max(x_edges))
    y_min, y_max = float(np.min(y_edges)), float(np.max(y_edges))
    ax_map.vlines(x_edges, y_min, y_max, colors="#b8b8b8", linewidth=0.35, alpha=0.8, zorder=4)
    ax_map.hlines(y_edges, x_min, x_max, colors="#b8b8b8", linewidth=0.35, alpha=0.8, zorder=4)

    def selection_color(index: int) -> str:
        """Colour of the index-th selected cell, cycling through the style colours."""
        return colors[index % len(colors)]

    def key_from_event(event) -> tuple[int, int] | None:
        """The grid cell under the mouse, or None outside the map."""
        if event.inaxes is not ax_map or event.xdata is None or event.ydata is None:
            return None
        return int(np.argmin(np.abs(y_mm - event.ydata))), int(np.argmin(np.abs(x_mm - event.xdata)))

    def append_selection(key: tuple[int, int]) -> None:
        """Add a cell once, keeping the selection order."""
        if key not in selected_order:
            selected_order.append(key)

    def current_map_data() -> np.ndarray:
        """The map shown now; peak times are relative to the reference (or to the first cell)."""
        if selected_map_key == "Peak time difference":
            ref_iy, ref_ix = reference_key if reference_key is not None else (0, 0)
            return peak_time_map - float(peak_time_map[ref_iy, ref_ix])
        return map_options[selected_map_key]

    def autoscale_colorbar() -> None:
        """Fit the colour limits to the map shown now."""
        map_data = current_map_data()
        mesh.set_clim(vmin=float(np.nanmin(map_data)), vmax=float(np.nanmax(map_data)))
        cbar.update_normal(mesh)

    def update_map_quantity(label: str, *, autoscale: bool = True) -> None:
        """Show another map quantity."""
        nonlocal selected_map_key
        selected_map_key = label
        mesh.set_array(current_map_data().ravel())
        cbar.set_label(cbar_labels[selected_map_key])
        selector_button.label.set_text(f"Map: {selected_map_key}")
        if autoscale:
            autoscale_colorbar()
        fig.canvas.draw_idle()

    def draw_selected_markers() -> None:
        """Frame every selected cell in its colour and the reference in black."""
        for patch in selection_patches:
            patch.remove()
        selection_patches.clear()
        framed = [(key, selection_color(index), 1.2, 5) for index, key in enumerate(selected_order)]
        if reference_key is not None:
            framed.append((reference_key, "black", 1.8, 6))
        for (iy, ix), color, linewidth, zorder in framed:
            rect = Rectangle(
                (x_edges[ix], y_edges[iy]), x_edges[ix + 1] - x_edges[ix], y_edges[iy + 1] - y_edges[iy],
                fill=False, edgecolor=color, linewidth=linewidth, zorder=zorder,
            )
            ax_map.add_patch(rect)
            selection_patches.append(rect)

    def redraw_lower_panels() -> None:
        """Peak amplitude and peak time (relative to the first selected cell) per selected cell."""
        ax_amp.clear()
        ax_tpeak.clear()
        ax_amp.set_xlabel("selection index")
        ax_amp.set_ylabel("peak amplitude")
        ax_tpeak.set_xlabel("selection index")
        ax_tpeak.set_ylabel("peak time difference (ps)")
        if selected_order:
            peaks = [find_peak(time_ps, trace_map.signal[key]) for key in selected_order]
            index = np.arange(1, len(peaks) + 1)
            peak_times = np.array([peak.time_ps for peak in peaks])
            for ax, values, color in (
                (ax_amp, [abs(peak.amplitude) for peak in peaks], colors[0]),
                (ax_tpeak, peak_times - peak_times[0], colors[2]),
            ):
                ax.plot(
                    index, values, marker="o", color=color, linestyle="-",
                    markerfacecolor="white", markeredgecolor=color,
                )
        _minor_ticks(ax_amp)
        _minor_ticks(ax_tpeak)

    def redraw_right_panels() -> None:
        """Traces and Re/Im of their spectra, or of their field ratio once a reference is chosen."""
        ax_td.clear()
        ax_td.set_xlabel("time (ps)")
        ax_td.set_ylabel("signal [arb. u]")
        for ax, part in ((ax_fft_real, "Re"), (ax_fft_imag, "Im")):
            ax.clear()
            ax.set_xlabel("frequency (THz)")
            ax.set_ylabel(f"{part}(FFT) [arb. u]")
            ax.set_xlim(0.1, 2.5)

        # The reference in black; its spectrum turns the other spectra into field ratios.
        reference_spectrum = None
        if reference_key is not None:
            ref_trace = trace_map.signal[reference_key]
            ax_td.plot(time_ps, ref_trace, color="black", linestyle="-", linewidth=1.2)
            _, reference_spectrum = fourier_transform(time_ps, ref_trace)
            ax_fft_real.set_ylabel("Re(field ratio) [-]")
            ax_fft_imag.set_ylabel("Im(field ratio) [-]")

        # Spectra of the selected cells. Without a reference they are shown relative to the largest |spectrum|
        # among them, since the discrete transform's size depends on the number of samples.
        spectra = [fourier_transform(time_ps, trace_map.signal[key]) for key in selected_order]
        largest = max((float(np.max(np.abs(spectrum))) for _, spectrum in spectra), default=1.0) or 1.0

        # Each selected cell in its colour; collect the in-band values for the y limits.
        real_band, imag_band = [], []
        for index, ((iy, ix), (freq_hz, spectrum)) in enumerate(zip(selected_order, spectra)):
            color = selection_color(index)
            ax_td.plot(time_ps, trace_map.signal[iy, ix], color=color, linestyle="-")
            if reference_spectrum is None:
                response = spectrum / largest
            else:
                response = compute_complex_field_ratio(spectrum, reference_spectrum)
            freq_thz = freq_hz * 1e-12
            in_band = (freq_thz >= 0.1) & (freq_thz <= 2.5)
            real_band.append(np.real(response[in_band]))
            imag_band.append(np.imag(response[in_band]))
            ax_fft_real.plot(freq_thz, np.real(response), color=color, linestyle="-")
            ax_fft_imag.plot(freq_thz, np.imag(response), color=color, linestyle="-")

        # y limits from the band only, with 8 % margin; outside the band the ratio is noise.
        for ax, band_values in ((ax_fft_real, real_band), (ax_fft_imag, imag_band)):
            flat = np.concatenate(band_values) if band_values else np.array([])
            flat = flat[np.isfinite(flat)]
            if flat.size:
                ymin, ymax = float(np.min(flat)), float(np.max(flat))
                pad = 0.08 * (ymax - ymin) if ymax > ymin else 0.1 * max(abs(ymax), 1e-6)
                ax.set_ylim(ymin - pad, ymax + pad)
        for ax in (ax_td, ax_fft_real, ax_fft_imag):
            _minor_ticks(ax)

    def redraw_after_selection_update(*, refresh_peak_time_map: bool = False) -> None:
        """Redraw everything that depends on the selection."""
        draw_selected_markers()
        if refresh_peak_time_map and selected_map_key == "Peak time difference":
            update_map_quantity(selected_map_key, autoscale=False)
        if show_analysis_panels:
            redraw_right_panels()
            redraw_lower_panels()
        fig.canvas.draw_idle()

    def select_wafer_region(_event) -> None:
        """Select every cell inside the wafer, row by row in a snake order from the top."""
        wafer = _wafer_mask(x_mm, y_mm, peak_amplitude_map)
        if wafer is None:
            return
        selected_order.clear()
        rows = sorted(range(wafer.shape[0]), key=lambda iy: float(y_mm[iy]), reverse=True)
        columns = sorted(range(wafer.shape[1]), key=lambda ix: float(x_mm[ix]))
        for row_number, iy in enumerate(rows):
            for ix in (columns if row_number % 2 == 0 else columns[::-1]):
                if wafer[iy, ix]:
                    append_selection((iy, ix))
        redraw_after_selection_update()

    def cycle_map_quantity(_event) -> None:
        """Switch to the next map quantity."""
        options = list(map_options)
        update_map_quantity(options[(options.index(selected_map_key) + 1) % len(options)])

    def reset_selection(_event) -> None:
        """Clear the selection and the reference."""
        nonlocal reference_key
        selected_order.clear()
        reference_key = None
        redraw_after_selection_update(refresh_peak_time_map=True)

    def on_map_view_changed(_ax) -> None:
        """Back at the full view (home button): refit the colour limits."""
        xlo, xhi = sorted(float(v) for v in ax_map.get_xlim())
        ylo, yhi = sorted(float(v) for v in ax_map.get_ylim())
        home = ((xlo, x_min), (xhi, x_max), (ylo, y_min), (yhi, y_max))
        if all(np.isclose(a, b, atol=1e-9, rtol=1e-7) for a, b in home):
            autoscale_colorbar()

    def on_mouse_press(event) -> None:
        """Left button starts a click or drag; right button sets or clears the reference."""
        nonlocal reference_key, drag_start_key
        key = key_from_event(event) if isinstance(event, MouseEvent) else None
        if key is None:
            return
        if event.button == 1:
            drag_start_key = key
        elif event.button == 3:
            reference_key = None if reference_key == key else key
            redraw_after_selection_update(refresh_peak_time_map=True)

    def on_mouse_release(event) -> None:
        """A click toggles one cell; a drag adds every cell on the line."""
        nonlocal drag_start_key
        if not isinstance(event, MouseEvent) or event.button != 1 or drag_start_key is None:
            return
        start_key, drag_start_key = drag_start_key, None
        end_key = key_from_event(event)
        if end_key is None:
            return
        if start_key == end_key:
            if end_key in selected_order:
                selected_order.remove(end_key)
            else:
                selected_order.append(end_key)
        else:
            for key in _line_cells(start_key, end_key):
                append_selection(key)
        redraw_after_selection_update()

    # Draw the empty panels, connect the callbacks and hand control to the user until the window closes.
    if show_analysis_panels:
        redraw_right_panels()
        redraw_lower_panels()
    reset_button.on_clicked(reset_selection)
    selector_button.on_clicked(cycle_map_quantity)
    wafer_button.on_clicked(select_wafer_region)
    ax_map.callbacks.connect("xlim_changed", on_map_view_changed)
    ax_map.callbacks.connect("ylim_changed", on_map_view_changed)
    fig.canvas.mpl_connect("button_press_event", on_mouse_press)
    fig.canvas.mpl_connect("button_release_event", on_mouse_release)
    plt.show()
    return MapSelection(reference_position=reference_key, selected_positions=tuple(selected_order))
