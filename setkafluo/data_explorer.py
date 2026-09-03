# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# SetkaFluo — data_explorer
# Utilities for interactive exploration of XRF hyperspectral data in Jupyter/Colab.
#
# Copyright (c) 2025
# European Synchrotron Radiation Facility (ESRF), Grenoble, France.
# All rights reserved under the Creative Commons Attribution-NonCommercial 4.0
# International License (CC BY-NC 4.0). See https://creativecommons.org/licenses/by-nc/4.0/
#
# Non-commercial use only:
# This software is licensed for non-commercial research and educational use.
# Commercial use is expressly prohibited without prior written permission from ESRF.
# For commercial licensing inquiries, please contact the authors below.
#
# Attribution & citation:
# If you use this software (or derivatives) in a publication or presentation, please
# acknowledge it and cite:
#   1. Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D, Cloetens P.
#      Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy
#      with Multi-Element Detectors. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-lsxpc
#
# Authors:
#   Lead developer:     Rodion Shishkov <rodion.shishkov@esrf.fr>
#   Reviewer/Maintainer: Dmitry Karpov <dmitry.karpov@univ-grenoble-alpes.fr>
#
# Disclaimer:
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a particular
# purpose and noninfringement. In no event shall the authors or ESRF be liable for any claim,
# damages or other liability, whether in an action of contract, tort or otherwise, arising
# from, out of or in connection with the software or the use or other dealings in the software.
#
# Third-party notices:
# This code depends on external libraries (NumPy, Matplotlib, ipywidgets, tifffile/imageio/PIL),
# each distributed under their own licenses.


"""Interactive viewers and utilities for XRF workshop.

Please acknowledge by citing:
  Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D, Cloetens P.
  Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors.
  ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-lsxpc

License: CC BY-NC 4.0 — © 2025 European Synchrotron Radiation Facility (ESRF).

Utilities used by the XRF workshop notebook to:
- Load hyperspectral cubes (NPZ) and reorder axes.
- Compute summed images over channel windows and spectra.
- Build fast, memory-lean interactive viewers in Jupyter/Colab:
  * Spectrum ↔ Image explorer for a chosen channel range.
  * Batch grid of selected channel windows.
  * Sum vs. individual detector browser.
  * Weighted TIFF (elemental area density) browser.

The functions are designed to be:
- Shape-explicit: cubes are always handled as (rows, cols, channels).
- Safe: many helpers validate dimensionality and clamp ranges.
- Notebook-friendly: viewers return ipywidgets containers (VBox) for display().

"""

from __future__ import annotations

from . import (
    __author__, __author_email__,
    __maintainer__, __maintainer_email__,
    __license__, __version__, __copyright__
)

import os, re, glob, math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Widgets & display
from ipywidgets import (
    IntRangeSlider, IntSlider, Button, Dropdown,
    HBox, VBox, Output, Layout, HTML
)
from IPython.display import display

# Preferred TIFF reader (tifffile → imageio.v3 → PIL fallback)
try:
    from tifffile import imread as TIFF_READ
except Exception:
    try:
        import imageio.v3 as iio
        TIFF_READ = iio.imread
    except Exception:
        from PIL import Image
        def TIFF_READ(p):  # fallback
            return np.array(Image.open(p))

# Matplotlib defaults
plt.rcParams.setdefault("figure.dpi", 100)

# =========================
# Shape / channel utilities
# =========================

def _ensure_3d(arr: np.ndarray):
    """Validate that `arr` is a 3D array.

    Used defensively across helpers that expect a hyperspectral cube.

    Parameters
    ----------
    arr : np.ndarray
        Array to validate.

    Raises
    ------
    ValueError
        If `arr.ndim != 3`.
    """
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D array, got shape {arr.shape}")

def channel_axis(arr: np.ndarray) -> int:
    """Return the index of the channel axis (heuristic).

    The workshop datasets typically have the spectral dimension as the
    largest axis (e.g., 2048 channels). This heuristic picks the axis
    with the maximum size.

    Parameters
    ----------
    arr : np.ndarray
        3D array with unknown axis order.

    Returns
    -------
    int
        Index of the axis assumed to be channels.

    Notes
    -----
    If your data doesn't follow this convention, reorder your array
    explicitly before passing it to downstream functions.
    """
    _ensure_3d(arr)
    return int(np.argmax(arr.shape))

def move_channels_last(arr: np.ndarray) -> np.ndarray:
    """Return `arr` with channels moved to the last position.

    Shape becomes (rows, cols, channels). This is the canonical shape
    used by the notebook for summing and visualization.

    Parameters
    ----------
    arr : np.ndarray
        3D input array with arbitrary axis order.

    Returns
    -------
    np.ndarray
        View of `arr` with channels-last layout.
    """
    _ensure_3d(arr)
    return np.moveaxis(arr, channel_axis(arr), -1)

def clamp_channel_range(lo: int, hi: int, n_channels: int) -> tuple[int, int]:
    """Clamp a (possibly unordered) [lo, hi] inclusive range to valid channel indices.

    Parameters
    ----------
    lo, hi : int
        Requested channel bounds (order-agnostic).
    n_channels : int
        Total number of channels.

    Returns
    -------
    (int, int)
        Clamped (lo_eff, hi_eff) satisfying 0 ≤ lo_eff ≤ hi_eff ≤ n_channels-1.
    """
    a = int(min(lo, hi)); b = int(max(lo, hi))
    a = max(0, a); b = min(n_channels - 1, b)
    return (a, b)

# =========================
# Summation / spectra
# =========================

def sum_channels_window(arr_rcz: np.ndarray, lo: int, hi: int,
                        out_dtype=np.int64) -> tuple[np.ndarray, tuple[int,int]]:
    """Sum a channel window (inclusive) into a 2D image.

    This is the core operation behind the workshop’s “channel sum image”.
    It expects channels-last arrays: (rows, cols, channels).

    Parameters
    ----------
    arr_rcz : np.ndarray
        Hyperspectral cube with shape (rows, cols, channels).
    lo, hi : int
        Requested channel window (inclusive, order-agnostic).
    out_dtype : dtype, optional
        Accumulator dtype for the sum. Use np.int64 for safety across large sums,
        or np.float32 to reduce memory.

    Returns
    -------
    img_rc : np.ndarray
        2D image (rows, cols) of the summed counts.
    (lo_eff, hi_eff) : tuple[int, int]
        Effective clamped bounds used for the sum.

    Notes
    -----
    This function clamps the range to available channels; if it collapses
    (lo_eff > hi_eff) it returns a zeros image.
    """
    _ensure_3d(arr_rcz)
    rows, cols, chn = arr_rcz.shape
    lo_eff, hi_eff = clamp_channel_range(lo, hi, chn)
    if lo_eff > hi_eff:
        return np.zeros((rows, cols), dtype=out_dtype), (lo_eff, hi_eff)
    img = arr_rcz[:, :, lo_eff:hi_eff+1].sum(axis=2, dtype=out_dtype)
    return img, (lo_eff, hi_eff)

def spectrum_from_cube(arr_rcz: np.ndarray) -> np.ndarray:
    """Compute a summed spectrum across all pixels.

    Parameters
    ----------
    arr_rcz : np.ndarray
        (rows, cols, channels) cube.

    Returns
    -------
    np.ndarray
        1D array of length `channels` with summed counts.

    Used in the notebook to draw the “Summed Spectrum (log-y)” preview.
    """
    _ensure_3d(arr_rcz)
    return arr_rcz.sum(axis=(0, 1))

def csum_channels(arr_rcz: np.ndarray, dtype=np.float64) -> np.ndarray:
    """Compute a cumulative sum along the channel axis.

    Parameters
    ----------
    arr_rcz : np.ndarray
        (rows, cols, channels) cube.
    dtype : dtype, optional
        Accumulator dtype for cumsum (np.float64 by default).

    Returns
    -------
    np.ndarray
        Same shape as input, where `csum[:,:,k]` equals sum of channels [0..k].

    Notes
    -----
    Not used in the current interactive cell (we recompute per range),
    but useful when you need *many* range sums with minimal recomputation.
    """
    _ensure_3d(arr_rcz)
    return arr_rcz.cumsum(axis=2, dtype=dtype)

def sum_range_actual_from_csum(csum: np.ndarray, ch_lo_global: int, ch_hi_global: int,
                               lo_actual: int, hi_actual: int) -> np.ndarray | None:
    """Sum an *actual* channel range using a precomputed cumulative sum.

    Parameters
    ----------
    csum : np.ndarray
        Cumulative-sum cube over a globally reduced channel window.
    ch_lo_global, ch_hi_global : int
        Global channel bounds used to compute `csum` in the first place.
    lo_actual, hi_actual : int
        Requested *actual* channel numbers (inclusive).

    Returns
    -------
    np.ndarray or None
        2D image (rows, cols) of the requested sum, or None if the range
        falls completely outside [ch_lo_global, ch_hi_global].

    Example
    -------
    In the notebook, you could precompute:

        c = csum_channels(arr_red)
        img = sum_range_actual_from_csum(c, CH_LO, CH_HI, 899, 944)

    to avoid re-summing the raw cube repeatedly.
    """
    a = int(min(lo_actual, hi_actual)); b = int(max(lo_actual, hi_actual))
    if b < ch_lo_global or a > ch_hi_global:
        return None
    a = max(a, ch_lo_global); b = min(b, ch_hi_global)
    lo = a - ch_lo_global; hi = b - ch_lo_global
    return csum[:, :, hi] if lo == 0 else (csum[:, :, hi] - csum[:, :, lo - 1])

def reduce_channels(arr_rcz: np.ndarray, ch_lo: int, ch_hi: int):
    """Slice channels [ch_lo, ch_hi] (inclusive) and return both the subcube and channel numbers.

    Parameters
    ----------
    arr_rcz : np.ndarray
        (rows, cols, channels) cube.
    ch_lo, ch_hi : int
        Requested channel bounds (must be valid; no clamping here).

    Returns
    -------
    arr_red : np.ndarray
        Reduced cube with shape (rows, cols, ch_hi-ch_lo+1).
    channel_numbers : np.ndarray
        1D array of length `ch_hi-ch_lo+1` with the actual channel indices.

    Raises
    ------
    ValueError
        If the requested window is out of bounds.

    Notes
    -----
    This strict behavior is intentional in the notebook to catch mistakes
    early when selecting the global analysis window.
    """
    _ensure_3d(arr_rcz)
    rows, cols, chn = arr_rcz.shape
    a, b = clamp_channel_range(ch_lo, ch_hi, chn)
    if a != ch_lo or b != ch_hi:
        raise ValueError(f"Requested channel window [{ch_lo}, {ch_hi}] out of bounds 0..{chn-1}")
    arr_red = arr_rcz[:, :, a:b+1]
    channel_numbers = np.arange(a, b + 1)
    return arr_red, channel_numbers

# =========================
# Display helpers
# =========================

def to_display(img_rc: np.ndarray, *, log1p: bool = False, transpose: bool = True) -> np.ndarray:
    """Prepare a 2D image for on-screen display.

    Converts to float32, optionally applies `log1p` for contrast, and
    optionally transposes so that x-axis corresponds to rows (geometry
    consistent with the notebook’s visual conventions).

    Parameters
    ----------
    img_rc : np.ndarray
        2D image (rows, cols).
    log1p : bool, optional
        Apply log1p transform in-place (on a float32 view).
    transpose : bool, optional
        If True, return (cols, rows) for display.

    Returns
    -------
    np.ndarray
        Prepared image for `imshow`.
    """
    arr = img_rc.astype(np.float32, copy=False)
    if log1p:
        np.log1p(arr, out=arr)
    return arr.T if transpose else arr

def compute_scale(img2d: np.ndarray, mode: str = "full", p_lo: float = 2.0, p_hi: float = 98.0):
    """Compute color limits for `imshow`.

    Parameters
    ----------
    img2d : np.ndarray
        2D image to analyze (already in display space if using log1p).
    mode : {"full", "percentile"}
        Full min–max or percentile scaling.
    p_lo, p_hi : float
        Percentiles for "percentile" mode.

    Returns
    -------
    (float, float)
        (vmin, vmax) suitable for `imshow(vmin=..., vmax=...)`.

    Notes
    -----
    Degenerate ranges (vmin == vmax or NaNs) are guarded against by
    inflating vmax slightly.
    """
    if mode == "full":
        vmin = float(np.nanmin(img2d)); vmax = float(np.nanmax(img2d))
    else:
        vmin, vmax = np.nanpercentile(img2d, [p_lo, p_hi])
        vmin = float(vmin); vmax = float(vmax)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax

def shared_scale(images: List[np.ndarray], mode="full", p_lo=2.0, p_hi=98.0):
    """Compute a single (vmin, vmax) shared across multiple images.

    Useful for fair comparisons (e.g., across detectors for a given range).

    Parameters
    ----------
    images : list of np.ndarray
        List of 2D images already in display space (log1p if desired).
    mode, p_lo, p_hi : see `compute_scale`.

    Returns
    -------
    (float, float)
        Shared (vmin, vmax).
    """
    vmins, vmaxs = [], []
    for im in images:
        vmin, vmax = compute_scale(im, mode=mode, p_lo=p_lo, p_hi=p_hi)
        vmins.append(vmin); vmaxs.append(vmax)
    vmin = min(vmins); vmax = max(vmaxs)
    if vmin == vmax: vmax = vmin + 1.0
    return vmin, vmax

def figure_size_for_image(ny: int, nx: int, img_width_in: float,
                          spec_width_in: float, cbar_width_in: float, margin_w: float):
    """Compute a figure size that preserves pixel geometry and aligns subplots vertically.

    Parameters
    ----------
    ny, nx : int
        Displayed image height/width (after transpose if any).
    img_width_in : float
        Target width of the image panel.
    spec_width_in : float
        Width of the spectrum panel (if present).
    cbar_width_in : float
        Width of the colorbar panel (if present).
    margin_w : float
        Extra horizontal margin.

    Returns
    -------
    (float, float)
        (fig_width, fig_height) in inches.
    """
    aspect = ny / nx
    fig_height = max(4.0, aspect * img_width_in)
    fig_width  = spec_width_in + img_width_in + cbar_width_in + margin_w
    return fig_width, fig_height

# =========================
# File helpers (NPZ, TIFF)
# =========================

def load_npz_cube_channels_last(npz_path: Path) -> np.ndarray:
    """Load a `.npz` file with key `'cube'` and return (rows, cols, channels).

    Parameters
    ----------
    npz_path : Path
        Path to an NPZ file containing a 3D array under the key `"cube"`.

    Returns
    -------
    np.ndarray
        The cube re-ordered to channels-last.

    Raises
    ------
    KeyError
        If the NPZ does not contain a `"cube"` key.
    ValueError
        If the loaded array is not 3D.
    """
    with np.load(npz_path) as npz:
        cube = npz["cube"]
    return move_channels_last(cube)

def list_detector_npz(data_dir: Path) -> List[Tuple[int, str, str]]:
    """List detector cubes in `data_dir`.

    Matches `detector_element_*.npz` and extracts the numeric ID.

    Parameters
    ----------
    data_dir : Path
        Directory to scan.

    Returns
    -------
    list of (det_id_int, det_id_str, path)
        Sorted by integer detector ID (e.g., 1, 4, 7).

    Notes
    -----
    Used by the “Detector-element explorer” cell in the notebook.
    """
    det_entries = []
    det_re = re.compile(r"detector_element_(\d+)\.npz$", re.IGNORECASE)
    for p in sorted(glob.glob(str(data_dir / "detector_element_*.npz"))):
        m = det_re.search(os.path.basename(p))
        if m:
            det_entries.append((int(m.group(1)), m.group(1), p))
    det_entries.sort(key=lambda t: t[0])
    return det_entries

def list_weighted_tiffs(weights_dir: Path, patterns=("IMG_weighted_*_area_density_ngmm2.tif",
                                                     "IMG_weighted_*_area_density_ngmm2.tiff")):
    """List PyMca-produced elemental area density maps.

    Parameters
    ----------
    weights_dir : Path
        Directory with weighted TIFFs.
    patterns : tuple of str
        Filename patterns to match.

    Returns
    -------
    list of (element_name, path)
        Sorted by element name parsed from filename.

    Notes
    -----
    These maps are quantitative (e.g., ng/mm²) and come from non-linear
    spectral fitting. The browser in the notebook displays them one by one
    with per-image (full-range) color scaling.
    """
    paths = []
    for pat in patterns:
        paths.extend(glob.glob(str(weights_dir / pat)))
    paths = sorted(paths)
    def parse_element(p):
        base = os.path.basename(p)
        m = re.search(r"IMG_weighted_(.+?)_area_density_ngmm2", base, flags=re.IGNORECASE)
        return m.group(1) if m else base
    entries = [(parse_element(p), p) for p in paths]
    entries.sort(key=lambda t: t[0].lower())
    return entries

# =========================
# Reusable UI builders
# =========================

def make_spectrum_image_viewer(arr, spectrum, chan_vals, ch_lo, ch_hi,
                               img_width_in, spec_width_in_base, spectrum_scale,
                               cbar_width_in, margin_w):
    """Build the interactive Spectrum ↔ Image viewer (one figure, cached artists).

    This viewer powers the cell where participants drag a channel window
    across the spectrum and immediately see the corresponding summed image.

    Parameters
    ----------
    arr : np.ndarray
        Reduced cube (rows, cols, channels) within [ch_lo, ch_hi].
    spectrum : np.ndarray
        1D summed spectrum for the same reduced cube.
    chan_vals : np.ndarray
        1D array of actual channel numbers (e.g., 100..1700).
    ch_lo, ch_hi : int
        Global bounds of the reduced cube in original channel numbering.
    img_width_in, spec_width_in_base, spectrum_scale, cbar_width_in, margin_w : float
        Layout parameters used to size the figure and sliders.

    Returns
    -------
    ipywidgets.VBox
        A VBox containing (Channels slider, Center slider, Output with figure).

    Notes
    -----
    - We compute sums on demand using a reusable buffer to avoid memory churn.
    - Color limits are recomputed per update (full range) to reflect the window.
    - The image is shown with preserved pixel geometry and log1p contrast.
    """
    rows, cols, channels_red = arr.shape
    SPEC_WIDTH_IN = spec_width_in_base * spectrum_scale

    # Sliders sized to spectrum width
    DPI = plt.rcParams.get("figure.dpi", 100)
    SPEC_WIDTH_PX = int(SPEC_WIDTH_IN * DPI)
    slider_layout = Layout(width=f"{SPEC_WIDTH_PX}px")
    slider_style  = {"description_width": "85px"}

    range_slider = IntRangeSlider(
        value=(int(chan_vals[0]), int(chan_vals[-1])),
        min=int(chan_vals[0]), max=int(chan_vals[-1]), step=1,
        description="Channels", continuous_update=False,
        layout=slider_layout, style=slider_style
    )
    center_slider = IntSlider(
        value=int((chan_vals[0] + chan_vals[-1]) // 2),
        min=int(chan_vals[0]), max=int(chan_vals[-1]), step=1,
        description="Center", continuous_update=False,
        layout=slider_layout, style=slider_style
    )

    out = Output()
    _updating = {"flag": False}

    # Reusable buffers
    buf_sum  = np.empty((rows, cols), dtype=np.int64)
    buf_disp = np.empty((rows, cols), dtype=np.float32)

    def _sync_center_from_range():
        lo, hi = range_slider.value
        center_slider.value = int((lo + hi) // 2)

    def _move_range_from_center():
        lo, hi = range_slider.value
        width = hi - lo
        c = center_slider.value
        new_lo = c - width // 2
        new_hi = new_lo + width
        if new_lo < ch_lo:
            new_lo, new_hi = ch_lo, ch_lo + width
        if new_hi > ch_hi:
            new_hi, new_lo = ch_hi, ch_hi - width
        range_slider.value = (int(new_lo), int(new_hi))

    # Initial sum for sizing/first draw
    lo0, hi0 = range_slider.value
    lo_idx = lo0 - ch_lo
    hi_idx = hi0 - ch_lo
    np.sum(arr[:, :, lo_idx:hi_idx+1], axis=2, dtype=buf_sum.dtype, out=buf_sum)
    buf_disp[:] = buf_sum
    np.log1p(buf_disp, out=buf_disp)
    img0_T = buf_disp.T
    ny, nx = img0_T.shape

    fig_w, fig_h = figure_size_for_image(ny, nx, img_width_in, SPEC_WIDTH_IN, cbar_width_in, margin_w)

    with plt.ioff():
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(1, 3, width_ratios=[SPEC_WIDTH_IN, img_width_in, cbar_width_in], wspace=0.25)

        # Spectrum
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(chan_vals, spectrum)
        ax1.set_yscale("log")
        ax1.set_xlabel("Channel number")
        ax1.set_ylabel("Intensity counts")
        ax1.set_title("Spectrum (log)")
        span = ax1.axvspan(lo0, hi0, alpha=0.2)

        # Image
        ax2 = fig.add_subplot(gs[1])
        im = ax2.imshow(img0_T, origin="upper", extent=[0, nx, ny, 0], interpolation="nearest")
        ax2.set_aspect('equal')
        ax2.set_title(f"Image summed over channels [{lo0}, {hi0}]")
        ax2.set_xlabel("Row index")
        ax2.set_ylabel("Column index")

        # Colorbar
        cax = fig.add_subplot(gs[2])
        cbar = fig.colorbar(im, cax=cax, label="Counts (log1p)")

    def _update_plot(lo_actual, hi_actual):
        # Update highlight
        nonlocal span
        try:
            span.remove()
        except Exception:
            pass
        span = ax1.axvspan(lo_actual, hi_actual, alpha=0.2)

        # Sum → buffers → image
        lo_idx = lo_actual - ch_lo
        hi_idx = hi_actual - ch_lo
        np.sum(arr[:, :, lo_idx:hi_idx+1], axis=2, dtype=buf_sum.dtype, out=buf_sum)
        buf_disp[:] = buf_sum
        np.log1p(buf_disp, out=buf_disp)

        im.set_data(buf_disp.T)
        vmin = float(buf_disp.min()); vmax = float(buf_disp.max())
        if vmin == vmax: vmax = vmin + 1.0
        im.set_clim(vmin=vmin, vmax=vmax)
        cbar.update_normal(im)
        ax2.set_title(f"Image summed over channels [{lo_actual}, {hi_actual}]")

    def render():
        lo_actual, hi_actual = range_slider.value
        _update_plot(lo_actual, hi_actual)
        with out:
            out.clear_output(wait=True)
            display(fig)

    def _on_range_change(change):
        if change["name"] != "value" or _updating["flag"]:
            return
        _updating["flag"] = True
        try:
            _sync_center_from_range()
            render()
        finally:
            _updating["flag"] = False

    def _on_center_change(change):
        if change["name"] != "value" or _updating["flag"]:
            return
        _updating["flag"] = True
        try:
            _move_range_from_center()
            render()
        finally:
            _updating["flag"] = False

    range_slider.observe(_on_range_change, names="value")
    center_slider.observe(_on_center_change, names="value")

    _sync_center_from_range()
    render()

    return VBox([range_slider, center_slider, out])

def plot_channel_ranges_grid(arr, ranges, ch_bounds, *,
                             n_cols=2, tile_width_in=5.0, use_log1p=False,
                             scale_mode="full", p_lo=2.0, p_hi=98.0):
    """Display a grid of images, each a sum over a specified channel window.

    This powers the “Batch visualization of selected channel ranges”
    cell in the notebook.

    Parameters
    ----------
    arr : np.ndarray
        Reduced cube (rows, cols, channels).
    ranges : list[tuple[int,int]]
        List of (lo, hi) channel pairs in *actual* channel numbers.
    ch_bounds : tuple[int, int]
        Global (CH_LO, CH_HI) used for `arr`; used to clamp/offset indices.
    n_cols : int
        Columns in the grid layout.
    tile_width_in : float
        Width of each image tile (inches).
    use_log1p : bool
        Apply log1p before display.
    scale_mode : {"full", "percentile"}
        Per-tile color scaling mode.
    p_lo, p_hi : float
        Percentiles for "percentile" mode.

    Returns
    -------
    None
        Displays the figure and closes it (keeps static output in the notebook).
    """
    rows, cols, _ = arr.shape
    ny, nx = cols, rows
    aspect = ny / nx

    n = len(ranges)
    nrows = math.ceil(n / n_cols)

    fig_w = n_cols * tile_width_in + (n_cols - 1) * 0.35 + 0.8
    fig_h = nrows * (aspect * tile_width_in) + (nrows - 1) * 0.4 + 0.6

    with plt.ioff():
        fig, axes = plt.subplots(nrows, n_cols, figsize=(fig_w, fig_h), constrained_layout=True)
        if nrows == 1 and n_cols == 1: axes = np.array([[axes]])
        elif nrows == 1: axes = np.array([axes])
        elif n_cols == 1: axes = axes[:, np.newaxis]

        buf = np.empty((rows, cols), dtype=np.int64)
        CH_MIN, CH_MAX = ch_bounds

        for idx, (a, b) in enumerate(ranges):
            r, c = divmod(idx, n_cols)
            ax = axes[r, c]
            lo = max(min(a, b), CH_MIN) - CH_MIN
            hi = min(max(a, b), CH_MAX) - CH_MIN
            if lo > hi:
                ax.axis("off"); continue

            np.sum(arr[:, :, lo:hi+1], axis=2, dtype=buf.dtype, out=buf)
            img_T = to_display(buf, log1p=use_log1p, transpose=True)

            vmin, vmax = compute_scale(img_T, mode=scale_mode, p_lo=p_lo, p_hi=p_hi)

            im = ax.imshow(img_T, origin="upper", extent=[0, nx, ny, 0],
                           interpolation="nearest", vmin=vmin, vmax=vmax)
            ax.set_box_aspect(aspect)
            ax.set_title(f"Channels {a}–{b}")

            if r < nrows - 1: ax.set_xticklabels([])
            else: ax.set_xlabel("Row index")
            if c > 0: ax.set_yticklabels([])
            else: ax.set_ylabel("Column index")

            cb = fig.colorbar(im, ax=ax, location="right", pad=0.02)
            cb.set_label("Counts (log1p)" if use_log1p else "Counts")

        # Hide empties
        for j in range(n, nrows*n_cols):
            r, c = divmod(j, n_cols); axes[r, c].axis("off")

        display(fig)
        plt.close(fig)

def make_detectors_viewer(data_dir: Path, ranges, *, use_log1p=True, img_width_in=7.0):
    """Build the Sum vs. Individual Detectors viewer.

    Loads `sum_spectrum.npz` and all `detector_element_*.npz` cubes, precomputes
    channel-summed images for each range, and provides a fast UI to compare
    detectors on a shared color scale (per range).

    Parameters
    ----------
    data_dir : Path
        Folder containing NPZ files (`sum_spectrum.npz` and detectors).
    ranges : list[tuple[int,int]]
        Channel windows in actual channel numbers.
    use_log1p : bool
        Apply log1p to all images for display.
    img_width_in : float
        Width of each figure panel.

    Returns
    -------
    ipywidgets.VBox
        A VBox with:
          - a Range index slider + label,
          - the sum image figure,
          - Prev/Next buttons + detector label,
          - the detector image figure.

    Notes
    -----
    - Images are computed once, then reused; switching is instantaneous.
    - Each range uses a *shared* (vmin, vmax) across detectors for fair comparison.
    """
    sum_path = data_dir / "sum_spectrum.npz"
    if not sum_path.exists():
        raise FileNotFoundError(f"Missing file: {sum_path}")

    det_entries = list_detector_npz(data_dir)
    if not det_entries:
        raise FileNotFoundError(f"No detector NPZ files found in {data_dir}")

    # Precompute images
    range_count = len(ranges)
    det_count   = len(det_entries)

    sum_imgs = [None] * range_count                     # (cols, rows)
    det_imgs_by_range = [[None]*det_count for _ in range(range_count)]

    # Sum spectrum once
    arr_sum = load_npz_cube_channels_last(sum_path)
    for ri, (a, b) in enumerate(ranges):
        img, _ = sum_channels_window(arr_sum, a, b, out_dtype=np.float32)
        sum_imgs[ri] = to_display(img, log1p=use_log1p, transpose=True)
    del arr_sum

    # Detectors
    for di, (_, det_str, det_path) in enumerate(det_entries):
        arr_det = load_npz_cube_channels_last(Path(det_path))
        for ri, (a, b) in enumerate(ranges):
            img, _ = sum_channels_window(arr_det, a, b, out_dtype=np.float32)
            det_imgs_by_range[ri][di] = to_display(img, log1p=use_log1p, transpose=True)
        del arr_det

    # Shared detector scales per range
    det_scales = [shared_scale(det_imgs_by_range[ri], mode="full") for ri in range(range_count)]

    # UI + figures
    range_idx = IntSlider(value=0, min=0, max=range_count-1, step=1,
                          description="Range idx", continuous_update=False)
    btn_prev = Button(description="◀ Prev", disabled=(det_count <= 1))
    btn_next = Button(description="Next ▶", disabled=(det_count <= 1))
    det_label = HTML("")
    range_label = HTML("")
    out_sum = Output()
    out_det = Output()

    # Build both figures once
    ri0 = range_idx.value
    ny_sum, nx_sum = sum_imgs[ri0].shape
    with plt.ioff():
        fig_sum = plt.figure(figsize=(img_width_in + 0.8, max(3.5, (ny_sum/nx_sum)*img_width_in)))
        ax_sum  = fig_sum.add_subplot(1,1,1)
        im_sum  = ax_sum.imshow(sum_imgs[ri0], origin="upper",
                                extent=[0, nx_sum, ny_sum, 0], interpolation="nearest")
        ax_sum.set_aspect('equal')
        a0,b0 = ranges[ri0]
        ax_sum.set_title(f"Sum spectrum — channels {a0}–{b0}")
        ax_sum.set_xlabel("Row index"); ax_sum.set_ylabel("Column index")
        cb_sum = fig_sum.colorbar(im_sum, ax=ax_sum, location="right", pad=0.02)
        cb_sum.set_label("Counts (log1p)" if use_log1p else "Counts")

    ny_det, nx_det = det_imgs_by_range[ri0][0].shape
    vmin0, vmax0 = det_scales[ri0]
    with plt.ioff():
        fig_det = plt.figure(figsize=(img_width_in + 0.8, max(3.5, (ny_det/nx_det)*img_width_in)))
        ax_det  = fig_det.add_subplot(1,1,1)
        im_det  = ax_det.imshow(det_imgs_by_range[ri0][0], origin="upper",
                                extent=[0, nx_det, ny_det, 0], interpolation="nearest",
                                vmin=vmin0, vmax=vmax0)
        ax_det.set_aspect('equal')
        det_id0, det_str0, _ = det_entries[0]
        ax_det.set_title(f"Detector {det_str0} (1/{det_count}) — channels {a0}–{b0}")
        ax_det.set_xlabel("Row index"); ax_det.set_ylabel("Column index")
        cb_det = fig_det.colorbar(im_det, ax=ax_det, location="right", pad=0.02)
        cb_det.set_label("Counts (log1p)" if use_log1p else "Counts")

    cur_det_idx = {"i": 0}

    def _refresh_titles_and_labels():
        ri = range_idx.value
        a,b = ranges[ri]
        ax_sum.set_title(f"Sum spectrum — channels {a}–{b}")
        _, det_str, _ = det_entries[cur_det_idx["i"]]
        ax_det.set_title(f"Detector {det_str} ({cur_det_idx['i']+1}/{det_count}) — channels {a}–{b}")
        det_label.value = f"<b>Detector {det_str}</b> ({cur_det_idx['i']+1}/{det_count})"
        range_label.value = f"&nbsp;&nbsp;Current range: <b>{a}–{b}</b>"

    def _show_figs():
        with out_sum:
            out_sum.clear_output(wait=True); display(fig_sum)
        with out_det:
            out_det.clear_output(wait=True); display(fig_det)

    def _on_range_change(change):
        if change["name"] != "value": return
        ri = range_idx.value
        # sum
        im_sum.set_data(sum_imgs[ri])
        vmin, vmax = compute_scale(sum_imgs[ri], mode="full")
        im_sum.set_clim(vmin=vmin, vmax=vmax); cb_sum.update_normal(im_sum)
        # reset detector
        cur_det_idx["i"] = 0
        vmin, vmax = det_scales[ri]
        im_det.set_data(det_imgs_by_range[ri][cur_det_idx["i"]])
        im_det.set_clim(vmin=vmin, vmax=vmax); cb_det.update_normal(im_det)
        _refresh_titles_and_labels()
        _show_figs()

    def _on_prev(_):
        ri = range_idx.value
        cur_det_idx["i"] = (cur_det_idx["i"] - 1) % det_count
        im_det.set_data(det_imgs_by_range[ri][cur_det_idx["i"]])
        vmin, vmax = det_scales[ri]
        im_det.set_clim(vmin=vmin, vmax=vmax); cb_det.update_normal(im_det)
        _refresh_titles_and_labels()
        _show_figs()

    def _on_next(_):
        ri = range_idx.value
        cur_det_idx["i"] = (cur_det_idx["i"] + 1) % det_count
        im_det.set_data(det_imgs_by_range[ri][cur_det_idx["i"]])
        vmin, vmax = det_scales[ri]
        im_det.set_clim(vmin=vmin, vmax=vmax); cb_det.update_normal(im_det)
        _refresh_titles_and_labels()
        _show_figs()

    range_idx.observe(_on_range_change, names="value")
    btn_prev.on_click(_on_prev)
    btn_next.on_click(_on_next)

    _refresh_titles_and_labels()
    _show_figs()

    return VBox([
        HBox([range_idx, range_label]),
        out_sum,
        HBox([btn_prev, det_label, btn_next]),
        out_det
    ])

def make_weighted_tiff_browser(weights_dir: Path, patterns, *, use_log1p=False, img_width_in=7.0):
    """Build a browser for fitted elemental area density maps (PyMca outputs).

    Scans for `IMG_weighted_*_area_density_ngmm2.tif[f]`, loads all 2D maps,
    applies per-image full-range color scaling (no shared normalization),
    and provides a quick dropdown / Prev–Next interface.

    Parameters
    ----------
    weights_dir : Path
        Directory containing the weighted TIFF maps.
    patterns : tuple[str, ...]
        Filename patterns to match (default includes .tif and .tiff).
    use_log1p : bool
        Apply log1p before display (useful for large dynamic range).
    img_width_in : float
        Figure width per image.

    Returns
    -------
    ipywidgets.VBox
        A VBox with (dropdown + buttons + title) and a figure output.

    Notes
    -----
    The values are quantitative (e.g., ng/mm²) and comparable *within*
    the same dataset and processing configuration. This browser is meant
    to visually inspect spatial patterns across elements.
    """
    entries = list_weighted_tiffs(weights_dir, patterns)
    if not entries:
        raise FileNotFoundError(f"No TIFF files found in {weights_dir} with patterns {patterns}")

    imgs, elements = [], []
    for el, p in entries:
        arr = TIFF_READ(p)
        arr = np.asarray(arr)
        if arr.ndim > 2: arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"{os.path.basename(p)}: expected 2D map, got shape {arr.shape}")
        imgs.append(to_display(arr, log1p=use_log1p, transpose=False))
        elements.append(el)

    n = len(imgs)
    ny, nx = imgs[0].shape
    for i, im_ in enumerate(imgs[1:], start=1):
        if im_.shape != (ny, nx):
            raise ValueError(f"Image shape mismatch at {entries[i][1]}: {im_.shape} != {(ny, nx)}")

    scales = [compute_scale(im_, mode="full") for im_ in imgs]

    idx = {"i": 0}
    dropdown = Dropdown(options=[(elements[i], i) for i in range(n)], value=0, description="Element")
    btn_prev = Button(description="◀ Prev", disabled=(n <= 1))
    btn_next = Button(description="Next ▶", disabled=(n <= 1))
    title = HTML("")
    out = Output()

    def _set_title(i):
        title.value = f"<b>{elements[i]}</b> ({i+1}/{n})"

    with plt.ioff():
        fig = plt.figure(figsize=(img_width_in + 0.8, max(3.5, (ny/nx) * img_width_in)))
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(imgs[idx["i"]], origin="upper", extent=[0, nx, ny, 0], interpolation="nearest",
                       vmin=scales[idx["i"]][0], vmax=scales[idx["i"]][1])
        ax.set_aspect('equal')
        ax.set_title(f"{elements[idx['i']]} area density (ng/mm²)")
        ax.set_xlabel("Column index")
        ax.set_ylabel("Row index")
        cbar = fig.colorbar(im, ax=ax, location="right", pad=0.02)
        cbar.set_label("ng/mm² (log1p)" if use_log1p else "ng/mm²")

    def _update(i):
        i = max(0, min(i, n-1))
        im.set_data(imgs[i])
        im.set_clim(*scales[i])
        ax.set_title(f"{elements[i]} area density (ng/mm²)")
        _set_title(i)
        with out:
            out.clear_output(wait=True)
            display(fig)

    def _on_prev(_):
        idx["i"] = (idx["i"] - 1) % n
        dropdown.value = idx["i"]
        _update(idx["i"])

    def _on_next(_):
        idx["i"] = (idx["i"] + 1) % n
        dropdown.value = idx["i"]
        _update(idx["i"])

    def _on_select(change):
        if change["name"] != "value": return
        idx["i"] = int(change["new"])
        _update(idx["i"])

    btn_prev.on_click(_on_prev)
    btn_next.on_click(_on_next)
    dropdown.observe(_on_select, names="value")

    _set_title(idx["i"])
    with out:
        out.clear_output(wait=True)
        display(fig)
    _update(idx["i"])

    return VBox([
        HBox([dropdown, btn_prev, btn_next, title]),
        out
    ])