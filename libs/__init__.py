# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# SetkaFluo — __init__
# Lightweight utilities for XRF hyperspectral *denoising* and visualization.
#
# Copyright (c) 2025
# European Synchrotron Radiation Facility (ESRF), Grenoble, France.
# All rights reserved under the Creative Commons Attribution-NonCommercial 4.0
# International License (CC BY-NC 4.0). See https://creativecommons.org/licenses/by-nc/4.0/
#
# Non-commercial use only:
# This package is licensed for non-commercial research and educational use.
# Commercial use is expressly prohibited without prior written permission from ESRF.
#
# Please cite when using this software:
#   1. Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D, Cloetens P.
#      Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy
#      with Multi-Element Detectors. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-lsxpc
#
# Authors:
#   Lead developer:      Rodion Shishkov <rodion.shishkov@esrf.fr>
#   Reviewer/Maintainer: Dmitry Karpov  <dmitry.karpov@univ-grenoble-alpes.fr>
#
# Attribution:
#   © European Synchrotron Radiation Facility (ESRF). Please acknowledge ESRF in derived works.
#
# Disclaimer:
#   Provided “as is”, without warranty of any kind. The authors and ESRF are not liable for any
#   damages arising from the use of this software.

"""
SetkaFluo
=========

A Python package accompanying the work on *self-supervised denoising for
XRF microscopy with multi-element detectors*. It provides:

- Core data utilities for XRF hyperspectral arrays (shape management, spectral sums).
- Interactive visualization helpers used in notebooks to explore spectra and images.
- Denoising components and training utilities.

License
-------
Non-commercial use under CC BY-NC 4.0. For commercial licensing, contact the authors.

Citation
--------
If this package or its derivatives contribute to your research, please cite:

    Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D, Cloetens P.
    Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with
    Multi-Element Detectors. ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-lsxpc
"""

# Metadata
__author__            = "Rodion Shishkov"
__author_email__      = "rodion.shishkov@esrf.fr"
__maintainer__        = "Dmitry Karpov"
__maintainer_email__  = "dmitry.karpov@univ-grenoble-alpes.fr"
__license__           = "CC BY-NC 4.0"
__copyright__         = "© 2025 European Synchrotron Radiation Facility (ESRF)"
__version__           = "0.1.0"

# Public API re-exports (keep imports light and explicit)

# ---- data exploration helpers ----
from .data_explorer import (
    # shape / channels
    move_channels_last,
    reduce_channels,
    # spectra / sums
    spectrum_from_cube,
    sum_channels_window,
    csum_channels,
    sum_range_actual_from_csum,
    # display helpers
    to_display,
    compute_scale,
    shared_scale,
    figure_size_for_image,
    # IO helpers
    load_npz_cube_channels_last,
    list_detector_npz,
    list_weighted_tiffs,
    # notebook/GUI helpers
    make_spectrum_image_viewer,
    plot_channel_ranges_grid,
    make_detectors_viewer,
    make_weighted_tiff_browser,
)

# ---- denoising helpers ----
# Try to import TensorFlow-backed denoise utilities; fall back to a lazy error if TF is missing.
try:
    from .denoise import (
        DenoiseConfig,
        make_unet,
        make_dataset,
        train,
        predict_tiled,
        predict_all_checkpoints,
        save_tiff,
        load_tiff,
    )
    _DENOISE_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover
    _DENOISE_IMPORT_ERROR = _e
    def __getattr__(name):  # lazy, clear error only when needed
        denoise_names = {
            "DenoiseConfig", "make_unet", "make_dataset", "train",
            "predict_tiled", "predict_all_checkpoints", "save_tiff", "load_tiff"
        }
        if name in denoise_names:
            raise RuntimeError(
                "setkafluo.denoise could not be imported. "
                "Ensure TensorFlow (and its GPU runtime, if desired) is installed.\n\n"
                f"Original import error: {_DENOISE_IMPORT_ERROR}"
            )
        raise AttributeError(name)

__all__ = [
    # --- data_explorer ---
    # shape / channels
    "move_channels_last",
    "reduce_channels",
    # spectra / sums
    "spectrum_from_cube",
    "sum_channels_window",
    "csum_channels",
    "sum_range_actual_from_csum",
    # display helpers
    "to_display",
    "compute_scale",
    "shared_scale",
    "figure_size_for_image",
    # IO helpers
    "load_npz_cube_channels_last",
    "list_detector_npz",
    "list_weighted_tiffs",
    # notebook/GUI helpers
    "make_spectrum_image_viewer",
    "plot_channel_ranges_grid",
    "make_detectors_viewer",
    "make_weighted_tiff_browser",

    # --- denoise ---
    "DenoiseConfig",
    "make_unet",
    "make_dataset",
    "train",
    "predict_tiled",
    "predict_all_checkpoints",
    "save_tiff",
    "load_tiff",

    # metadata
    "__version__",
    "__license__",
    "__author__",
]