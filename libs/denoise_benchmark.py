# -*- coding: utf-8 -*-
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# SetkaFluo — denoise
# Noise2Noise-based denoising for XRF images with multi-element detectors (TensorFlow/Keras).
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
#   Lead developer:      Rodion Shishkov <rodion.shishkov@esrf.fr>
#   Reviewer/Maintainer: Dmitry Karpov  <dmitry.karpov@univ-grenoble-alpes.fr>
#
# Disclaimer:
# This software is provided "as is", without warranty of any kind, express or implied,
# including but not limited to the warranties of merchantability, fitness for a particular
# purpose and noninfringement. In no event shall the authors or ESRF be liable for any claim,
# damages or other liability, whether in an action of contract, tort or otherwise, arising
# from, out of or in connection with the software or the use or other dealings in the software.
#
# Third-party notices:
# This code depends on external libraries (TensorFlow/Keras, NumPy, tifffile, Matplotlib),
# each distributed under their own licenses.

"""XRF denoising utilities (Noise2Noise)

Please acknowledge by citing:
  Shishkov R, Laugros A, Vigano N, Bohic S, Karpov D, Cloetens P.
  Self-Supervised Deep-Learning Denoising for X-Ray Fluorescence Microscopy with Multi-Element Detectors.
  ChemRxiv. 2025; doi:10.26434/chemrxiv-2025-lsxpc

License: CC BY-NC 4.0 — © 2025 European Synchrotron Radiation Facility (ESRF).

"""

from __future__ import annotations

from . import (
    __author__, __author_email__,
    __maintainer__, __maintainer_email__,
    __license__, __version__, __copyright__
)

import time
from dataclasses import replace
from typing import Dict, Any, Optional, Tuple, Iterable

import numpy as np
import tensorflow as tf

# Import your training bits
from .denoise import (
    DenoiseConfig,
    make_unet,
    make_dataset,
    mse_loss,
    PSNR,
)

__all__ = [
    "clone_cfg",
    "ensure_compiled_model",
    "benchmark_steps",
]

# -----------------------
# Small utilities
# -----------------------

def clone_cfg(cfg: DenoiseConfig,
              *,
              lr: Optional[float] = None,
              batch_size: Optional[int] = None,
              patch_size: Optional[int] = None,
              steps_per_epoch: Optional[int] = None,
              epochs: Optional[int] = None,
              min_overlap: Optional[int] = None) -> DenoiseConfig:
    """Return a shallow copy of cfg with any provided overrides."""
    kwargs = {}
    if lr is not None: kwargs["lr"] = lr
    if batch_size is not None: kwargs["batch_size"] = int(batch_size)
    if patch_size is not None: kwargs["patch_size"] = int(patch_size)
    if steps_per_epoch is not None: kwargs["steps_per_epoch"] = int(steps_per_epoch)
    if epochs is not None: kwargs["epochs"] = int(epochs)
    if min_overlap is not None: kwargs["min_overlap"] = int(min_overlap)
    return replace(cfg, **kwargs)


def ensure_compiled_model(cfg: DenoiseConfig, model: Optional[tf.keras.Model] = None) -> tf.keras.Model:
    """
    Use the provided model if it looks like a Keras Model; otherwise build
    your U-Net and compile it with the same loss/metrics as train().
    """
    if not isinstance(model, tf.keras.Model):
        model = make_unet(cfg)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
                      loss=mse_loss, metrics=[PSNR])
    else:
        # If user passed a model that isn't compiled, compile it.
        if not hasattr(model, "loss") or model.loss is None:
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr),
                          loss=mse_loss, metrics=[PSNR])
    return model


def _train_once_version_safe(model: tf.keras.Model, x, y):
    """
    Keras added 'reset_metrics' kwarg at some point; make this call version-safe.
    """
    try:
        return model.train_on_batch(x, y, reset_metrics=False)
    except TypeError:
        return model.train_on_batch(x, y)


def _synthetic_batch(cfg: DenoiseConfig) -> Tuple[tf.Tensor, tf.Tensor]:
    """
    Generate a synthetic (x, y) batch with the correct shapes & dtypes for quick benchmarking.
    """
    bs, P = int(cfg.batch_size), int(cfg.patch_size)
    x = tf.random.uniform((bs, P, P, 1), dtype=tf.float32)
    y = tf.random.uniform((bs, P, P, 1), dtype=tf.float32)
    return x, y


def _real_dataset_iter(stack: np.ndarray, cfg: DenoiseConfig) -> Iterable[Tuple[tf.Tensor, tf.Tensor]]:
    """
    Create an iterator over your *real* training pipeline (Noise2Noise pairs,
    standardization, random patches, augmentation). Infinite iterator.
    """
    ds = make_dataset(stack, cfg)            # infinite, prefetch done inside
    it = iter(ds)
    while True:
        yield next(it)


# -----------------------
# Main public entrypoint
# -----------------------

def benchmark_steps(cfg: DenoiseConfig,
                    *,
                    steps: int = 30,
                    warmup: int = 5,
                    model: Optional[tf.keras.Model] = None,
                    use_real_pipeline: bool = False,
                    stack: Optional[np.ndarray] = None,
                    device: Optional[str] = None) -> Dict[str, Any]:
    """
    Measure steps/sec for your exact model/loss/metrics.

    Args:
        cfg:           DenoiseConfig (you can clone_cfg(...) to override batch/patch/steps).
        steps:         Number of timed training steps.
        warmup:        Warmup steps (graph/XLA/cuDNN autotune).
        model:         Use this model if provided; else we build & compile your U-Net.
        use_real_pipeline:
                       If True, pull (x,y) from your real tf.data pipeline via make_dataset(stack, cfg).
                       If False, use synthetic random batches with the same shapes.
        stack:         Required if use_real_pipeline=True. Must be a NumPy array of shape (N, H, W).
        device:        Optional TF device context (e.g., '/GPU:0'). If None, let TF pick.

    Returns:
        dict with keys:
            'steps'            (int)
            'warmup'           (int)
            'seconds'          (float)
            'steps_per_second' (float)
            'batch_size'       (int)
            'patch_size'       (int)
            'samples_per_epoch_est' (int)  # batch_size * cfg.steps_per_epoch
            'seconds_per_epoch_est'  (float)  # cfg.steps_per_epoch / steps_per_second
            'used_pipeline'    ('real' or 'synthetic')
    """
    if use_real_pipeline and stack is None:
        raise ValueError("benchmark_steps: 'stack' must be provided when use_real_pipeline=True.")

    mdl = ensure_compiled_model(cfg, model=model)

    # Build a batch supplier
    if use_real_pipeline:
        supplier = _real_dataset_iter(stack, cfg)
        def next_batch():
            return next(supplier)  # tensors from tf.data
        used = "real"
    else:
        def next_batch():
            return _synthetic_batch(cfg)    # float32 tensors
        used = "synthetic"

    # Optional device context
    context = tf.device(device) if device else nullcontext()

    with context:
        # Warm-up
        for _ in range(int(max(0, warmup))):
            x, y = next_batch()
            _train_once_version_safe(mdl, x, y)

        # Timed loop
        t0 = time.time()
        for _ in range(int(steps)):
            x, y = next_batch()
            _train_once_version_safe(mdl, x, y)
        dt = time.time() - t0

    sps = float(steps) / max(dt, 1e-9)
    return {
        "steps": int(steps),
        "warmup": int(max(0, warmup)),
        "seconds": float(dt),
        "steps_per_second": sps,
        "batch_size": int(cfg.batch_size),
        "patch_size": int(cfg.patch_size),
        "samples_per_epoch_est": int(cfg.batch_size) * int(cfg.steps_per_epoch),
        "seconds_per_epoch_est": float(cfg.steps_per_epoch) / max(sps, 1e-9),
        "used_pipeline": used,
    }


# Python 3.8 compat
class nullcontext:
    def __enter__(self): return self
    def __exit__(self, *exc): return False