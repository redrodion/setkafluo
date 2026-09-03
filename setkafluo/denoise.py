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

This module provides:
- DenoiseConfig: small, explicit configuration container (paths, LR, epochs, patch size, overlap).
- make_unet: U-Net backbone with symmetric padding and dtype-safe float32 output (mixed precision ready).
- make_dataset: tf.data pipeline generating Noise2Noise pairs with standardization and augmentations.
- train: compile (Adam + MSE + PSNR) and fit, with epoch-based checkpoint saving.
- predict_tiled: overlap-tiled inference with edge padding and Hann blending + de-standardization.
- predict_all_checkpoints: batch inference across saved checkpoints (epoch-sorted) to TIFF.

Designed for:
- Reproducible XRF denoising experiments in Jupyter/Colab.
- Consistent preprocessing/standardization for quantitative output units.
- Easy integration with the SetkaFluo data exploration utilities.
"""

from __future__ import annotations

from . import (
    __author__, __author_email__,
    __maintainer__, __maintainer_email__,
    __license__, __version__, __copyright__
)

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import tifffile
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input, Conv2D, Conv2DTranspose, UpSampling2D, Dropout,
    AveragePooling2D, MaxPooling2D, ReLU, Concatenate,
    Lambda, LayerNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.utils import register_keras_serializable

# -----------------------
# Utility I/O
# -----------------------
def save_tiff(image: np.ndarray, path: str | os.PathLike):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(path), image)

def load_tiff(path: str | os.PathLike) -> np.ndarray:
    return np.array(tifffile.imread(str(path)))

# -----------------------
# Config
# -----------------------
@dataclass
class DenoiseConfig:
    # data / training
    lr: float = 1e-5
    batch_size: int = 4
    patch_size: int = 32
    steps_per_epoch: int = 128
    epochs: int = 50
    # model
    input_channels: int = 1
    output_channels: int = 1
    start_ch: int = 64
    depth: int = 4
    inc_rate: float = 2.0
    dropout: float = 0.0
    instancenorm: bool = False
    averagepool: bool = False
    upconv: bool = False
    residual: bool = False
    lambda_reg: float = 0.0
    reg_l1: bool = False
    # inference
    min_overlap: int = 16
    # checkpoints
    save_models_dir: str | os.PathLike | None = None
    save_every_epochs: int = 25
    # runtime
    mixed_precision_warn: bool = True

# -----------------------
# Augment
# -----------------------
def augment_patch(patch_in: np.ndarray, patch_target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    opt = np.random.randint(0, 4)
    if opt == 0:
        return patch_in, patch_target
    if opt == 1:
        k = np.random.randint(1, 4)
        return np.rot90(patch_in, k), np.rot90(patch_target, k)
    if opt == 2:
        return np.fliplr(patch_in), np.fliplr(patch_target)
    return np.flipud(patch_in), np.flipud(patch_target)

# -----------------------
# Standardization
# -----------------------
def standardize_images(stack: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if stack.ndim != 3:
        raise ValueError("standardize_images expects a stack with shape (N, H, W).")
    eps = 1e-8
    out, means, stds = [], [], []
    for img in stack:
        m = float(np.mean(img)); m = m if abs(m) > eps else eps
        z = (img / m) - 1.0
        s = float(np.std(z)); s = s if s > eps else eps
        out.append((z / s).astype(np.float32))
        means.append(m); stds.append(s)
    return np.asarray(out, np.float32), np.asarray(means, np.float32), np.asarray(stds, np.float32)

def undo_standardization(img_std: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (img_std * std + 1.0) * mean

# -----------------------
# Noise2Noise sampling
# -----------------------
def generate_noise2noise_samples(stack: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
    M = stack.shape[0]
    half = int(np.floor(M / 2))
    idx_all = np.arange(M)
    xin, xgt = [], []
    for _ in range(n_samples):
        idx_in  = np.random.choice(idx_all, size=half, replace=False)
        idx_out = np.setdiff1d(idx_all, idx_in)
        xin.append(stack[idx_in].mean(axis=0))
        xgt.append(stack[idx_out].mean(axis=0))
    return np.asarray(xin), np.asarray(xgt)

def extract_random_patches(imgs_in: np.ndarray, imgs_gt: np.ndarray, patch: int) -> Tuple[np.ndarray, np.ndarray]:
    n = imgs_in.shape[0]
    out_in, out_gt = [], []
    for i in range(n):
        H, W = imgs_in[i].shape[:2]
        if patch > H or patch > W:
            raise ValueError(f"Patch {patch} exceeds image ({H}x{W}).")
        top  = np.random.randint(0, H - patch + 1)
        left = np.random.randint(0, W - patch + 1)
        pi = imgs_in[i][top:top+patch, left:left+patch]
        pg = imgs_gt[i][top:top+patch, left:left+patch]
        if pi.ndim == 2: pi = pi[..., None]
        if pg.ndim == 2: pg = pg[..., None]
        out_in.append(pi); out_gt.append(pg)
    return np.asarray(out_in, np.float32), np.asarray(out_gt, np.float32)

# -----------------------
# Tiled inference helpers
# -----------------------
def extract_covering_patches_with_overlap_pad(image: np.ndarray, patch: int, min_overlap: int):
    H, W = image.shape[:2]
    n_rows = int(np.ceil((H - patch) / (patch - min_overlap))) + 1
    n_cols = int(np.ceil((W - patch) / (patch - min_overlap))) + 1
    stride = patch - min_overlap

    tot_y = patch + (n_rows - 1) * stride
    tot_x = patch + (n_cols - 1) * stride
    cy, cx = H / 2.0, W / 2.0
    grid_top  = int(round(cy - tot_y / 2))
    grid_left = int(round(cx - tot_x / 2))

    patches, coords = [], []
    for i in range(n_rows):
        for j in range(n_cols):
            top  = grid_top  + i * stride
            left = grid_left + j * stride
            y0, x0 = max(top, 0), max(left, 0)
            y1, x1 = min(top + patch, H), min(left + patch, W)
            valid = image[y0:y1, x0:x1] if image.ndim == 2 else image[y0:y1, x0:x1, :]
            pad_t = y0 - top
            pad_l = x0 - left
            pad_b = (top + patch) - y1
            pad_r = (left + patch) - x1
            pad = ((pad_t, pad_b), (pad_l, pad_r)) if image.ndim == 2 else ((pad_t, pad_b), (pad_l, pad_r), (0, 0))
            p = np.pad(valid, pad, mode='edge')
            patches.append(p); coords.append((top, left))
    return np.asarray(patches), coords

def reconstruct_from_covering_patches_hann_custom(patches, coordinates, out_shape, patch):
    H, W = out_shape[:2]; C = out_shape[2] if len(out_shape) == 3 else None

    row = {}
    for idx, (y, x) in enumerate(coordinates):
        row.setdefault(y, []).append((x, idx))
    for y in row: row[y].sort(key=lambda p: p[0])

    col = {}
    for idx, (y, x) in enumerate(coordinates):
        col.setdefault(x, []).append((y, idx))
    for x in col: col[x].sort(key=lambda p: p[0])

    def ext_hann(taper):
        if taper <= 0: return None, None
        win = np.hanning(2 * taper)
        left = np.flip(win[taper:]); left /= (left.max() if left.max() else 1)
        right = np.flip(win[:taper]); right /= (right.max() if right.max() else 1)
        return left, right

    masks = [None] * len(coordinates)
    for idx, (y, x) in enumerate(coordinates):
        xs = [p[0] for p in row[y]]; pos = xs.index(x)
        left_t  = 0 if pos == 0 else int(round((patch - (x - xs[pos-1])) / 2))
        right_t = 0 if pos == len(xs)-1 else int(round((patch - (xs[pos+1] - x)) / 2))
        ys = [p[0] for p in col[x]]; posv = ys.index(y)
        top_t    = 0 if posv == 0 else int(round((patch - (y - ys[posv-1])) / 2))
        bottom_t = 0 if posv == len(ys)-1 else int(round((patch - (ys[posv+1] - y)) / 2))

        wx = np.ones(patch, np.float32)
        if left_t  > 0: wx[:left_t]   = ext_hann(left_t)[0]
        if right_t > 0: wx[-right_t:] = ext_hann(right_t)[1]
        wy = np.ones(patch, np.float32)
        if top_t    > 0: wy[:top_t]     = ext_hann(top_t)[0]
        if bottom_t > 0: wy[-bottom_t:] = ext_hann(bottom_t)[1]
        mask = np.outer(wy, wx)
        if patches.ndim == 4 and C is not None:
            mask = mask[..., None]
        masks[idx] = mask

    acc  = np.zeros(out_shape, dtype=patches.dtype)
    wmap = np.zeros(out_shape if C is not None else (H, W), dtype=np.float32)
    if C is not None and wmap.ndim == 2: wmap = wmap[..., None]

    for p, (top, left), m in zip(patches, coordinates, masks):
        ph, pw = p.shape[:2]
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + ph, H), min(left + pw, W)
        py0, px0 = 0 if top >= 0 else -top, 0 if left >= 0 else -left
        py1, px1 = ph if top + ph <= H else ph - ((top + ph) - H), pw if left + pw <= W else pw - ((left + pw) - W)
        vp = p[py0:py1, px0:px1, ...]
        vm = m[py0:py1, px0:px1, ...] if m.ndim == 3 else m[py0:py1, px0:px1]
        acc[y0:y1, x0:x1, ...] += vp * vm
        if C is None:
            wmap[y0:y1, x0:x1] += vm
        else:
            wmap[y0:y1, x0:x1, 0] += vm[..., 0]

    wmap[wmap == 0] = 1.0
    return acc / wmap

# -----------------------
# Registered ops for Lambda (safe serialization)
# -----------------------
@register_keras_serializable(package='Custom', name='edge_padding')
def _edge_padding(inputs, pad_width):
    return tf.pad(inputs, [[0, 0], [pad_width, pad_width], [pad_width, pad_width], [0, 0]], mode='SYMMETRIC')

@register_keras_serializable(package='Custom', name='pad_or_crop_to_match')
def _pad_or_crop_to_match(inputs):
    x, ref = inputs
    xs, rs = tf.shape(x), tf.shape(ref)
    dh, dw = rs[1] - xs[1], rs[2] - xs[2]
    ph, pw = tf.maximum(dh, 0), tf.maximum(dw, 0)
    pt, pb = ph // 2, ph - ph // 2
    pl, pr = pw // 2, pw - pw // 2
    x = tf.pad(x, [[0,0],[pt,pb],[pl,pr],[0,0]], mode='SYMMETRIC')
    ch, cw = tf.maximum(-dh, 0), tf.maximum(-dw, 0)
    ct, cb = ch // 2, ch - ch // 2
    cl, cr = cw // 2, cw - cw // 2
    nh = tf.shape(x)[1] - ct - cb
    nw = tf.shape(x)[2] - cl - cr
    return x[:, ct:ct+nh, cl:cl+nw, :]

@register_keras_serializable(package='Custom', name='cast_to_float32')
def _cast_to_float32(x):
    return tf.cast(x, tf.float32)

# -----------------------
# Model
# -----------------------
def make_unet(cfg: DenoiseConfig) -> Model:
    def conv_block(m, dim, instn, res, do, reg):
        m_in = m
        m = Lambda(_edge_padding, arguments={'pad_width': 1})(m)
        n = Conv2D(dim, 3, activation=None, padding='valid', kernel_regularizer=reg)(m)
        n = ReLU()(n)
        if instn: n = LayerNormalization()(n)
        if do:    n = Dropout(do)(n)

        n = Lambda(_edge_padding, arguments={'pad_width': 1})(n)
        n = Conv2D(dim, 3, activation=None, padding='valid', kernel_regularizer=reg)(n)
        n = ReLU()(n)
        if instn: n = LayerNormalization()(n)
        return Concatenate()([m_in, n]) if res else n

    def level_block(m, dim, depth, inc, do, instn, avgp, up, res, reg):
        if depth > 0:
            n = conv_block(m, dim, instn, res, do, reg)
            if avgp:
                m = AveragePooling2D(pool_size=(2, 2), padding='same')(n)
            else:
                m = MaxPooling2D(pool_size=(2, 2), padding='same')(n)
            m = level_block(m, int(inc * dim), depth - 1, inc, do, instn, avgp, up, res, reg)
            if up:
                m = Lambda(_edge_padding, arguments={'pad_width': 1})(m)
                m = Conv2DTranspose(dim, 3, strides=2, padding='valid', kernel_regularizer=reg)(m)
            else:
                m = UpSampling2D()(m)
                m = Lambda(_edge_padding, arguments={'pad_width': 1})(m)
                m = Conv2D(dim, 2, activation=None, padding='valid', kernel_regularizer=reg)(m)
                m = ReLU()(m)
                if instn: m = LayerNormalization()(m)
            m = Lambda(_pad_or_crop_to_match)([m, n])
            n = Concatenate()([n, m])
            m = conv_block(n, dim, instn, res, do, reg)
        else:
            m = conv_block(m, dim, instn, res, do, reg)
        return m

    reg = l1(cfg.lambda_reg) if cfg.reg_l1 else l2(cfg.lambda_reg)
    inp = Input(shape=(None, None, cfg.input_channels))
    x = level_block(inp, cfg.start_ch, cfg.depth, cfg.inc_rate, cfg.dropout,
                    cfg.instancenorm, cfg.averagepool, cfg.upconv, cfg.residual, reg)
    out = Conv2D(cfg.output_channels, 1, activation=None, use_bias=True)(x)

    # if mixed precision compute dtype is float16, cast output back to float32 (stable loss/metrics)
    if tf.keras.mixed_precision.global_policy().compute_dtype == 'float16':
        out = Lambda(_cast_to_float32, name="cast_to_float32")(out)

    return Model(inp, out)

# -----------------------
# Dataset / training
# -----------------------
def make_dataset(stack: np.ndarray, cfg: DenoiseConfig) -> tf.data.Dataset:
    def gen():
        while True:
            xin, xgt = generate_noise2noise_samples(stack, cfg.batch_size)
            xin_std, _, _ = standardize_images(xin)
            xgt_std, _, _ = standardize_images(xgt)
            pin, pgt = extract_random_patches(xin_std, xgt_std, cfg.patch_size)
            ain, agt = [], []
            for k in range(pin.shape[0]):
                a, b = augment_patch(pin[k], pgt[k])
                ain.append(a); agt.append(b)
            yield np.asarray(ain, np.float32), np.asarray(agt, np.float32)

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(cfg.batch_size, cfg.patch_size, cfg.patch_size, 1), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg.batch_size, cfg.patch_size, cfg.patch_size, 1), dtype=tf.float32),
        ),
    )
    return ds.prefetch(tf.data.AUTOTUNE)

def tf_log10(x): return tf.math.log(x) / tf.math.log(tf.constant(10.0, dtype=x.dtype))

def PSNR(y_true, y_pred):
    y_pred = tf.clip_by_value(y_pred, 0.0, 1.0)
    mse = tf.reduce_mean(tf.square(y_pred - y_true)) + 1e-8
    return 10.0 * tf_log10(1.0 / mse)

def mse_loss(y_true, y_pred): return tf.reduce_mean(tf.square(y_true - y_pred))

def train(stack: np.ndarray, cfg: DenoiseConfig) -> Tuple[Model, tf.keras.callbacks.History]:
    if cfg.mixed_precision_warn:
        try:
            from tensorflow.keras import mixed_precision as _mp
            print("Mixed precision policy:", _mp.global_policy())
        except Exception:
            pass

    model = make_unet(cfg)
    model.compile(optimizer=Adam(learning_rate=cfg.lr), loss=mse_loss, metrics=[PSNR])

    callbacks = []
    if cfg.save_models_dir:
        os.makedirs(cfg.save_models_dir, exist_ok=True)
        class _Saver(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % cfg.save_every_epochs == 0:
                    p = os.path.join(cfg.save_models_dir, f"model_epoch_{epoch+1:05d}.weights.h5")
                    # weights-only -> compact & reliable size, no Lambda serialization involved
                    self.model.save_weights(p)
                    print(f"[checkpoint] saved weights: {p}")
        callbacks.append(_Saver())

    ds = make_dataset(stack, cfg)
    hist = model.fit(ds, steps_per_epoch=cfg.steps_per_epoch, epochs=cfg.epochs,
                     callbacks=callbacks, verbose=2)
    return model, hist

# -----------------------
# Inference
# -----------------------
def predict_tiled(img2d: np.ndarray, model: Model, cfg: DenoiseConfig) -> np.ndarray:
    if img2d.ndim != 2:
        raise ValueError("predict_tiled expects a single 2D image.")
    H, W = img2d.shape
    std, m, s = standardize_images(np.asarray([img2d]))
    patches, coords = extract_covering_patches_with_overlap_pad(std[0], cfg.patch_size, cfg.min_overlap)
    if patches.ndim == 3: patches = patches[..., None]
    preds = model.predict(patches, verbose=0)
    recon = reconstruct_from_covering_patches_hann_custom(preds, coords, (H, W, 1), cfg.patch_size)
    if recon.ndim == 3 and recon.shape[-1] == 1:
        recon = recon[..., 0]
    return undo_standardization(recon, float(m[0]), float(s[0]))

def predict_all_checkpoints(checkpoints_dir: str | os.PathLike,
                            out_dir: str | os.PathLike,
                            img2d: np.ndarray,
                            cfg: DenoiseConfig,
                            delete_after: bool = False):
    """
    Evaluate every weights-only checkpoint (*.weights.h5) in `checkpoints_dir`, sorted by trailing epoch number.
    Saves predictions to `out_dir` as prediction_epoch_00025.tif, etc.
    """
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = Path(checkpoints_dir)

    def epoch_num(p: Path) -> int | None:
        m = re.search(r'(\d+)(?=\.weights\.h5$)', p.name)
        return int(m.group(1)) if m else None

    items = [(p, epoch_num(p)) for p in ckpt_dir.glob("*.weights.h5")]
    items = [(p, ep) for (p, ep) in items if ep is not None]
    items.sort(key=lambda t: t[1])

    print(f"[predict_all_checkpoints] found {len(items)} checkpoints in {ckpt_dir}")
    if not items:
        print("  -> no matching files like model_epoch_00025.weights.h5")
        return

    for p, ep in items:
        print(f"  -> loading weights {p.name} (epoch={ep:05d})")
        try:
            model = make_unet(cfg)
            model.load_weights(str(p))
        except Exception as e:
            print(f"     [skip] load failed: {type(e).__name__}: {e}")
            continue

        try:
            den = predict_tiled(img2d, model, cfg)
            out_name = f"prediction_epoch_{ep:05d}.tif"
            save_tiff(den, out_dir / out_name)
            print(f"     [ok] saved -> {out_name}")
        except Exception as e:
            print(f"     [skip] inference failed: {type(e).__name__}: {e}")
        finally:
            del model
            tf.keras.backend.clear_session()

    if delete_after:
        for p, _ in items:
            try:
                p.unlink()
            except Exception as e:
                print(f"     [warn] could not delete {p.name}: {e}")