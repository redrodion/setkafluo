"""Pure-NumPy behaviour tests for setkafluo.denoise.

Importing setkafluo.denoise pulls in TensorFlow at module top, so the whole
module is skipped when TF is absent (the functions exercised here are otherwise
pure NumPy). We do not restructure the module to avoid this; a skip is fine.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from setkafluo.denoise import (
    standardize_images,
    undo_standardization,
    generate_noise2noise_samples,
    extract_random_patches,
    augment_patch,
    extract_covering_patches_with_overlap_pad,
    reconstruct_from_covering_patches_hann_custom,
)


def test_standardize_roundtrip():
    stack = np.random.poisson(20, size=(3, 40, 56)).astype(np.float64)
    z, means, stds = standardize_images(stack)
    assert z.shape == (3, 40, 56)
    for i in range(3):
        assert abs(float(z[i].mean())) < 1e-5
        assert abs(float(z[i].std()) - 1.0) < 1e-4
        rec = undo_standardization(z[i], means[i], stds[i])
        assert np.allclose(rec, stack[i], rtol=1e-4, atol=1e-3)


def test_standardize_constant_image():
    stack = np.full((1, 10, 10), 5.0, dtype=np.float64)
    z, means, stds = standardize_images(stack)
    assert np.all(np.isfinite(z))
    assert np.all(np.isfinite(means)) and np.all(np.isfinite(stds))


def test_standardize_rejects_2d():
    with pytest.raises(ValueError):
        standardize_images(np.zeros((40, 56)))


@pytest.mark.parametrize("M", [15, 16])
def test_n2n_halves_disjoint_and_covering(M):
    # image i is the constant float(i); the two averaged halves then read out
    # as the mean of each index set.
    stack = np.stack([np.full((8, 8), float(i)) for i in range(M)], axis=0)
    half = M // 2
    xin, xgt = generate_noise2noise_samples(stack, n_samples=20)
    assert xin.shape == (20, 8, 8) and xgt.shape == (20, 8, 8)

    total = float(sum(range(M)))
    splits = set()
    for k in range(20):
        # half*mean_in + (M-half)*mean_gt == sum of all indices iff the two
        # index sets are disjoint and together cover all M elements.
        value = half * xin[k, 0, 0] + (M - half) * xgt[k, 0, 0]
        assert abs(value - total) < 1e-6
        splits.add((round(float(xin[k, 0, 0]), 6), round(float(xgt[k, 0, 0]), 6)))
    assert len(splits) >= 2  # re-randomization actually happens


def test_extract_random_patches_shape_and_bounds():
    H, W = 50, 70
    # coordinate ramp: value = row*100 + col + image_offset, so the crop's
    # top-left corner reveals its (top, left) location. All values are small
    # integers, exactly representable in float32.
    base = (np.arange(H)[:, None] * 100 + np.arange(W)[None, :]).astype(np.float64)
    imgs = np.stack([base + n for n in range(4)], axis=0)
    patch = 32
    pin, pgt = extract_random_patches(imgs, imgs, patch)
    assert pin.shape == (4, patch, patch, 1)
    assert pgt.shape == (4, patch, patch, 1)
    assert pin.dtype == np.float32 and pgt.dtype == np.float32
    for n in range(4):
        crop = pin[n, :, :, 0]
        corner = int(round(float(crop[0, 0]) - n))
        top, left = corner // 100, corner % 100
        expected = imgs[n][top:top + patch, left:left + patch]
        assert np.allclose(crop, expected)
        assert np.allclose(pgt[n, :, :, 0], expected)


def test_extract_random_patches_too_large_raises():
    imgs = np.zeros((2, 50, 50), dtype=np.float64)
    with pytest.raises(ValueError):
        extract_random_patches(imgs, imgs, patch=64)


def test_augment_covers_all_8_orientations():
    p = np.arange(64).reshape(8, 8)  # no symmetry -> 8 distinct orientations
    seen = set()
    for _ in range(400):
        a, b = augment_patch(p, p)
        assert np.array_equal(a, b)  # identical transform applied to both
        seen.add(a.tobytes())
    assert len(seen) == 8


def _coverage(coords, H, W, patch):
    cov = np.zeros((H, W), dtype=int)
    for (top, left) in coords:
        y0, x0 = max(top, 0), max(left, 0)
        y1, x1 = min(top + patch, H), min(left + patch, W)
        cov[y0:y1, x0:x1] += 1
    return cov


@pytest.mark.parametrize("shape", [(64, 64), (50, 73), (33, 100), (100, 33)])
@pytest.mark.parametrize("min_overlap", [8, 16])
def test_tiling_roundtrip_identity(shape, min_overlap):
    H, W = shape
    patch = 32
    img = np.random.rand(H, W).astype(np.float64)
    patches, coords = extract_covering_patches_with_overlap_pad(img, patch, min_overlap)
    recon = reconstruct_from_covering_patches_hann_custom(patches, coords, img.shape, patch)
    assert np.allclose(recon, img, atol=1e-5)
    assert np.all(_coverage(coords, H, W, patch) >= 1)


@pytest.mark.parametrize("shape", [(64, 64), (50, 73), (33, 100), (100, 33)])
@pytest.mark.parametrize("min_overlap", [8, 16])
def test_tiling_roundtrip_with_channel(shape, min_overlap):
    H, W = shape
    patch = 32
    img = np.random.rand(H, W, 1).astype(np.float64)
    patches, coords = extract_covering_patches_with_overlap_pad(img, patch, min_overlap)
    recon = reconstruct_from_covering_patches_hann_custom(patches, coords, (H, W, 1), patch)
    assert recon.shape == (H, W, 1)
    assert np.allclose(recon, img, atol=1e-5)
    assert np.all(_coverage(coords, H, W, patch) >= 1)
