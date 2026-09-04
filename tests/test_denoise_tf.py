"""TensorFlow-backed tests for setkafluo.denoise.

All tests take the ``tf`` fixture; the module is skipped cleanly without TF.
"""
import numpy as np
import pytest

pytest.importorskip("tensorflow")

from setkafluo.denoise import make_unet, make_dataset, train, DenoiseConfig


# Captured from tag v0.1.0-paper: make_unet(DenoiseConfig()) — the weighted-layer
# sequence and parameter count that paper-era checkpoints were trained with.
# Hard-coded deliberately so the test does not depend on git at runtime.
PAPER_LAYER_NAMES = [
    "InputLayer", "Lambda", "Conv2D", "ReLU", "Lambda", "Conv2D", "ReLU",
    "MaxPooling2D", "Lambda", "Conv2D", "ReLU", "Lambda", "Conv2D", "ReLU",
    "MaxPooling2D", "Lambda", "Conv2D", "ReLU", "Lambda", "Conv2D", "ReLU",
    "MaxPooling2D", "Lambda", "Conv2D", "ReLU", "Lambda", "Conv2D", "ReLU",
    "MaxPooling2D", "Lambda", "Conv2D", "ReLU", "Lambda", "Conv2D", "ReLU",
    "UpSampling2D", "Lambda", "Conv2D", "ReLU", "Lambda", "Concatenate",
    "Lambda", "Conv2D", "ReLU", "Lambda", "Conv2D", "ReLU", "UpSampling2D",
    "Lambda", "Conv2D", "ReLU", "Lambda", "Concatenate", "Lambda", "Conv2D",
    "ReLU", "Lambda", "Conv2D", "ReLU", "UpSampling2D", "Lambda", "Conv2D",
    "ReLU", "Lambda", "Concatenate", "Lambda", "Conv2D", "ReLU", "Lambda",
    "Conv2D", "ReLU", "UpSampling2D", "Lambda", "Conv2D", "ReLU", "Lambda",
    "Concatenate", "Lambda", "Conv2D", "ReLU", "Lambda", "Conv2D", "ReLU",
    "Conv2D",
]
PAPER_PARAM_COUNT = 31030593  # captured from v0.1.0-paper


@pytest.mark.parametrize("shape", [(32, 32), (48, 48), (50, 70), (33, 33)])
def test_unet_preserves_spatial_shape(tf, shape):
    H, W = shape
    model = make_unet(DenoiseConfig())
    x = np.zeros((1, H, W, 1), dtype=np.float32)
    y = model(x)
    assert tuple(y.shape) == (1, H, W, 1)


def test_unet_layer_sequence_matches_paper_tag(tf):
    # Checkpoint-compatibility invariant: the sequence of weighted Keras layers
    # and the parameter count must match tag v0.1.0-paper, so paper-era weights
    # still load unchanged.
    model = make_unet(DenoiseConfig())
    names = [layer.__class__.__name__ for layer in model.layers]
    assert names == PAPER_LAYER_NAMES
    assert model.count_params() == PAPER_PARAM_COUNT


def test_make_dataset_yields_contract(tf):
    stack = np.random.rand(15, 64, 64).astype(np.float64)
    cfg = DenoiseConfig(batch_size=4, patch_size=32)
    x, y = next(iter(make_dataset(stack, cfg)))
    assert x.dtype == tf.float32 and y.dtype == tf.float32
    assert tuple(x.shape) == (4, 32, 32, 1)
    assert tuple(y.shape) == (4, 32, 32, 1)
    xn, yn = x.numpy(), y.numpy()
    for b in range(4):
        assert abs(float(xn[b].mean())) < 1.0
        assert abs(float(yn[b].mean())) < 1.0


def test_train_one_step_smoke(tf):
    stack = np.random.rand(15, 64, 64).astype(np.float64)
    cfg = DenoiseConfig(epochs=1, steps_per_epoch=2, batch_size=2, patch_size=32,
                        start_ch=8, depth=2, mixed_precision_warn=False)
    model, hist = train(stack, cfg)
    assert model is not None
    assert "loss" in hist.history
    losses = hist.history["loss"]
    assert len(losses) == 1 and np.isfinite(losses[0])
