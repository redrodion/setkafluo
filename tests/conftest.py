import numpy as np
import pytest


@pytest.fixture(autouse=True)
def rng_seed():
    """Seed NumPy's global RNG before every test for reproducibility."""
    np.random.seed(0)


@pytest.fixture
def tf():
    """Import TensorFlow or skip the test cleanly if it is not installed."""
    return pytest.importorskip("tensorflow")
