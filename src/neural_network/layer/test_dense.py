import numpy as np
import pytest
from nnfs.datasets import spiral_data

from neural_network.layer.dense import Dense


@pytest.fixture
def n_inputs() -> int:
    return 2


@pytest.fixture
def n_neurons() -> int:
    return 3


@pytest.fixture
def dense(n_inputs: int, n_neurons: int) -> Dense:
    return Dense(n_inputs, n_neurons)


@pytest.fixture
def n_samples() -> int:
    return 100


@pytest.fixture
def n_classes() -> int:
    return 3


def test_forward(dense: Dense, n_neurons: int, n_samples: int, n_classes: int):
    # Create dataset
    X, _ = spiral_data(samples=n_samples, classes=n_classes)

    # Forward pass
    dense.forward(X)

    assert isinstance(dense.output, np.ndarray)
    assert dense.output.shape == (n_samples * n_classes, n_neurons)
