import numpy as np
import pytest

from neural_network.activation.relu import ReLU


@pytest.fixture
def relu() -> ReLU:
    return ReLU()


@pytest.fixture
def inputs() -> np.ndarray:
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
    return np.array(inputs)


def test_forward(relu: ReLU, inputs: np.ndarray):
    relu.forward(inputs)
    assert relu.output.shape == inputs.shape
    np.testing.assert_array_equal(relu.output, np.array([0, 2, 0, 3.3, 0, 1.1, 2.2, 0]))
