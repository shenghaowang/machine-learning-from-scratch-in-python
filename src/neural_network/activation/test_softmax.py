import numpy as np
import pytest

from neural_network.activation.softmax import Softmax


@pytest.fixture
def softmax() -> Softmax:
    return Softmax()


@pytest.fixture
def inputs() -> np.ndarray:
    inputs = [[4.8, 1.21, 2.385], [8.9, -1.81, 0.2], [1.41, 1.051, 0.026]]
    return np.array(inputs)


def test_forward(softmax: Softmax, inputs: np.ndarray):
    softmax.forward(inputs)
    print(np.sum(softmax.output, axis=1))
    assert softmax.output.shape == inputs.shape
    np.testing.assert_allclose(
        np.sum(softmax.output, axis=1), np.array([1.0, 1.0, 1.0])
    )
