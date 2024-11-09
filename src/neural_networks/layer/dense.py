import numpy as np


# Dense layer
class Dense:
    """Class of implementation for dense layer"""

    # Layer initialization
    def __init__(self, n_inputs: int, n_neurons: int):
        """Initialize weights and biases

        Parameters
        ----------
        n_inputs : int
            number of input instances
        n_neurons : int
            number of neurons of the dense layer
        """
        self.weights = np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs: np.ndarray):
        """Calculate output values from inputs, weights and biases

        Parameters
        ----------
        inputs : np.ndarray
            inputs array
        """
        self.output = np.dot(inputs, self.weights) + self.biases
