import numpy as np


class Softmax:
    """Class of Softmax activation function"""

    # Forward pass
    def forward(self, inputs: np.ndarray):
        """Calculate output values of the softmax function

        Parameters
        ----------
        inputs : np.ndarray
            input array
        """

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
