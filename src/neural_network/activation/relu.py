import numpy as np


class ReLU:
    """Class of ReLu activation function"""

    # Forward pass
    def forward(self, inputs: np.ndarray):
        """Calculate output values of the ReLu function

        Parameters
        ----------
        inputs : np.ndarray
            input array
        """

        self.output = np.maximum(0, inputs)
