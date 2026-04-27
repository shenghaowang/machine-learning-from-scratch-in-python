# https://www.deep-ml.com/problems/7

import numpy as np


def transform_matrix(
    A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]
) -> list[list[int | float]]:
    """Transformation: T^(-1)AS"""
    # Convert to numpy arrays for easier manipulation
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)

    # Check if the matrices T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        # raise ValueError("The matrices T and/or S are not invertible.")
        return -1

    # Compute the inverse of T
    T_inv = np.linalg.inv(T)

    # Perform the matrix transformation; use @ for better readability
    transformed_matrix = np.round(T_inv @ A @ S, 3)

    return transformed_matrix.tolist()


def test_transform_matrix(
    A: list[list[int | float]],
    T: list[list[int | float]],
    S: list[list[int | float]],
    groundtruth: list[list[int | float]],
) -> None:
    transformed_matrix = transform_matrix(A, T, S)
    print(transformed_matrix)
    assert transformed_matrix == groundtruth


if __name__ == "__main__":
    test_transform_matrix(
        A=[[1, 2], [3, 4]],
        T=[[2, 0], [0, 2]],
        S=[[1, 1], [0, 1]],
        groundtruth=[[0.5, 1.5], [1.5, 3.5]],
    )
    test_transform_matrix(
        A=[[2, 3], [1, 4]], T=[[3, 0], [0, 3]], S=[[1, 1], [1, 1]], groundtruth=-1
    )
