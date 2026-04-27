# https://www.deep-ml.com/problems/6

import numpy as np


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # Solve (a - lambda)(d - lambda) - bc = 0
    # lambda^2 - (a+d)lambda + (ad-bc) = 0
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    trace = a + d
    determinant = a * d - b * c
    discriminant = trace**2 - 4 * determinant

    # Solve for lambdas
    lambda_1 = (trace + discriminant**0.5) / 2
    lambda_2 = (trace - discriminant**0.5) / 2

    return [lambda_1, lambda_2]


def test_calculate_eigenvalues(
    matrix: list[list[float | int]], groundtruth: list[float]
) -> None:
    res = calculate_eigenvalues(matrix)
    print(res)

    assert np.array_equal(res, groundtruth), "Arrays are not equal"


if __name__ == "__main__":
    test_calculate_eigenvalues(matrix=[[2, 1], [1, 2]], groundtruth=[3.0, 1.0])
