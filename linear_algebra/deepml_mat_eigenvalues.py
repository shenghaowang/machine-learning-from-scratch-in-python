import numpy as np


def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    eigenvalues, _ = np.linalg.eig(matrix)
    return eigenvalues


def test_calculate_eigenvalues(
    matrix: list[list[float | int]], groundtruth: list[float]
) -> None:
    res = calculate_eigenvalues(matrix)
    print(res)

    assert np.array_equal(res, groundtruth), "Arrays are not equal"


if __name__ == "__main__":
    test_calculate_eigenvalues(matrix=[[2, 1], [1, 2]], groundtruth=[3.0, 1.0])
