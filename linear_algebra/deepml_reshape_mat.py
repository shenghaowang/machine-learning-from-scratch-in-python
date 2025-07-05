# https://www.deep-ml.com/problems/3

from typing import List, Tuple

import numpy as np


def reshape_matrix(
    a: List[List[int | float]], new_shape: Tuple[int, int]
) -> List[List[int | float]]:
    # Write your code here and return a python list after reshaping by using numpy's tolist() method

    nrow, ncol = len(a), len(a[0])
    if nrow * ncol != new_shape[0] * new_shape[1]:
        return []

    reshaped_matrix = np.reshape(a, new_shape).tolist()

    return reshaped_matrix


def test_reshape_matrix(
    a: List[List[int | float]],
    new_shape: Tuple[int, int],
    groundtruth: List[List[int | float]],
) -> None:

    reshaped_matrix = reshape_matrix(a, new_shape)
    print(reshaped_matrix)

    assert reshaped_matrix == groundtruth


if __name__ == "__main__":
    test_reshape_matrix(
        a=[[1, 2, 3, 4], [5, 6, 7, 8]],
        new_shape=(4, 2),
        groundtruth=[[1, 2], [3, 4], [5, 6], [7, 8]],
    )
