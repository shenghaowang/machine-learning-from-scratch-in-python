# https://www.deep-ml.com/problems/8


def inverse_2x2(matrix: list[list[float]]) -> list[list[float]] | None:
    """
    Calculate the inverse of a 2x2 matrix.

    Args:
        matrix: A 2x2 matrix represented as [[a, b], [c, d]]

    Returns:
        The inverse matrix as a 2x2 list, or None if the matrix is singular
        (i.e., determinant equals zero)
    """
    # Your code here
    a, b = matrix[0]
    c, d = matrix[1]

    det = a * d - b * c
    if det == 0:
        return None

    return [[d / det, -b / det], [-c / det, a / det]]


def test_inverse_2x2(
    matrix: list[list[float]], groundtruth: list[list[float]] | None
) -> None:
    inverse_matrix = inverse_2x2(matrix)
    print(inverse_matrix)
    assert inverse_matrix == groundtruth


if __name__ == "__main__":
    test_inverse_2x2(matrix=[[4, 7], [2, 6]], groundtruth=[[0.6, -0.7], [-0.2, 0.4]])
