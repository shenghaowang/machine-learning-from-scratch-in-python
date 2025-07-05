# https://www.deep-ml.com/problems/5


def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    return [[val * scalar for val in row] for row in matrix]


def test_scalar_multiply(
    matrix: list[list[int | float]],
    scalar: int | float,
    groundtruth: list[list[int | float]],
) -> None:
    res = scalar_multiply(matrix, scalar)
    print(res)

    assert res == groundtruth


if __name__ == "__main__":
    test_scalar_multiply(
        matrix=[[1, 2], [3, 4]], scalar=2, groundtruth=[[2, 4], [6, 8]]
    )
