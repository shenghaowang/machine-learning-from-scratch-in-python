# https://www.deep-ml.com/problems/4


def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == "row":
        return [sum(row) / len(row) for row in matrix]

    elif mode == "column":
        return [sum(col) / len(col) for col in zip(*matrix)]

    else:
        raise ValueError(f"Unrecognized mode: {mode}")


def test_calculate_matrix_mean(
    matrix: list[list[float]], mode: str, groundtruth: list[float]
) -> None:
    res = calculate_matrix_mean(matrix, mode)
    print(res)

    assert res == groundtruth


if __name__ == "__main__":
    test_calculate_matrix_mean(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        mode="column",
        groundtruth=[4.0, 5.0, 6.0],
    )
    test_calculate_matrix_mean(
        matrix=[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        mode="row",
        groundtruth=[2.0, 5.0, 8.0],
    )
