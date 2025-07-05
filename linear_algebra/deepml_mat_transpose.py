# https://www.deep-ml.com/problems/2

from typing import List


def transpose_matrix(a: List[List[int|float]]) -> List[List[int|float]]:
    b = []
    ncols = len(a[0])

    for i in range(ncols):
        row_b = []
        for row_a in a:
            row_b.append(row_a[i])
        
        b.append(row_b)
    
    return b

    # return list(map(list, zip(*a)))


def test_transpose_matrix(
    a: List[List[int|float]], groundtruth: List[List[int|float]]
) -> None:
    b = transpose_matrix(a)
    print(b)

    assert b == groundtruth


if __name__ == "__main__":
    test_transpose_matrix(
        a=[[1,2,3],[4,5,6]],
        groundtruth=[[1,4],[2,5],[3,6]]
    )
