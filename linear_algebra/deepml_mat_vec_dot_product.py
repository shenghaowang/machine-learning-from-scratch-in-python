# https://www.deep-ml.com/problems/1

from typing import List


def matrix_dot_vector(a: List[List[int|float]], b: List[int|float]) -> List[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.

    if len(a[0]) != len(b):
        return -1
	
    res = [sum(x * y for x, y in zip(row, b)) for row in a]
    
    return res


def test_matrix_dot_vector(
    a: List[List[int|float]], b: List[int|float], groundtruth:  List[int|float]
) -> None:
    res = matrix_dot_vector(a, b)
    print(res)

    assert res == groundtruth


if __name__ == "__main__":
    a = [[1, 2], [2, 4]]
    b = [1, 2]
    groundtruth = [5, 10]
    test_matrix_dot_vector(a, b, groundtruth)
