# https://neetcode.io/problems/gradient-descent


class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        minimizer = init

        for _ in range(iterations):
            derivative = 2 * minimizer
            minimizer = minimizer - learning_rate * derivative

        return round(minimizer, 5)


def test_get_minimizer(
    iterations: int, learning_rate: float, init: int, groundtruth: float
) -> None:
    sol = Solution()
    res = sol.get_minimizer(iterations, learning_rate, init)
    print(res)

    assert res == groundtruth


if __name__ == "__main__":
    test_get_minimizer(iterations=0, learning_rate=0.01, init=5, groundtruth=5)
    test_get_minimizer(iterations=10, learning_rate=0.01, init=5, groundtruth=4.08536)
