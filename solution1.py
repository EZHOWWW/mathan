from types import FunctionType
from math import sqrt


def mgch(x: float, *args) -> float:
    res = 0
    for i, k in enumerate(args):
        res += x**i * k
    return res


class Solution1:
    def get_func_min(
        self, func: FunctionType, method: str = "dichotomy", *args, **kwargs
    ) -> float:
        pass

    def dichotomy_method(
        self,
        f: FunctionType,
        a0: float,
        b0: float,
        epsilon: float,
        delta: float = 1e-5,
        max_iterations: int = 1000,
    ) -> tuple[float, list[tuple[float, float]], list[float]]:
        a = a0
        b = b0
        intervals = [(a, b)]
        x_points = []
        for k in range(max_iterations):
            mid = (a + b) / 2
            yk = mid - delta
            zk = mid + delta
            f_yk = f(yk)
            f_zk = f(zk)

            if f_yk <= f_zk:
                b = zk
            else:
                a = yk

            intervals.append((a, b))
            x_points.append((a + b) / 2)

            if abs(b - a) <= epsilon:
                break
        x_star = (a + b) / 2
        return x_star, intervals, x_points

    def golden_section_method(self, f, a0, b0, epsilon, max_iterations=1000):
        phi = (1 + sqrt(5)) / 2
        resphi = 2 - phi

        a = a0
        b = b0
        intervals = [(a, b)]
        x_points = []

        # Инициализация точек
        y = a + resphi * (b - a)
        z = b - resphi * (b - a)
        f_y = f(y)
        f_z = f(z)

        for k in range(max_iterations):
            if f_y <= f_z:
                b = z
                z = y
                f_z = f_y
                y = a + resphi * (b - a)
                f_y = f(y)
            else:
                a = y
                y = z
                f_y = f_z
                z = b - resphi * (b - a)
                f_z = f(z)

            intervals.append((a, b))
            x_points.append((a + b) / 2)

            if abs(b - a) <= epsilon:
                break
        x_star = (a + b) / 2
        return x_star, intervals, x_points


def main():
    s = Solution1()

    def f1(x: float) -> float:
        return mgch(x, 1, 3, -5, 1)

    f = f1

    a0 = 0
    b0 = 5
    epsilon = 1e-5

    x_min_dichotomy, intervals_dichotomy, points_dichotomy = s.dichotomy_method(
        f, a0, b0, epsilon
    )
    x_min_golden, intervals_golden, points_golden = s.golden_section_method(
        f, a0, b0, epsilon
    )
    print(
        f"Минимум методом дихотомии: x = {x_min_dichotomy}, f(x) = {f(x_min_dichotomy)}"
    )
    print(
        f"Минимум методом золотого сечения: x = {x_min_golden}, f(x) = {f(x_min_golden)}"
    )


if __name__ == "__main__":
    main()
