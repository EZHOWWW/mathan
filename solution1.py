from types import FunctionType
from math import sqrt
import matplotlib.pyplot as plt

import numpy as np  # для визуализации (1 применение строка:89)


class Solution1:
    def dichotomy_method(
        f: FunctionType,
        a0: float,
        b0: float,
        epsilon: float,
        delta: float = 1e-5,
        max_iterations: int = 1000,
        find_min=True,
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

            if (
                f_yk <= f_zk if find_min else f_yk >= f_yk
            ):  # если find_min = True ищем минимум, иначе максимум
                b = zk
            else:
                a = yk

            intervals.append((a, b))
            x_points.append((a + b) / 2)

            if abs(b - a) <= epsilon:
                break
        x_star = (a + b) / 2
        return x_star, intervals, x_points

    def golden_section_method(f, a0, b0, epsilon, max_iterations=1000, find_min=True):
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
            if (
                f_y <= f_z if find_min else f_y >= f_y
            ):  # если find_min = True ищем минимум, иначе максимум
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

    def plot_iterations(f, a0, b0, points, title):
        x = np.linspace(a0, b0, 512)
        y = f(x)
        plt.figure(figsize=(10, 6))
        plt.plot(x, y, label="f(x)")
        plt.plot(
            points, [f(p) for p in points], "ro", label="Предполагаемые экстремумы"
        )
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.legend()
        plt.grid(True)
        plt.show()


def min_example(f, a0, b0, epsilon):
    s = Solution1
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

    # Визуализация для метода дихотомии
    s.plot_iterations(
        f,
        a0,
        b0,
        points_dichotomy,
        "Метод дихотомии: положение предполагаемого экстремума",
    )

    # Визуализация для метода золотого сечения
    s.plot_iterations(
        f,
        a0,
        b0,
        points_golden,
        "Метод золотого сечения: положение предполагаемого экстремума",
    )


def max_example(f, a0, b0, epsilon):
    s = Solution1
    x_min_dichotomy, intervals_dichotomy, points_dichotomy = s.dichotomy_method(
        f, a0, b0, epsilon
    )
    x_min_golden, intervals_golden, points_golden = s.golden_section_method(
        f, a0, b0, epsilon
    )
    print(
        f"Максимум методом дихотомии: x = {x_min_dichotomy}, f(x) = {f(x_min_dichotomy)}"
    )
    print(
        f"Максимум методом золотого сечения: x = {x_min_golden}, f(x) = {f(x_min_golden)}"
    )
    # Визуализация для метода дихотомии
    s.lplot_iterations(
        f,
        a0,
        b0,
        points_dichotomy,
        "Метод дихотомии: положение предполагаемого инфинума",
    )

    # Визуализация для метода золотого сечения
    s.plot_iterations(
        f,
        a0,
        b0,
        points_golden,
        "Метод золотого сечения: положение предполагаемого инфинума",
    )


def main():
    def mgch(x: float, *args) -> float:  # многочлен
        res = 0
        for i, k in enumerate(args):
            res += x**i * k
        return res

    def f1(x: float) -> float:
        return mgch(x, 1, 3, -5, 1)

    def f2(x: float) -> float:
        return mgch(x, -1, -3, 5, -1)

    a0 = 0
    b0 = 5
    epsilon = 1e-5

    # Найдем минимум
    min_example(f1, a0, b0, epsilon)

    # Найдем максимум
    max_example(f1, a0, b0, epsilon)


if __name__ == "__main__":
    main()
