from types import FunctionType
from math import sqrt
import matplotlib.pyplot as plt
import unittest

import numpy as np  # для визуализации (1 применение строка:89)


class Solution1:
    def dichotomy_method(
        f: FunctionType,
        a0: float,
        b0: float,
        epsilon: float = 1e-4,
        delta: float = 1e-4,
        max_iterations: int = 1000,
        find_min=True,
    ) -> tuple[float, list[tuple[float, float]], list[float]]:
        """
        Нахождение min or max непрерывной функции в [a0, b0] методом дехотомии

        Input:
            f - Исследуемая функция (непрерывна на [a0, b0]
            [a0, b0] - границы множества на котором проводим поиск
            epsilon - если на каком то этапе |a - b| < epsilon => выходим, требуемая точность
            delta - очень малое число, нужное для разделения точки по середине
            max_iterations - максимально кол-во итераций
        Return:
            X в котором f приблежино к max или к min, список промежутков, список точек минимума.

        """
        a = a0
        b = b0
        intervals = [(a, b)]
        x_points = [(a + b) / 2]
        need_break = False
        for k in range(max_iterations):
            mid = (a + b) / 2
            yk = mid - delta
            zk = mid + delta
            f_yk = f(yk)
            f_zk = f(zk)

            if find_min:
                if f_yk <= f_zk:
                    b = zk
                else:
                    a = yk
            else:
                if f_yk < f_zk:
                    a = zk
                else:
                    b = yk

            intervals.append((a, b))
            x_points.append((a + b) / 2)

            if b - a < epsilon * epsilon:
                break
        x_star = (a + b) / 2
        return x_star, intervals, x_points

    def golden_section_method(
        f: FunctionType,
        a0: float,
        b0: float,
        epsilon: float = 1e-4,
        max_iterations: int = 1000,
        find_min: bool = True,
    ) -> tuple[float, list[tuple[float, float]], list[float]]:
        """
        Нахождение min or max непрерывной функции в [a0, b0] методом золотого сечения

        Input:
            f - Исследуемая функция (непрерывна на [a0, b0]
            [a0, b0] - границы множества на котором проводим поиск
            epsilon - если на каком то этапе |a - b| < epsilon => выходим, требуемая точность
            max_iterations - максимально кол-во итераций
        Return:
            X в котором f приблежино к max или к min, список промежутков, список точек минимума.

        """
        phi = (1 + sqrt(5)) / 2  # Золоточе сечение, число фи
        resphi = 2 - phi

        a = a0
        b = b0
        intervals = [(a, b)]
        x_points = [(a + b) / 2]

        # Инициализация точек
        y = a + resphi * (b - a)
        z = b - resphi * (b - a)
        f_y = f(y)
        f_z = f(z)

        for k in range(max_iterations):
            if (f_y <= f_z and find_min) or (f_y >= f_z and not find_min):
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

            if b - a < epsilon * epsilon:
                break
        x_star = (a + b) / 2
        return x_star, intervals, x_points

    def plot_iterations(f: FunctionType, a0: float, b0: float, points: int, title: str):
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


def mgch(x: float, *args) -> float:  # многочлен
    res = 0
    for i, k in enumerate(args):
        res += x**i * k
    return res


class TestSolution1(unittest.TestCase):
    def setUp(self):
        self.s = Solution1

    def test_dichotomy_method(self):
        epsilon = 1e-4
        self.assertAlmostEqual(
            self.s.dichotomy_method(
                lambda x: mgch(x, 1, 3, -5, 1), 0, 1000, epsilon=epsilon
            )[0],
            3.0,
            delta=epsilon,
        )
        self.assertAlmostEqual(
            self.s.dichotomy_method(
                lambda x: mgch(x, -1, -3, 5, -1),
                0,
                1000,
                find_min=False,
                epsilon=epsilon,
            )[0],
            3.0,
            delta=epsilon,
        )

    def test_golden_section_method(self):
        epsilon = 1e-4
        self.assertAlmostEqual(
            self.s.golden_section_method(
                lambda x: mgch(x, 2, 6, -10, 2), 0, 10, epsilon=epsilon
            )[0],
            3,
            delta=epsilon,
        )
        self.assertAlmostEqual(
            self.s.golden_section_method(
                lambda x: mgch(x, 2, 6, -10, 2), 0, 4, epsilon=epsilon, find_min=False
            )[0],
            1 / 3,
            delta=epsilon,
        )


def min_example(f: FunctionType, a0: float, b0: float):
    s = Solution1
    x_min_dichotomy, intervals_dichotomy, points_dichotomy = s.dichotomy_method(
        f, a0, b0
    )
    x_min_golden, intervals_golden, points_golden = s.golden_section_method(f, a0, b0)
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


def max_example(f: FunctionType, a0: float, b0: float):
    s = Solution1
    x_min_dichotomy, intervals_dichotomy, points_dichotomy = s.dichotomy_method(
        f, a0, b0, find_min=False
    )
    x_min_golden, intervals_golden, points_golden = s.golden_section_method(
        f, a0, b0, find_min=False
    )
    print(
        f"Максимум методом дихотомии: x = {x_min_dichotomy}, f(x) = {f(x_min_dichotomy)}"
    )
    print(
        f"Максимум методом золотого сечения: x = {x_min_golden}, f(x) = {f(x_min_golden)}"
    )
    # Визуализация для метода дихотомии
    s.plot_iterations(
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
    unittest.main()

    def f1(x: float) -> float:
        return mgch(x, 1, 3, -5, 1)

    def f2(x: float) -> float:
        return mgch(x, -1, -3, 5, -1)

    def f3(x: float) -> float:
        return mgch(x, 0, 0, 1)

    a0 = 0
    b0 = 5

    # Найдем минимум
    min_example(f1, a0, b0)

    # Найдем максимум
    max_example(f2, a0, b0)


if __name__ == "__main__":
    main()
