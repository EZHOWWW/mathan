from types import FunctionType
from math import cos
import numpy as np


class Solution2:
    def bisection_method(f, a, b, epsilon=1e-5, max_iterations=1000):
        if f(a) * f(b) > 0:
            raise ValueError(f"Значения в a и b должны быть разных знаков({a}, {b})")
        elif f(a) * f(b) == 0:
            if f(a) == 0:
                return a, 0
            else:
                return b, 0

        iteration = 0
        while (b - a) / 2.0 > epsilon and iteration < max_iterations:
            c = (a + b) / 2.0
            fc = f(c)
            if fc == 0:
                return c, iteration
            elif f(a) * fc < 0:
                b = c
            else:
                a = c
            iteration += 1
        return (a + b) / 2.0, iteration


def main():
    s = Solution2

    def f1(x):
        return x**3 - x - 2

    def f2(x):
        return cos(x) - x

    # Поиск корней
    root1, iterations1 = s.bisection_method(f1, 1, 2)
    print(f"Корень f1 на [1, 2]: {root1} за {iterations1} итераций.")

    root2, iterations2 = s.bisection_method(f2, 0, 1)
    print(f"Корень f2 на [0, 1]: {root2} за {iterations2} итераций.")

    # Известные истинные корни
    true_roots = {"f1": 1.52138, "f2": 0.739085}

    # Найденные корни с epsilon = 1e-5
    found_roots = {"f1": root1, "f2": root2}

    # Рассчёт СКО
    errors = []
    for key in true_roots:
        error = (found_roots[key] - true_roots[key]) ** 2
        errors.append(error)

    rmse = np.sqrt(np.mean(errors))
    print(f"Среднеквадратичное отклонение найденных корней от истинных: {rmse}")

    # Определение функции без корня на заданном интервале
    def f3(x):
        return x**2 + 1  # Нет корней на действительных числах

    try:
        root3, result3 = s.bisection_method(f3, 0, 1)
    except ValueError as e:
        print(f"f3 на [0, 1]: Нет,", e)
    else:
        print(f"Корень f3 на [0, 1]: {root3} за {result3} итераций.")


if __name__ == "__main__":
    main()
