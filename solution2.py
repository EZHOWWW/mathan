from types import FunctionType
from math import cos


class Solution2:
    def bisection_method(
        f: FunctionType, a: float, b: float, epsilon=1e-5, max_iterations=1000
    ) -> tuple[float, int]:
        """
        Метод дихотомии для поиска корня функции f на интервале [a, b].

        :param f: Исследуемая функция.
        :param a: Левая граница интервала.
        :param b: Правая граница интервала.
        :param epsilon: Точность нахождения корня.
        :param max_iterations: Максимальное количество итераций.
        :return: Приближенное значение корня, количество итераций.
        """
        if f(a) * f(b) >= 0:
            raise ValueError(
                "Метод дихотомии не применим: f(a) и f(b) должны иметь противоположные знаки."
            )

        iteration = 0
        while (b - a) / 2.0 > epsilon and iteration < max_iterations:
            c = (a + b) / 2.0
            fc = f(c)
            # Проверка, является ли средняя точка корнем
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


if __name__ == "__main__":
    main()
