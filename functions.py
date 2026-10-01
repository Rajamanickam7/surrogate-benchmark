import numpy as np


def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c = 1.0, 5.1 / (4 * np.pi**2), 5 / np.pi
    r, s, t = 6.0, 10.0, 1 / (8 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s


def rosenbrock(X):
    X = np.atleast_2d(X)
    x, y = X[:, 0], X[:, 1]
    return (1 - x) ** 2 + 100 * (y - x**2) ** 2


PROBLEMS = {
    "branin": {
        "f": branin,
        "bounds": np.array([[-5.0, 10.0], [0.0, 15.0]]),
        "f_min": 0.397887,
    },
    "rosenbrock": {
        "f": rosenbrock,
        "bounds": np.array([[-2.0, 2.0], [-1.0, 3.0]]),
        "f_min": 0.0,
    },
}
