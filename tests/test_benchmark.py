import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from functions import PROBLEMS, branin, rosenbrock
from metrics import r2
from sampling import lhs
from surrogates import make_kriging, make_rsm


def test_known_minima():
    for x in [(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)]:
        assert abs(branin(np.array([x]))[0] - 0.397887) < 1e-5
    assert rosenbrock(np.array([[1.0, 1.0]]))[0] == 0.0


def test_lhs_inside_bounds_and_stratified():
    bounds = PROBLEMS["branin"]["bounds"]
    X = lhs(bounds, 20, seed=0)
    assert X.shape == (20, 2)
    assert np.all(X >= bounds[:, 0]) and np.all(X <= bounds[:, 1])
    # Latin Hypercube: exactly one point per stratum in each dimension
    unit = (X - bounds[:, 0]) / (bounds[:, 1] - bounds[:, 0])
    for d in range(2):
        assert sorted(np.floor(unit[:, d] * 20).astype(int)) == list(range(20))


def test_rsm_recovers_a_quadratic_exactly():
    bounds = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    X = lhs(bounds, 15, seed=1)
    f = lambda X: 1 + 2 * X[:, 0] - X[:, 1] + 3 * X[:, 0] * X[:, 1] + X[:, 1] ** 2
    X_test = lhs(bounds, 100, seed=2)
    assert r2(f(X_test), make_rsm().fit(X, f(X)).predict(X_test)) > 0.999999


def test_kriging_nearly_interpolates_training_data():
    problem = PROBLEMS["branin"]
    X = lhs(problem["bounds"], 20, seed=0)
    y = problem["f"](X)
    model = make_kriging(2, seed=0).fit(X, y)
    assert np.max(np.abs(model.predict(X) - y)) < 1e-2 * np.std(y)
