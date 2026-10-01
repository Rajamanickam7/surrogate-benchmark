import numpy as np
from scipy.optimize import minimize

from sampling import lhs
from surrogates import SURROGATES


class CountedFunction:
    def __init__(self, f):
        self.f = f
        self.n_evals = 0
        self.best = np.inf

    def __call__(self, x):
        self.n_evals += 1
        y = float(self.f(np.asarray(x)[None, :])[0])
        self.best = min(self.best, y)
        return y


def run_direct(problem, method, seed, max_evals=500):
    rng = np.random.default_rng(seed)
    bounds = problem["bounds"]
    x0 = rng.uniform(bounds[:, 0], bounds[:, 1])
    f = CountedFunction(problem["f"])
    options = {"maxfev": max_evals} if method == "Nelder-Mead" else {"maxfun": max_evals}
    minimize(f, x0, method=method, bounds=bounds, options=options)
    return {"n_evals": f.n_evals, "best_f": f.best, "gap": f.best - problem["f_min"]}


def run_surrogate_opt(problem, surrogate_name, n_train, seed, n_starts=10):
    bounds = problem["bounds"]
    X = lhs(bounds, n_train, seed=seed)
    y = problem["f"](X)
    model = SURROGATES[surrogate_name](len(bounds), seed).fit(X, y)

    def predict(x):
        return float(model.predict(np.asarray(x)[None, :])[0])

    starts = lhs(bounds, n_starts, seed=seed + 1000)
    results = [minimize(predict, x0, method="L-BFGS-B", bounds=bounds) for x0 in starts]
    x_best = min(results, key=lambda r: r.fun).x

    f_true = float(problem["f"](x_best[None, :])[0])
    best_f = min(f_true, float(y.min()))
    return {"n_evals": n_train + 1, "best_f": best_f, "gap": best_f - problem["f_min"]}
