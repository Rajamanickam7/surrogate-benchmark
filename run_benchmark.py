from pathlib import Path
import time
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from functions import PROBLEMS
from metrics import all_metrics
from optimizers import run_direct, run_surrogate_opt
from sampling import lhs
from surrogates import SURROGATES

N_TRAIN = [10, 20, 40, 80]
SEEDS = range(5)
N_TEST = 500
DIRECT_METHODS = ["Nelder-Mead", "L-BFGS-B"]
OUT = Path("results")

warnings.filterwarnings("ignore", module="sklearn.gaussian_process")


def accuracy_study():
    rows = []
    for pname, problem in PROBLEMS.items():
        bounds = problem["bounds"]
        X_test = lhs(bounds, N_TEST, seed=12345)
        y_test = problem["f"](X_test)
        for sname, make in SURROGATES.items():
            for n in N_TRAIN:
                for seed in SEEDS:
                    X = lhs(bounds, n, seed=seed)
                    y = problem["f"](X)
                    t0 = time.perf_counter()
                    model = make(len(bounds), seed).fit(X, y)
                    fit_time = time.perf_counter() - t0
                    m = all_metrics(y_test, model.predict(X_test))
                    rows.append({"problem": pname, "surrogate": sname, "n_train": n,
                                 "seed": seed, "fit_time_s": fit_time, **m})
    return pd.DataFrame(rows)


def optimization_study():
    rows = []
    for pname, problem in PROBLEMS.items():
        for seed in SEEDS:
            for method in DIRECT_METHODS:
                r = run_direct(problem, method, seed)
                rows.append({"problem": pname, "approach": f"direct:{method}", "seed": seed, **r})
            for sname in SURROGATES:
                for n in N_TRAIN:
                    r = run_surrogate_opt(problem, sname, n, seed)
                    rows.append({"problem": pname, "approach": f"surrogate:{sname}@{n}",
                                 "seed": seed, **r})
    return pd.DataFrame(rows)


def plot_accuracy(acc):
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(10, 4))
    for ax, pname in zip(axes, PROBLEMS):
        sub = acc[acc.problem == pname]
        for sname, g in sub.groupby("surrogate"):
            stats = g.groupby("n_train")["nrmse"].agg(["mean", "std"])
            ax.errorbar(stats.index, stats["mean"], yerr=stats["std"], marker="o",
                        capsize=3, label=sname)
        ax.set(title=pname, xlabel="training samples", ylabel="NRMSE (test)",
               xscale="log", yscale="log")
        ax.set_xticks(N_TRAIN, labels=[str(n) for n in N_TRAIN])
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_samples.png", dpi=150)


def main():
    OUT.mkdir(exist_ok=True)
    acc = accuracy_study()
    acc.to_csv(OUT / "accuracy.csv", index=False)
    plot_accuracy(acc)

    opt = optimization_study()
    opt.to_csv(OUT / "optimization.csv", index=False)

    pd.set_option("display.width", 120)
    print("=== Surrogate accuracy (mean over seeds) ===")
    print(acc.groupby(["problem", "surrogate", "n_train"])[["nrmse", "r2", "fit_time_s"]]
          .mean().round(4).to_string())
    print("\n=== Optimization: evaluations vs gap to true minimum (median over seeds) ===")
    print(opt.groupby(["problem", "approach"])[["n_evals", "gap"]]
          .median().round(4).to_string())


if __name__ == "__main__":
    main()
