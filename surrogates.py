import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def make_rsm(degree=2):
    return make_pipeline(MinMaxScaler(), PolynomialFeatures(degree), LinearRegression())


def make_kriging(dim, seed=None):
    kernel = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(length_scale=np.ones(dim), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(1e-6, (1e-10, 1e-1))
    )
    gp = GaussianProcessRegressor(
        kernel, normalize_y=True, n_restarts_optimizer=5, random_state=seed
    )
    return make_pipeline(MinMaxScaler(), gp)


SURROGATES = {
    "rsm_quadratic": lambda dim, seed: make_rsm(),
    "kriging": make_kriging,
}
