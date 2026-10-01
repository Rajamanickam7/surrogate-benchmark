"""Surrogate (metamodel) factories.

Inputs are scaled to [0, 1]^d before fitting so that kernel length scales and
polynomial coefficients are comparable across dimensions.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures


def make_rsm(degree=2):
    """Response surface: full quadratic polynomial fitted by least squares."""
    return make_pipeline(MinMaxScaler(), PolynomialFeatures(degree), LinearRegression())


def make_kriging(dim, seed=None):
    """Kriging / Gaussian process with an anisotropic Matern-5/2 kernel.

    Hyperparameters (signal variance, one length scale per input, nugget) are
    fitted by maximising the log marginal likelihood, with restarts to avoid
    poor local optima. The WhiteKernel is the nugget: it lets the model stop
    interpolating exactly, which keeps the covariance matrix well conditioned.
    """
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
