"""Design of experiments: Latin Hypercube sampling inside box bounds."""
from scipy.stats import qmc


def lhs(bounds, n, seed=None):
    """Return n Latin Hypercube points, shape (n, d), scaled to bounds (d, 2)."""
    sampler = qmc.LatinHypercube(d=len(bounds), rng=seed)
    unit = sampler.random(n)
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])
