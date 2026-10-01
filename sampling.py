from scipy.stats import qmc


def lhs(bounds, n, seed=None):
    sampler = qmc.LatinHypercube(d=len(bounds), rng=seed)
    unit = sampler.random(n)
    return qmc.scale(unit, bounds[:, 0], bounds[:, 1])
