from functools import partial

import numpy as np
from pyproximal import (
    L0,
    L0Ball,
    L1,
    L1Ball,
    L2,
    L21,
    L21_plus_L1,
    Nuclear,
    NuclearBall,
    Log,
    Log1,
    Euclidean,
    EuclideanBall,
)
from pyproximal.ProxOperator import _check_tau
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman


class Dummy:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, x):
        return 0

    def prox(self, x, tau):
        return x


class TV:
    def __init__(self, sigma, dims, isotropic=True, max_iter=1000):
        self.sigma = sigma
        self.isotropic = isotropic
        self.max_iter = max_iter
        self.count = 0
        self.dims = dims

    def _increment_count(func):
        """Increment counter"""

        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)

        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        x = x.reshape(self.dims)
        if self.isotropic:
            return denoise_tv_chambolle(
                x, weight=tau * self.sigma / 2, max_num_iter=self.max_iter
            )
        else:
            return denoise_tv_bregman(
                x,
                weight=tau * self.sigma / 2,
                isotropic=self.isotropic,
                max_num_iter=self.max_iter,
            )

    def __call__(self, x):
        m = np.zeros_like(x)
        for d in range(x.ndim):
            diff = np.gradient(x, axis=d)
            if self.isotropic:
                m += diff**2
            else:
                m += np.abs(diff)
        if not self.isotropic:
            m = np.sqrt(m)
        return m.sum()


PROXIMAL_OPERATORS = {
    "Dummy": Dummy,
    "L0": L0,
    "L0Ball": L0Ball,
    "L1": L1,
    "L1Ball": L1Ball,
    "L2": L2,
    "L21": L21,
    "L21_plus_L1": L21_plus_L1,
    "TV": TV,
    "Nuclear": Nuclear,
    "NuclearBall": NuclearBall,
    "Log": Log,
    "Log1": Log1,
    "Euclidean": Euclidean,
    "EuclideanBall": EuclideanBall,
}

PROXIMAL_PARAMS = {
    "Dummy": (),
    "L0": frozenset(["sigma"]),
    "L0Ball": frozenset(["radius"]),
    "L1": frozenset(["sigma"]),
    "L1Ball": frozenset(["sigma","g"]),
    "L2": frozenset(["sigma"]),
    "L21": frozenset(["ndim", "sigma"]),
    "L21_plus_L1": frozenset(["sigma", "rho"]),
    "TV": frozenset(["sigma", "isotropic", "dims"]),
    "Nuclear": frozenset(["dim", "sigma"]),
    "NuclearBall": frozenset(["dims", "radius"]),
    "Log": frozenset(["sigma", "gamma"]),
    "Log1": frozenset(["sigma", "delta"]),
    "Euclidean": frozenset(["sigma"]),
}


def _proximal_operators(proximal, filter_params=True, **params):
    if proximal in PROXIMAL_OPERATORS:
        if filter_params:
            params = {k: params[k] for k in params if k in PROXIMAL_PARAMS[proximal]}
        return PROXIMAL_OPERATORS[proximal](**params)
    elif callable(proximal):
        return partial(proximal, **params)
