from abc import abstractmethod

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

from skprox.operators import Dummy, TV, TVL1

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
    "TVL1": TVL1,
}

PROXIMAL_PARAMS = {
    "Dummy": (),
    "L0": frozenset(["sigma"]),
    "L0Ball": frozenset(["radius"]),
    "L1": frozenset(["sigma"]),
    "L1Ball": frozenset(["n", "radius"]),
    "L2": frozenset(["sigma"]),
    "L21": frozenset(["ndim", "sigma"]),
    "L21_plus_L1": frozenset(["sigma", "rho"]),
    "TV": frozenset(["sigma", "isotropic", "dims"]),
    "Nuclear": frozenset(["dim", "sigma"]),
    "NuclearBall": frozenset(["dims", "radius"]),
    "Log": frozenset(["sigma", "gamma"]),
    "Log1": frozenset(["sigma", "delta"]),
    "Euclidean": frozenset(["sigma"]),
    "TVL1": frozenset(["sigma", "rho"]),
}


class _PGDMixin:
    """
    This class will be used by all proximal gradient descent algorithms.

    We would like it to flexibly allow for different gradient descent algorithms including gradient descent with momentum, Nesterov's accelerated gradient descent, and stochastic gradient descent.
    """
    def _proximal_gradient_descent(self, X, y, coef_):
        """
        This is the main proximal gradient descent algorithm.
        """


    @abstractmethod
    def _compute_gradient(self, X, y, coef_):
        pass

    @abstractmethod
    def _objective(self, X, y, coef_):
        pass