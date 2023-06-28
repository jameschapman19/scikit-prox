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
    "TVL1": frozenset(["sigma", "shape", "l1_ratio"]),
}


def _proximal_operators(proximal, filter_params=True, **params):
    if proximal in PROXIMAL_OPERATORS:
        if filter_params:
            params = {k: params[k] for k in params if k in PROXIMAL_PARAMS[proximal]}
        return PROXIMAL_OPERATORS[proximal](**params)
    elif callable(proximal):
        return proximal(**params)
