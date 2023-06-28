"""
Dykstra's composite algorithm for combining proximal operators.

This is a simple implementation of Dykstra's algorithm for combining proximal
operators. It is not optimized for speed, but rather for readability. It is
intended to be used as a building block for more complex algorithms.

References
----------
[1] Boyle, J. P.; Dykstra, R. L. (1986). A method for finding projections onto the intersection of convex sets in Hilbert spaces. Lecture Notes in Statistics. Vol. 37. pp. 28–47

[2] https://en.wikipedia.org/wiki/Dykstra%27s_projection_algorithm

[3] https://github.com/neurospin/pylearn-parsimony/blob/734437565cc3bdd0785786a433ad852421556668/parsimony/algorithms/proximal.py

@author: James Chapman
@email: james.chapman.19@ucl.ac.uk
"""

import numpy as np
from pyproximal import ProxOperator


class Composite(ProxOperator):
    def __init__(self, prox_ops, max_iter=5000, weights=None):
        """
        Dykstra's composite algorithm for combining proximal operators.

        Parameters
        ----------
        prox_ops : list of ProxOperator
        max_iter : int
        weights : list of float


        """
        super().__init__()
        self.prox_ops = prox_ops
        self.max_iter = max_iter
        self.weights = weights
        if self.weights is None:
            self.weights = [1. / float(len(prox_ops))] * (len(prox_ops))

    def prox(self, x, tau, **kwargs):
        x_new = np.zeros_like(x)
        z = []
        p = []
        for _ in self.prox_ops:
            z.append(x.copy())
            p.append(np.zeros_like(x))
        for it in range(self.max_iter):
            x_old = x_new.copy()
            x_new[:] = 0
            for i, prox_op in enumerate(self.prox_ops):
                p[i] = prox_op.prox(z[i], tau)
                x_new += p[i] * self.weights[i]
            if np.allclose(x_new, x_old):
                break
            for i, _ in enumerate(self.prox_ops):
                z[i] += x_new - p[i]
        return x_new

    def __call__(self, x):
        """
        Parameters
        ----------
        x : np.ndarray
            The point at which to evaluate the composite proximal operator.

        Returns
        -------
        val : float
            The value of the composite proximal operator at x.
        constraints_satisfied : bool
            True if all constraints are satisfied, False otherwise.


        """
        val = 0
        constraints_satisfied = True
        for prox_op in self.prox_ops:
            v = prox_op(x)
            if type(v) == np.bool_:
                constraints_satisfied &= v
            else:
                val += prox_op(x)
        if constraints_satisfied:
            return val
        else:
            return constraints_satisfied
