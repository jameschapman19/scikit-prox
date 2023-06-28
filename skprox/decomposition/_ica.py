from scipy.linalg import pinv
from sklearn.decomposition import FastICA
from sklearn.utils import check_random_state, as_float_array

from skprox._gep import _GEPMixin
import numpy as np

from skprox._pgd import _PGDMixin


class ICA(FastICA, _GEPMixin, _PGDMixin):
    def __init__(
            self,
            n_components=None,
            max_iter=200,
            tol=1e-4,
            w_init=None,
            random_state=None,
            proximal="Dummy",
            proximal_params=None,
            sigma=1.0,
            rho=0.8,
            radius=1.0,
            gamma=1.3,
            delta=1e-10,
            positive=False,
            max_iter_pgd=1000,
            tol_pgd=1e-9,
            learning_rate=0.01,
            g=None,
            nesterov=True,
    ):
        super().__init__(
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            w_init=w_init,
            random_state=random_state,
        )
        self.proximal = proximal
        self.proximal_params = proximal_params
        self.sigma = sigma
        self.max_iter_pgd = max_iter_pgd
        self.tol_pgd = tol_pgd
        self.learning_rate = learning_rate
        self.g = g
        self.nesterov = nesterov

        self.rho = rho
        self.radius = radius
        self.gamma = gamma
        self.delta = delta
        self.positive = positive

    def fit(self, X, y=None):
        self.ndim = self.n_components
        self.dims = (X.shape[1], self.n_components)
        self.proximal = self._get_proximal()
        XT = self._validate_data(
            X, copy=self._whiten, dtype=[np.float64, np.float32], ensure_min_samples=2
        ).T
        random_state = check_random_state(self.random_state)
        X1 = as_float_array(XT, copy=False)
        w_init = self.w_init
        if w_init is None:
            w_init = np.asarray(
                random_state.normal(size=(self.n_components, self.n_components)), dtype=X1.dtype
            )

        else:
            w_init = np.asarray(w_init)
            if w_init.shape != (self.n_components, self.n_components):
                raise ValueError(
                    "w_init has invalid shape -- should be %(shape)s"
                    % {"shape": (self.n_components, self.n_components)}
                )

        W = self._proximal_gradient_descent(X1, y, w_init)
        self.mixing_ = pinv(self.components_, check_finite=False)
        self._unmixing = W
        return self

    def _get_AB(self, X, y):
        B = np.cov(X.T)
        K = X @ X.T
        A = np.cov(np.diag(K) * X, rowvar=False) / np.trace(K) - np.trace(B) * B - 2 * B @ B
        return A, B
