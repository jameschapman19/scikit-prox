import numbers

import numpy as np
from sklearn.decomposition import PCA
from sklearn.utils import check_scalar

from skprox._gep import _GEPMixin
from skprox._pgd import _PGDMixin


class PCA(PCA, _GEPMixin, _PGDMixin):
    def __init__(
            self,
            n_components=None,
            copy=True,
            whiten=False,
            svd_solver='auto',
            iterated_power='auto',
            random_state=None,
            proximal="Dummy",
            proximal_params=None,
            sigma=1.0,
            rho=0.8,
            radius=1.0,
            gamma=1.3,
            delta=1e-10,
            positive=False,
            max_iter=1000,
            tol=1e-9,
            learning_rate=0.01,
            g=None,
            nesterov=True,
    ):
        super().__init__(
            n_components=n_components,
            copy=copy,
            whiten=whiten,
            svd_solver=svd_solver,
            tol=tol,
            iterated_power=iterated_power,
            random_state=random_state,
        )
        self.proximal = proximal
        self.proximal_params = proximal_params
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.g = g
        self.nesterov = nesterov

        self.rho = rho
        self.radius = radius
        self.gamma = gamma
        self.delta = delta
        self.positive = positive

    def fit(self, X, y=None):
        """Fit the model with X.

                Parameters
                ----------
                X : array-like of shape (n_samples, n_features)
                    Training data, where `n_samples` is the number of samples
                    and `n_features` is the number of features.

                y : Ignored
                    Ignored.

                Returns
                -------
                self : object
                    Returns the instance itself.
                """
        check_scalar(
            self.n_oversamples,
            "n_oversamples",
            min_val=1,
            target_type=numbers.Integral,
        )

        self._fit(X)
        return self

    def _fit(self, X, y=None):
        X = self._validate_data(
            X, dtype=[np.float64, np.float32], ensure_2d=True, copy=self.copy
        )

        # Handle n_components==None
        if self.n_components is None:
            self.n_components = min(X.shape)

        self.ndim = self.n_components
        self.dims = (X.shape[1], self.n_components)
        self.proximal = self._get_proximal()
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        n_samples, n_features = X.shape
        coef_ = np.random.normal(0, 1, (n_features, self.n_components))
        coef_ /= np.linalg.norm(coef_, axis=0)
        A, B = self._get_AB(X, coef_)
        self.components_ = self._proximal_gradient_descent(A, B, coef_)

    def _get_AB(self, X, coef_):
        A = np.cov(X.T)
        B = np.eye(X.shape[1])
        return A, B
