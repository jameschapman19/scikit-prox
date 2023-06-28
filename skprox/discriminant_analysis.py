import warnings

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, _class_means, \
    _class_cov, _cov
from sklearn.utils._array_api import get_namespace
from sklearn.utils.multiclass import unique_labels

from skprox._gep import _GEPMixin
from skprox._proximal_operators import _PGDMixin


class RegularisedLinearDiscriminantAnalysis(LinearDiscriminantAnalysis, _GEPMixin, _PGDMixin):
    def __init__(
            self,
            n_components=None,
            priors=None,
            shrinkage=None,
            store_covariance=False,
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
            nesterov=False,
    ):
        super().__init__(
            n_components=n_components,
            priors=priors,
            shrinkage=shrinkage,
            solver='eigen',
            store_covariance=store_covariance,
            tol=tol,
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

    def fit(self, X, y):
        xp, _ = get_namespace(X)

        X, y = self._validate_data(
            X, y, ensure_min_samples=2, dtype=[xp.float64, xp.float32]
        )
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = self.classes_.shape[0]

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        if self.priors is None:  # estimate priors from sample
            _, cnts = xp.unique_counts(y)  # non-negative ints
            self.priors_ = xp.astype(cnts, xp.float64) / float(y.shape[0])
        else:
            self.priors_ = xp.asarray(self.priors)

        if xp.any(self.priors_ < 0):
            raise ValueError("priors must be non-negative")

        if xp.abs(xp.sum(self.priors_) - 1.0) > 1e-5:
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(n_classes - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components
        self.ndim = self.n_components
        self.dims = (X.shape[1], self.n_components)
        self.proximal = self._get_proximal()
        self.means_ = _class_means(X, y)
        A, B = self._get_AB(X, y)
        coef_ = np.random.normal(0, 1, size=self.dims)
        coef_ /= np.linalg.norm(coef_, axis=0)
        evecs = self._proximal_gradient_descent(A, B, coef_).T
        evals = np.diag(evecs.T @ A @ evecs)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
                                         : self._max_components
                                         ]
        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def _get_AB(self, X, y):
        self.covariance_ = _class_cov(
            X, y, self.priors_, self.shrinkage, self.covariance_estimator
        )
        Sw = self.covariance_  # within scatter
        St = _cov(X, self.shrinkage, self.covariance_estimator)  # total scatter
        Sb = St - Sw  # between scatter
        return Sb, Sw
