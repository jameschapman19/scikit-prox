import numpy as np
from sklearn.linear_model._base import _preprocess_data, _rescale_data, LinearRegression
from sklearn.utils.validation import _check_sample_weight

from skprox.proximal_operators import _proximal_operators


class RegularisedRegression(LinearRegression):
    def __init__(
        self,
        fit_intercept=True,
        copy_X=True,
        proximal="Dummy",
        proximal_params=None,
        sigma=1.0,
        rho=0.8,
        radius=1.0,
        gamma=1.3,
        delta=1e-10,
        positive=False,
        max_iter=1000,
        tol=1e-4,
        learning_rate=0.01,
    ):
        super().__init__(fit_intercept=fit_intercept, copy_X=copy_X, positive=positive)
        self.proximal = proximal
        self.proximal_params = proximal_params
        self.sigma = sigma
        self.max_iter = max_iter
        self.tol = tol
        self.learning_rate = learning_rate
        self.rho = rho
        self.radius = radius
        self.gamma = gamma
        self.delta = delta

    def fit(self, X, y, sample_weight=None):
        # self._validate_params()

        # Convert data
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=("csr", "csc"),
            multi_output=True,
            y_numeric=True,
        )
        sample_weight = _check_sample_weight(
            sample_weight, X, dtype=X.dtype, only_non_negative=True
        )

        X, y, X_offset, y_offset, X_scale = _preprocess_data(
            X,
            y,
            fit_intercept=self.fit_intercept,
            copy=self.copy_X,
            sample_weight=sample_weight,
        )

        # Sample weight can be implemented via a simple rescaling.
        X, y, sample_weight_sqrt = _rescale_data(X, y, sample_weight)
        n_samples, n_features = X.shape
        if y.ndim == 1:
            self.ndim = 1
            self.dims = (n_features,)
        else:
            self.ndim = y.shape[1]
            self.dims = (n_features, y.shape[1])
        self.proximal = self._get_proximal()
        # optimise by proximal gradient descent
        self.coef_ = self._proximal_gradient_descent(X, y).T
        self._set_intercept(X_offset, y_offset, X_scale)
        return self

    def _proximal_gradient_descent(self, X, y):
        n_samples, n_features = X.shape
        if y.ndim == 1:
            coef_ = np.zeros((n_features,))
        else:
            coef_ = np.zeros((n_features, y.shape[1]))
        for i in range(self.max_iter):
            grad = X.T @ (X @ coef_ - y) / X.shape[0]
            coef_ = self.proximal.prox(
                (coef_.flatten() - self.learning_rate * grad.flatten()), self.sigma
            ).reshape(coef_.shape)
        return coef_

    def _get_proximal(self):
        if callable(self.proximal):
            params = self.proximal_params or {}
        else:
            params = {
                "ndim": self.ndim,
                "sigma": self.sigma,
                "isotropic": True,
                "positive": self.positive,
                "dims": self.dims,
                "rho": self.rho,
                "dim": self.dims,
                "radius": self.radius,
                "gamma": self.gamma,
                "delta": self.delta,
            }
        return _proximal_operators(self.proximal, **params)


if __name__ == "__main__":
    from sklearn.datasets import make_regression
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split

    X, y, t = make_regression(
        n_samples=100,
        n_features=200,
        n_informative=10,
        n_targets=5,
        random_state=1,
        coef=True,
    )
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )
    reg = RegularisedRegression(proximal="L21", sigma=0.1, max_iter=10000)
    reg.fit(x_train, y_train)
    print(reg.score(x_test, y_test))
    lreg = LinearRegression()
    lreg.fit(x_train, y_train)
    print(lreg.score(x_test, y_test))
    print()
