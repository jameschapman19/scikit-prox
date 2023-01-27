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
        tol=1e-9,
        learning_rate=0.01,
            g=None,
    ):
        """


        Parameters
        ----------
        fit_intercept : bool, optional
            Whether to calculate the intercept for this model. If set
            to False, no intercept will be used in calculations
            (e.g. data is expected to be already centered).
        copy_X : bool, optional, default True
            If True, X will be copied; else, it may be overwritten.
        proximal : str, optional
            The proximal operator to use. Default is 'Dummy'.
        proximal_params : dict, optional
            The parameters for the proximal operator. Default is None.
        sigma : float, optional
            Multiplicative coefficient of the penalty. Default is 1.0.
        rho : float, optional
            Balancing between sparsity of :math:`L_1` and grouping of :math:`L_{2,1}`. Default is 0.8.
        radius : float, optional
            The radius of the ball for the proximal operator. Default is 1.0.
        gamma : float, optional
            Regularization parameter for Log. Default is 1.3.
        delta : float, optional
            Regularization parameter for Log1. Default is 1e-10.
        positive : bool, optional
            Whether to constrain the coefficients to be positive. Default is False.
        max_iter : int, optional
            The maximum number of iterations. Default is 1000.
        tol : float, optional
            The tolerance for the optimization. Default is 1e-4.
        learning_rate : float, optional
            The learning rate for the optimization. Default is 0.01.
        g : :obj:`np.ndarray`, optional
            Vector to be subtracted. Default is None.
        """
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
        self.g=g

    def fit(self, X, y, sample_weight=None):
        """
        Fit model.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.
               parameter *sample_weight* support to LinearRegression.
        Returns
        -------
        self : object
            Fitted Estimator.
        """
        self._validate_params()

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
            if np.linalg.norm(grad) < self.tol:
                break
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