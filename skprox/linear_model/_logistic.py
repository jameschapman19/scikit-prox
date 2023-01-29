from sklearn.linear_model import LogisticRegression
from sklearn.linear_model._base import _preprocess_data, _rescale_data
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _check_sample_weight
import numpy as np
from skprox.proximal_operators import _proximal_operators


class RegularisedLogisticRegression(LogisticRegression):
    def __init__(
            self,
            fit_intercept=True,
            copy_X=True,
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
            nesterov=False,
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
        super().__init__(fit_intercept=fit_intercept,max_iter=max_iter, tol=tol, random_state=random_state)
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
        self.copy_X = copy_X
    def fit(self, X, y, sample_weight=None):
        self._validate_params()

        # Convert data
        X, y = self._validate_data(
            X,
            y,
            accept_sparse=("csr", "csc"),
            multi_output=True,
            y_numeric=True,
        )
        check_classification_targets(y)
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        classes_ = self.classes_
        if n_classes < 2:
            raise ValueError(
                "This solver needs samples of at least 2 classes"
                " in the data, but the data contains only one"
                " class: %r"
                % classes_[0]
            )
        if len(self.classes_) == 2:
            n_classes = 1
            classes_ = classes_[1:]

        if self.warm_start:
            warm_start_coef = getattr(self, "coef_", None)
        else:
            warm_start_coef = None
        if warm_start_coef is not None and self.fit_intercept:
            warm_start_coef = np.append(
                warm_start_coef, self.intercept_[:, np.newaxis], axis=1
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
            self.dims = (n_features,1)
        else:
            self.ndim = y.shape[1]
            self.dims = (n_features, y.shape[1])
        self.proximal = self._get_proximal()
        # optimise by proximal gradient descent
        self.coef_ = self._proximal_gradient_descent(X, y).T
        #ensure self.coef_ is a 2D array
        if self.coef_.ndim == 1:
            self.coef_ = self.coef_[np.newaxis,:]
        self.intercept_ = y_offset - X_offset.dot(self.coef_.T)
        return self

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

    def _proximal_gradient_descent(self, X, y):
        self.track = []
        n_samples, n_features = X.shape
        if y.ndim == 1:
            coef_ = np.zeros((n_features,))
        else:
            coef_ = np.zeros((n_features, y.shape[1]))
        old_coef_ = coef_
        # proximal gradient descent with a backtracking line search
        for i in range(self.max_iter):
            if self.nesterov:
                v=coef_+((i-1)/(i+2))*(coef_-old_coef_)
                old_coef_=coef_
            else:
                v=coef_
            # compute gradient
            grad = self._compute_gradient(X, y, v)
            coef_ = self.proximal.prox(
                (v.flatten() - self.learning_rate * grad.flatten()), self.sigma
            ).reshape(coef_.shape)
            self.track.append(self._objective(X, y, coef_))
        return coef_.T

    def _objective(self, X, y, coef_):
        y_hat = 1. / (1. + np.exp(-np.dot(X, coef_)))  # predicted y by the LR model
        J = np.mean(-y * np.log2(y_hat) - (1 - y) * np.log2(1 - y_hat))  # the binary cross entropy loss function
        return J + self.proximal(coef_)

    def _compute_gradient(self, X, y, coef_):
        y_hat = 1. / (1. + np.exp(-np.dot(X, coef_)))  # predicted y by the LR model
        return np.dot(y_hat - y,X)/X.shape[0] # the gradient of the loss function

def sigmoid(x):
    return 1/(1+np.exp(-x))