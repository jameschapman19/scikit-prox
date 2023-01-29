import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from skprox.linear_model import RegularisedLinearRegression

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


def test_L0_logistic():
    reg = RegularisedLinearRegression(proximal="L0", sigma=0.1, max_iter=10000)
    reg.fit(x_train, y_train)
    # test L0 has some zero coefficients
    assert np.sum(reg.coef_ == 0) > 0

    # test L0 has some non-zero coefficients
    assert np.sum(reg.coef_ != 0) > 0

    # test L0 has the same number of coefficients as the number of targets
    assert reg.coef_.shape[0] == 5

    # test L0 has the same number of coefficients as the number of features
    assert reg.coef_.shape[1] == 200


def test_L1_regression():
    reg = RegularisedLinearRegression(proximal="L1", sigma=0.1, max_iter=10000)
    reg.fit(x_train, y_train)
    # test L1 has some zero coefficients
    assert np.sum(reg.coef_ == 0) > 0

    # test L1 has some non-zero coefficients
    assert np.sum(reg.coef_ != 0) > 0

    # test L1 has the same number of coefficients as the number of targets
    assert reg.coef_.shape[0] == 5

    # test L1 has the same number of coefficients as the number of features
    assert reg.coef_.shape[1] == 200


def test_nuclearball_regression():
    reg = RegularisedLinearRegression(proximal="NuclearBall", radius=10, max_iter=10000)
    reg.fit(x_train, y_train)

    # test nuclearball has norm of coefficients about the same as the radius
    assert np.isclose(np.linalg.norm(reg.coef_), 10, atol=0.1)

    # test nuclear has the same number of coefficients as the number of targets
    assert reg.coef_.shape[0] == 5

    # test nuclear has the same number of coefficients as the number of features
    assert reg.coef_.shape[1] == 200


def test_gridsearch():
    from sklearn.model_selection import GridSearchCV

    reg = GridSearchCV(
        RegularisedLinearRegression(max_iter=1000),
        param_grid={"proximal": ["L0", "L1"], "sigma": [1e-1, 1, 10]},
        cv=5,
    )
    reg.fit(x_train, y_train)

    # test gridsearch has the same number of coefficients as the number of targets
    assert reg.best_estimator_.coef_.shape[0] == 5

    # test gridsearch has the same number of coefficients as the number of features
    assert reg.best_estimator_.coef_.shape[1] == 200
