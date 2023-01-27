import numpy as np

from skprox.linear_model import RegularisedRegression


def test_L0_regression():
    from sklearn.datasets import make_regression
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
    reg = RegularisedRegression(proximal="L0", sigma=0.1, max_iter=10000)
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
    from sklearn.datasets import make_regression
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
    reg = RegularisedRegression(proximal="L1", sigma=0.1, max_iter=10000)
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
    from sklearn.datasets import make_regression
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
    reg = RegularisedRegression(proximal="NuclearBall", radius=10, max_iter=10000)
    reg.fit(x_train, y_train)

    # test nuclearball has norm of coefficients about the same as the radius
    assert np.isclose(np.linalg.norm(reg.coef_), 10, atol=0.1)

    # test nuclear has the same number of coefficients as the number of targets
    assert reg.coef_.shape[0] == 5

    # test nuclear has the same number of coefficients as the number of features
    assert reg.coef_.shape[1] == 200
