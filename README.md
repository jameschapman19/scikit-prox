# Scikit-Prox
The goal of this project is to implement a set of algorithms for solving the following optimization problem:
minimize f(x) + g(x) where f is a smooth function and g is a proximal operator. The proximal operator of a function g is defined as:
proxg(λx) = argmin y g(y) + 1/2λ‖y − x‖2

## Installation
To install the package, run the following command:
pip install scikit-prox

## Usage

### Example 1: Lasso
The following code solves the following optimization problem:
minimize 1/2‖Ax − b‖2 + λ‖x‖1

```python
import numpy as np
from scipy import sparse
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from skprox.linear_model import RegularisedRegression

# Generate data
X, y = make_regression(n_samples=100, n_features=1000, random_state=0, noise=4.0, bias=100.0)
X = sparse.csr_matrix(X)

# Solve the problem using scikit-learn
model = Lasso(alpha=0.1)
model.fit(X, y)
print("scikit-learn solution: {}".format(model.coef_))

# Solve the problem using scikit-prox
model = RegularisedRegression(proximal='L1', sigma=0.1)
model.fit(X, y)
print("scikit-prox solution: {}".format(model.coef_))
```

### Example 2: Total Variation Regression
The following code solves the following optimization problem:
minimize 1/2‖Ax − b‖2 + λ‖∇x‖1

```python
import numpy as np
from scipy import sparse
from sklearn.datasets import make_regression
from skprox.linear_model import RegularisedRegression

# Generate data
X, y = make_regression(n_samples=100, n_features=1000, random_state=0, noise=4.0, bias=100.0)
X = sparse.csr_matrix(X)

# Solve the problem using scikit-prox
model = RegularisedRegression(proximal='TV', sigma=0.1)
model.fit(X, y)
print("scikit-prox solution: {}".format(model.coef_))
```

## Documentation
The documentation is available at https://scikit-prox.readthedocs.io/en/latest/

## License
This project is licensed under the MIT License - see the LICENSE.md file for details

## Acknowledgments
This project leans on the pyproximal package borrowing all the proximal operators except for Total Variation which
is implemented using functions from skimage.