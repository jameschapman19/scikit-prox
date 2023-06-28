---
title: 'Scikit-Prox: a scikit-learn style framework for regularized linear models using proximal gradient descent'
tags:
  - Python
  - Machine Learning 
authors:
  - name: James Chapman 
    orcid: 0000-0002-9364-8118 
    affiliation: 1
affiliations:
  - name: Centre for Medical Image Computing, University College London, London, UK 
    index: 1
date: 1 February 2023 
bibliography: paper.bib
---

# Summary

Regularized linear models are a class of machine learning models that are widely used in practice. They are particularly useful for high dimensional data where the number of features exceeds the number of samples. Regularization is a technique that can be used to improve the generalization performance of a model by penalizing the model complexity. Regularization can be applied to a wide range of models including linear regression, logistic regression, and support vector machines.

Proximal gradient descent is a first-order optimization algorithm that can be used to solve a wide range of optimization problems. It is particularly useful for solving regularized linear models. The algorithm is based on the proximal operator, which is a generalization of the gradient operator. The proximal operator is a useful tool for solving optimization problems with non-smooth objective functions.

`scikit-prox` is a Python package that implements a range of regularized linear models using proximal gradient descent. This allows the user to use any regularisation functional from `pyproximal` or implement their own, expanding on the options available through `scikit-learn` [@pedregosa2011scikit] while also being fully compatible with the `scikit-learn` API. This allows users to easily switch between `scikit-learn` and `scikit-prox` models. Furthermore, this means that `scikit-prox` models can use the full range of utilities for model selection and hyperparameter tuning from `scikit-learn`.

# Statement of need

Existing implementations of regularized linear models in Python such as `scikit-learn` are either limited to a small number of regularisation functions or incompatible with the `scikit-learn` API.

This package is intended to fill the gap between the high-quality implementations of a large range of proximal operators from `pyproximal` and the robust and modelling pipeline in `scikit-learn` with limited regularisation functionals, allowing users to build linear models with regularisation functions that are best suited to their data.

The intended audience for `scikit-prox` is researchers and practitioners who are interested in using regularized linear models in their work, and in particular those working in fields with structured data such as medical imaging.

# Implementation

## Regularized linear models

| Model Class | Description                             |
|-------------|-----------------------------------------|
| RegularisedLinearRegression | Linear regression with regularisation   |
| RegularisedLogisticRegression | Logistic regression with regularisation |

## Regularisation functionals

| Functional | Description                    |
|------------|--------------------------------|
| L0 | L0 norm                        |
| L0Ball | L0 ball constraint             |
| L1 | L1 norm                        |
| L1Ball | L1 ball constraint             |
| L2 | L2 norm                        |
| L21 | L21 norm                       |
| L21_plus_L1 | L21 plus L1                    |
| Nuclear | Nuclear norm                   |
| NuclearBall | Nuclear norm ball constraint   |
| Log | Logarithmic penalty            |
| Log1 | Logarithmic penalty 2          |
| Euclidean | Euclidean norm                 |
| EuclideanBall | Euclidean norm ball constraint |

## Example

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from skprox.linear_model import LogisticRegression

# Generate data
X, y = make_classification(n_samples=1000, n_features=100, n_informative=10, n_redundant=10, n_classes=2,
                           random_state=0)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create model
model = LogisticRegression(proximal='L1', sigma=0.1)

# Fit model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

## Documentation

Further documentation for `scikit-prox` is available at https://scikit-prox.readthedocs.io/en/latest/.

# Conclusion



# Acknowledgements

JC is supported by the EPSRC-funded UCL Centre for Doctoral Training in Intelligent, Integrated Imaging in Healthcare (
i4health) (EP/S021930/1) and the Department of Health’s NIHR-funded Biomedical Research Centre at University College
London Hospitals.

# References