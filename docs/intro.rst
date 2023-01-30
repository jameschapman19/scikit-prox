Introduction
=======================================

The goal of this project is to implement a set of algorithms for solving the following optimization problem:
minimize f(x) + g(x) where f is a smooth function and g is a proximal operator. The proximal operator of a function g is defined as:
proxg(λx) = argmin y g(y) + 1/2λ‖y − x‖2