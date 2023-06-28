from math import sqrt

import numpy as np
from nilearn.decoding.objective_functions import _tv_l1_from_gradient
from nilearn.decoding.proximal_operators import _projector_on_tvl1_dual, _dual_gap_prox_tvl1
from pyproximal import ProxOperator


class TVL1(ProxOperator):
    def __init__(
            self,
            sigma,
            shape=None,
            l1_ratio=0.05,
            dgap_tol=5.0e-5,
            x_tol=1e-2,
            max_iter=200,
            check_gap_frequency=10,
    ):
        super().__init__()

        self.l1_ratio = l1_ratio
        self.sigma = sigma
        self.dgap_tol = dgap_tol
        self.x_tol = x_tol
        self.max_iter = max_iter
        self.check_gap_frequency = check_gap_frequency
        self.shape = shape

    def unmask(self, x):
        return x.reshape(self.shape)

    def mask(self, x):
        return x.reshape(-1)

    def prox(self, x, tau, **kwargs):
        x = self.unmask(x)
        # loop over the last dimension
        x = _prox_tvl1(
            x,
            self.l1_ratio,
            self.sigma * tau,
            self.x_tol,
            self.max_iter,
            self.check_gap_frequency,
        )[0]
        return self.mask(x)

    def __call__(self, x):
        x = self.unmask(x)
        # loop over the last dimension
        total = self.sigma * _tv_l1_from_gradient(
            np.ma.getdata(_gradient_id(x, l1_ratio=self.l1_ratio))
        )
        return total


def _prox_tvl1(
    input_img,
    l1_ratio=0.05,
    weight=50,
    dgap_tol=5.0e-5,
    x_tol=None,
    max_iter=200,
    check_gap_frequency=4,
    fista=True,
    init=None,
):
    """
    Compute the TV-L1 proximal (ie total-variation +l1 denoising) on 3d images.

    Find the argmin `res` of
        1/2 * ||im - res||^2 + weight * TVl1(res),

    Parameters
    ----------
    input_img : ndarray of floats (2-d or 3-d)
        Input data to be denoised. `im` can be of any numeric type,
        but it is cast into an ndarray of floats for the computation
        of the denoised image.

    weight : float, optional
        Denoising weight. The greater ``weight``, the more denoising (at
        the expense of fidelity to ``input``)

    dgap_tol : float, optional
        Precision required. The distance to the exact solution is computed
        by the dual gap of the optimization problem and rescaled by the
        squared l2 norm of the image (for contrast invariance).

    x_tol : float or None, optional
        The maximal relative difference between input and output. If
        specified, this specifies a stopping criterion on x, rather than
        the dual gap.

    max_iter : int, optional
        Maximal number of iterations used for the optimization.

    val_min : None or float, optional
        An optional lower bound constraint on the reconstructed image.

    val_max : None or float, optional
        An optional upper bound constraint on the reconstructed image.

    verbose : bool, optional
        If True, print the dual gap of the optimization

    fista : bool, optional
        If True, uses a FISTA loop to perform the optimization.
        if False, uses an ISTA loop.

    callback : callable
        Callable that takes the local variables at each
        steps. Useful for tracking.

    init : array of shape as im
        Starting point for the optimization.

    check_gap_frequency : int, optional (default 4)
        Frequency at which duality gap is checked for convergence.

    Returns
    -------
    out : ndarray
        TV-l1-denoised image.

    Notes
    -----
    The principle of total variation denoising is explained in
    http://en.wikipedia.org/wiki/Total_variation_denoising

    The principle of total variation denoising is to minimize the
    total variation of the image, which can be roughly described as
    the integral of the norm of the image gradient. Total variation
    denoising tends to produce "cartoon-like" images, that is,
    piecewise-constant images.

    This function implements the FISTA (Fast Iterative Shrinkage
    Thresholding Algorithm) algorithm of Beck et Teboulle, adapted to
    total variation denoising in "Fast gradient-based algorithms for
    constrained total variation image denoising and deblurring problems"
    (2009).

    For details on implementing the bound constraints, read the aforementioned
    Beck and Teboulle paper.
    """
    dtype= input_img.dtype
    weight = float(weight)
    input_img_flat = input_img.reshape(-1)
    input_img_norm = np.dot(input_img_flat, input_img_flat)
    if not input_img.dtype.kind == "f":
        input_img = input_img.astype(dtype)
    shape = [len(input_img.shape) + 1] + list(input_img.shape)
    grad_im = np.zeros(shape).astype(dtype)
    grad_aux = np.zeros(shape).astype(dtype)
    t = 1.0
    i = 0
    lipschitz_constant = 1.1 * (
        4 * input_img.ndim * (1 - l1_ratio) ** 2 + l1_ratio**2
    )

    # negated_output is the negated primal variable in the optimization
    # loop
    negated_output = -input_img if init is None else -init

    dgap = np.inf

    # A boolean to control if we are going to do a fista step
    fista_step = fista

    while i < max_iter:
        grad_tmp = _gradient_id(negated_output, l1_ratio=l1_ratio)
        grad_tmp *= 1.0 / (lipschitz_constant * weight)
        grad_aux += grad_tmp
        grad_tmp = _projector_on_tvl1_dual(grad_aux, l1_ratio)

        # Careful, in the next few lines, grad_tmp and grad_aux are a
        # view on the same array, as _projector_on_tvl1_dual returns a view
        # on the input array
        t_new = 0.5 * (1.0 + sqrt(1.0 + 4.0 * t * t))
        t_factor = (t - 1.0) / t_new
        if fista_step:
            grad_aux = (1 + t_factor) * grad_tmp - t_factor * grad_im
        else:
            grad_aux = grad_tmp
        grad_im = grad_tmp
        t = t_new
        gap = weight * _div_id(grad_aux, l1_ratio=l1_ratio)

        # Compute the primal variable
        negated_output = gap - input_img
        if (i % check_gap_frequency) == 0:
            if x_tol is None:
                old_dgap = dgap
                dgap = _dual_gap_prox_tvl1(
                    input_img_norm,
                    -negated_output,
                    gap,
                    weight,
                    l1_ratio=l1_ratio,
                )
                if dgap < dgap_tol:
                    break
                if old_dgap < dgap:
                    # M-FISTA strategy: switch to an ISTA to have
                    # monotone convergence
                    fista_step = False
                elif fista:
                    fista_step = True
        i += 1

    # Compute the primal variable, however, here we must use the ista
    # value, not the fista one
    output = input_img - weight * _div_id(grad_im, l1_ratio=l1_ratio)
    return output, dict(converged=(i < max_iter))

def _gradient_id(img, l1_ratio=0.5):
    """Compute gradient + id of an image.

    Parameters
    ----------
    img : ndarray, shape (nx, ny, nz, ...)
        N-dimensional image

    l1_ratio : float in the interval [0, 1]; optional (default .5)
        Constant that mixes L1 and spatial prior terms in the penalization.

    Returns
    -------
    gradient : ndarray, shape (4, nx, ny, nz, ...).
        Spatial gradient of the image: the i-th component along the first
        axis is the gradient along the i-th axis of the original array img.

    Raises
    ------
    RuntimeError

    """
    if not (0.0 <= l1_ratio <= 1.0):
        raise RuntimeError(
            f"l1_ratio must be in the interval [0, 1]; got {l1_ratio}"
        )

    shape = [img.ndim + 1] + list(img.shape)
    gradient = np.zeros(shape, img.dtype)

    # @numba.jit
    # def diff(img, d):
    #     gradient[tuple([0, slice(None, -1)])] = np.diff(img, axis=d)
    #     return gradient
    #
    # for d in range(img.ndim):
    #     gradient = diff(gradient, d)

    # the gradient part: 'Clever' code to have a view of the gradient
    # with dimension i stop at -1
    slice_all = [0, slice(None, -1)]
    for d in range(img.ndim):
        gradient[tuple(slice_all)] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))

    gradient[:-1] *= 1.0 - l1_ratio

    # the identity part
    gradient[-1] = l1_ratio * img

    return gradient

def _div_id(grad, l1_ratio=0.5):
    """Compute divergence + id of image gradient + id.

    Parameters
    ----------
    grad : ndarray, shape (4, nx, ny, nz, ...)
        where `img_shape` is the shape of the brain bounding box, and
        n_axes = len(img_shape).

    l1_ratio : float in the interval [0, 1]; optional (default .5)
        Constant that mixes L1 and spatial prior terms in the penalization.

    Returns
    -------
    res : ndarray, shape (nx, ny, nz, ...)
        The computed divergence + id operator.

    Raises
    ------
    RuntimeError

    """
    if not (0.0 <= l1_ratio <= 1.0):
        raise RuntimeError(
            f"l1_ratio must be in the interval [0, 1]; got {l1_ratio}"
        )

    res = np.zeros(grad.shape[1:]).astype(grad.dtype)

    # the divergence part
    for d in range(grad.shape[0] - 1):
        this_grad = np.rollaxis(grad[d], d)
        this_res = np.rollaxis(res, d)
        this_res[:-1] += this_grad[:-1]
        this_res[1:-1] -= this_grad[:-2]
        if len(this_grad) > 1:
            this_res[-1] -= this_grad[-2]

    res *= 1.0 - l1_ratio

    # the identity part
    res -= l1_ratio * grad[-1]

    return res