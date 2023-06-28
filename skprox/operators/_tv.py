import numpy as np
from pyproximal.ProxOperator import _check_tau
from skimage.restoration import denoise_tv_chambolle, denoise_tv_bregman


class TV:
    def __init__(self, sigma, dims, isotropic=True, max_iter=1000):
        self.sigma = sigma
        self.isotropic = isotropic
        self.max_iter = max_iter
        self.count = 0
        self.dims = dims

    def unmask(self, x):
        return x.reshape(self.dims)

    def mask(self, x):
        return x.reshape(-1)

    def _increment_count(func):
        """Increment counter"""

        def wrapped(self, *args, **kwargs):
            self.count += 1
            return func(self, *args, **kwargs)

        return wrapped

    @_increment_count
    @_check_tau
    def prox(self, x, tau):
        x = self.unmask(x)
        if self.isotropic:
            x= denoise_tv_chambolle(
                x, weight=tau * self.sigma, max_num_iter=self.max_iter
            )
        else:
            x= denoise_tv_bregman(
                x,
                weight=tau * self.sigma,
                isotropic=self.isotropic,
                max_num_iter=self.max_iter,
            )
        return self.mask(x)

    def __call__(self, x):
        m = np.zeros_like(x)
        for d in range(x.ndim):
            diff = np.gradient(x, axis=d)
            if self.isotropic:
                m += diff ** 2
            else:
                m += np.abs(diff)
        if not self.isotropic:
            m = np.sqrt(m)
        return m.sum()
