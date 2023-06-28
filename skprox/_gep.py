from abc import abstractmethod

import numpy as np


class _GEPMixin:

    @abstractmethod
    def _get_AB(self, X, y):
        pass

    def _compute_gradient(self, A, B, coef_):
        Aw, Bw, wAw, wBw = self._get_terms(A, B, coef_)
        grads = 2 * Aw - (Aw @ np.triu(wBw) + Bw @ np.triu(wAw))
        return -grads

    def _get_terms(self, A, B, coef_):
        Aw = A @ coef_
        Bw = B @ coef_
        wAw = coef_.T @ Aw
        wBw = coef_.T @ Bw
        wAw[np.diag_indices_from(wAw)] = np.where(np.diag(wAw) > 0, np.diag(wAw), 0)
        wBw[np.diag_indices_from(wBw)] = np.where(np.diag(wAw) > 0, np.diag(wBw), 0)
        return Aw, Bw, wAw, wBw

    def _objective(self, A, B, coef_):
        Aw, Bw, wAw, wBw = self._get_terms(A, B, coef_)
        return 2 * np.trace(wAw) - np.trace(wAw @ wBw)
