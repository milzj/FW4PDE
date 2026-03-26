import numpy as np

class NumpyBoxLMO(object):
    """
    See Lem. 5.6 in Ref. [1]

    References:
    ----------
    [1] 	K. Kunisch and D. Walter, On fast convergence rates for generalized conditional
        gradient methods with backtracking stepsize, preprint,
        https://arxiv.org/abs/2109.15217, 2021
    """

    def __init__(self, lb, ub, beta, alpha=0.0):

        n = lb.size
        self._gradient = np.zeros(n)
        self._solution = np.zeros(n)

        self._lb = lb
        self._ub = ub
        self._beta = beta
        self._alpha = alpha

    @property
    def solution(self):
        return self._solution

    def solve(self, gradient):
        self._solve(gradient)

    def _solve(self, gradient):
        """
        Note:
        ----

        We implement the cases grad(x) > beta and grad(x) < -beta
        to allow for meaningful results when beta is zero.
        """

        beta = self._beta
        alpha = self._alpha
        lb = self._lb
        ub = self._ub

        self._gradient[:] = gradient[:]

        if alpha > 0.0:
            # Exact solution of
            # min_x 0.5*alpha*||x||_2^2 + <gradient, x> + beta*||x||_1
            # s.t. lb <= x <= ub
            z = -self._gradient
            self._solution[:] = np.sign(z) * np.maximum(np.abs(z) - beta, 0.0) / alpha
            self._solution[:] = np.clip(self._solution, lb, ub)
            return

        self._solution[:] = np.clip(0.0, lb, ub)

        idx = self._gradient > beta
        self._solution[idx] = lb[idx]

        idx = self._gradient < -beta
        self._solution[idx] = ub[idx]



class MoolaBoxLMO(NumpyBoxLMO):

    def __init__(self, lb, ub, beta, alpha=0.0):

        super().__init__(lb, ub, beta, alpha)


    def solve(self, gradient, v):

        self._solve(gradient.data.vector().get_local())

        v.data.vector().set_local(self._solution)
        v.bump_version()
