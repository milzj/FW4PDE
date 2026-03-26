from dolfin import assemble, inner, dx

from fw4pde.base import NormL1

class ScaledL1Norm(object):

    def __init__(self, function_space, beta, alpha=0.0):

        self._beta = beta
        self._alpha = alpha

        self._normL1 = NormL1(function_space)


    def __call__(self, control):

        l1_term = self._beta*self._normL1(control)
        if self._alpha == 0.0:
            return l1_term

        l2_term = 0.5*self._alpha*assemble(inner(control, control)*dx)
        return l2_term + l1_term

    @property
    def beta(self):
        return self._beta

    @property
    def alpha(self):
        return self._alpha
