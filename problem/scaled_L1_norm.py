from base import NormL1

class ScaledL1Norm(object):

    def __init__(self, function_space, beta):

        self._beta = beta

        self._normL1 = NormL1(function_space)


    def __call__(self, control):

        return self._beta*self._normL1(control)

    @property
    def beta(self):
        return self._beta
