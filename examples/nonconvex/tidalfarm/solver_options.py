import numpy as np
import fw4pde

class SolverOptions(object):

    def __init__(self):

        maxiter = 10
        gtol = 1e-4
        ftol = -np.inf

        self._options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}
        self._stepsize = fw4pde.stepsize.DecreasingStepSize()

    @property
    def options(self):
        "Termination options for conditional gradient method."
        return self._options

    @property
    def stepsize(self):
        "Step size rule for conditional gradient method."
        return self._stepsize
