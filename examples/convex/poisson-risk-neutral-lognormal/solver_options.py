class SolverOptions(object):

    def __init__(self):

            tol = 1e-8
            self._options = {"ftol": tol, "gtol": tol, "display": 3}

    @property
    def options(self):
        return self._options
