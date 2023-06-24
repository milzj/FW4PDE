from dolfin import *
from dolfin_adjoint import *

class FarmDomain(SubDomain):
    """Farm domain
    source: https://zenodo.org/record/224251

    Modications:
    - Added init
    """
    def __init__(self, x_max=1.0, y_max=1.0, **kwargs):

        super(FarmDomain, self).__init__(**kwargs)
        self._x_max = x_max
        self._y_max = y_max

    def inside(self, x, on_boundary):
        return between(x[0], (0.375*self._x_max, 0.625*self._x_max)) and between(x[1], (0.35*self._y_max, 0.65*self._y_max))
