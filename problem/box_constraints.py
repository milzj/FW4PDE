import numpy as np

class BoxConstraints(object):

    def __init__(self, function_space, lb, ub):

        import fenics

        _lb = fenics.Function(function_space)
        _lb.interpolate(lb)
        self._lb = _lb
        self._lb_vec = _lb.vector().get_local()

        _ub = fenics.Function(function_space)
        _ub.interpolate(ub)
        self._ub = _ub
        self._ub_vec = _ub.vector().get_local()


    @property
    def lb(self):
        return self._lb_vec

    @property
    def ub(self):
        return self._ub_vec
