import numpy as np

class BoxConstraints(object):

    def __init__(self, function_space, lb, ub):

        import fenics

        _lb = fenics.project(lb, function_space)
        self._lb = _lb
        self._lb_vec = _lb.vector().get_local()

        _ub = fenics.project(ub, function_space)
        self._ub = _ub
        self._ub_vec = _ub.vector().get_local()


    @property
    def lb(self):
        return self._lb_vec

    @property
    def ub(self):
        return self._ub_vec
