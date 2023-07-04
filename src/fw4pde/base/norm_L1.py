import numpy as np

class NormL1(object):
    """Implements an evaluation of the L1 norm.

    See p. 137 in [1].

    References:
    ----------
    [1] E. Casas, K. Kunisch, F. Tröltzsch, Optimal control of PDEs and FE-approximation,
    Handbook of Numerical Analysis, 2022, Pages 115-163, 
    https://doi.org/10.1016/bs.hna.2021.12.004
    """

    def __init__(self, function_space):

        import fenics

        if function_space.ufl_element().family() != 'Discontinuous Lagrange':
            raise TypeError("function_space.ufl_element().family()={} should be DG.".format(function_space.ufl_element().family()))

        if function_space.ufl_element().degree() != 0:
            raise TypeError("function_space.ufl_element().degree()={} should be zero.".format(function_space.ufl_element().degree()))


        v = fenics.TestFunction(function_space)

        _w = fenics.assemble(fenics.Constant(1.0)*v*fenics.dx)
        self._w = _w
        self._control_coefficients = np.zeros(function_space.dim())


    def __call__(self, control):

        self._control_coefficients[:] = control.vector()[:]

        return np.dot(np.abs(self._control_coefficients), self._w)
