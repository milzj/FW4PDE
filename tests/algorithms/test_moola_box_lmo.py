import pytest

import numpy as np
from algorithms import MoolaBoxLMO

from dolfin import *
from dolfin_adjoint import *
import moola

@pytest.mark.parametrize("n", [64, 128, 32])
@pytest.mark.parametrize("beta", [0.1, 1e-3, 1e-2, 1.0])
def test_moola_box_lmo(n, beta):
    """
    The solution to min_x (g, x) + beta norm(x,L1) s.t. lb <= x <= ub
    satisfies the fixed point equation

    x = projection(u-projection(u, -beta, beta), lb, ub),

    where u = x - g.
    """


    atol = 1e-6
    rtol = 1e-15

    mesh = UnitSquareMesh(n, n)

    V = FunctionSpace(mesh, "CG", 1)
    W = FunctionSpace(mesh, "DG", 0)

    lb = -np.ones(W.dim())
    ub = np.ones(W.dim())

    moola_box_lmo = MoolaBoxLMO(lb, ub, beta)


    f = interpolate(Expression("sin(2*pi*x[0])+exp(x[1])", name='Control', degree=2), W)
    u = Function(V, name='State')
    v = TestFunction(V)

    F = (inner(grad(u), grad(v)) - f * v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, u, bc)

    control = Control(f)

    w = Expression("sin(pi*x[0])*sin(pi*x[1])", degree=3)
    d = 1 / (2 * pi ** 2)
    d = Expression("d*w", d=d, w=w, degree=3)

    J = assemble((0.5 * inner(u - d, u - d)) * dx)

    rf = ReducedFunctional(J, control)
    problem = MoolaOptimizationProblem(rf)

    f_moola = moola.DolfinPrimalVector(f)
    problem.obj(f_moola)
    gradient = problem.obj.derivative(f_moola).primal()
    v_moola = f_moola.copy()

    moola_box_lmo.solve(gradient, v_moola)

    def projection(v, a, b):
        return np.clip(v, a, b)

    # LMO solution
    solution = v_moola.data
    # Evaluate fixed point equation (output is _solution)
    u = solution.vector().get_local() - gradient.data.vector().get_local()

    u_projection = projection(u, -beta, beta)
    __solution = projection(u-u_projection, lb, ub)
    _solution = Function(W)
    _solution.vector().set_local(__solution)

    assert errornorm(solution, _solution, degree_rise = 0) < atol
    assert errornorm(solution,_solution, degree_rise = 0)/(norm(solution)+1.0) < rtol

