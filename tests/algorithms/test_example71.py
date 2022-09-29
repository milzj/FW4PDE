import pytest

import numpy as np
from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import QuasiArmijoGoldstein
from stepsize import DecreasingStepSize
from stepsize import DunnHarshbargerStepSize
from stepsize import DunnScalingStepSize
from stepsize import DemyanovRubinovAdaptiveStepSize
from stepsize import DemyanovRubinovOptimalStepSize

def example71():
    """Implements desired state of Example 7.1 in Ref. [1].
    as fenics expression and function.

    Replaced +4pi^2sin(2pi x) by -4pi^2sin(2pi x) in yd computation (see p. 877 in Ref. [1]).

    References:
    -----------
    [1] G. Wachsmuth and D. Wachsmuth: Convergence and regularization results for
    optimal control problems with sparsity functional, ESAIM: COCV 17 (2011) 858-886,
    https://doi.org/10.1051/cocv/2010027
    """

    lb = Constant(-30.0)
    ub = Constant(30.0)

    beta = 0.5
    n = 500

    yd_str1 = "-5.0*x-4.0*pi*pi*sin(2.0*pi*x)"

    yd_str = "(x <= 1.0/12) ? 0.0 + yy : " +\
        "(x > 1.0/12 && x <= 5.0/12) ? 5.0/48 - 5.0/2*x + 15.0*x*x + yy :" +\
        "(x > 5.0/12 && x <= 7.0/12) ? -5.0/2 + 10.0*x + yy :" +\
        "(x > 7.0/12 && x <= 11.0/12) ? -365.0/48 + 55.0/2*x - 15.0*x*x + yy : 5.0 + yy"


    yd_str = yd_str.replace("yy", yd_str1)
    yd_str = yd_str.replace("x", "x[0]")
    yd_expr = Expression(yd_str, degree=1)
    yd = yd_expr

    mesh = UnitIntervalMesh(n)

    U = FunctionSpace(mesh, "DG", 0)
    V = FunctionSpace(mesh, "CG", 1)

    u = Function(U)
    y = Function(V)
    v = TestFunction(V)

    F = (inner(grad(y), grad(v)) - u * v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")
    solve(F == 0, y, bc)

    J = assemble(0.5*inner(y-yd,y-yd)*dx)

    control = Control(u)
    rf = ReducedFunctional(J, control)

    return rf, u, U, lb, ub, beta

@pytest.mark.parametrize("linesearch", [QuasiArmijoGoldstein(), DecreasingStepSize(), \
                                        DunnHarshbargerStepSize(), DunnScalingStepSize(), \
                                        DemyanovRubinovAdaptiveStepSize(), \
                                        DemyanovRubinovOptimalStepSize()])
def test_example71(linesearch):
    "Test is designed to test step sizes."

    maxiter = 1000
    gtol = 1e-6
    ftol = 1e-6

    options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

    rf, u, U, lb, ub, beta = example71()

    problem = MoolaOptimizationProblem(rf)
    u_moola = moola.DolfinPrimalVector(u)

    scaled_L1_norm = ScaledL1Norm(U,beta)
    box_constraints = BoxConstraints(U, lb, ub)
    moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    solver = FrankWolfe(problem, initial_point=u_moola,\
                nonsmooth_functional=scaled_L1_norm, linesearch=linesearch,\
                lmo=moola_box_lmo, options=options)

    sol = solver.solve()

    obj_final = sol["objective_final"]
    obj_best = sol["objective_best"]

    # https://www.wolframalpha.com/input?i=+1%2F2+2+30*4%2F12+%2B+integrate+1%2F2+%284+pi%5E2+sin%282+pi+x%29%29%5E2+for+x%3D0..1+
    obj_true = 10 + 4*np.pi**4

    assert (obj_final-obj_true)/obj_true < ftol
    assert (obj_best-obj_true)/obj_true < ftol
