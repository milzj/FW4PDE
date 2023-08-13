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
from stepsize import DemyanovRubinovOptimalStepSize
from stepsize import DemyanovRubinovAdaptiveStepSize

# Source https://github.com/dolfin-adjoint/pyadjoint/blob/master/pyadjoint/verification.py
def convergence_rates(E_values, eps_values, show=True):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i] / E_values[i - 1])
            / log(eps_values[i] / eps_values[i - 1]))
    if show:
        print("Computed convergence rates: {}".format(r))
    return r


def errornormL1(u, uh, mesh=mesh):
    # TODO: Take difference first
    F = abs(u-uh)*dx(mesh, {'quadrature_degre': 5})
    f = assemble(F)

    return f

def solve_problem(n, n_ref,  u_init=None, maxiter=1000, gtol=1e-15, ftol=-np.inf, discrete_gradient=None):
    """Problem taken from Casas and Mateos (2021).

    References:
    ----------
    Eduardo Casas and Mariano Mateos. State error estimates for the nu-
    merical approximation of sparse distributed control problems in the ab-
    sence of Tikhonov regularization. Vietnam J. Math., 49(3):713–738, 2021.
    doi:10.1007/s10013-021-00491-x.
    """

    set_working_tape(Tape())

    # x0 should be a grid point

    beta = 0.0
    lb = Expression("-1", degree = 0)
    ub = Expression("1+0.1*sin(2*pi*x[0])", degree = 0)

    mesh = UnitIntervalMesh(n)
    U = FunctionSpace(mesh, "DG", 0)

    scaled_L1_norm = ScaledL1Norm(U,beta)

    yd = Expression("x[0]*x[0]", degree=0)

    u = Function(U)

    if u_init != None:
        u = project(u_init, U)

    V = FunctionSpace(mesh, "CG", 1)
    y = Function(V)
    w = TrialFunction(V)
    v = TestFunction(V)

    bc = DirichletBC(V, 0.0, "on_boundary")

    a = inner(grad(w), grad(v))*dx
    L = u*v*dx
    A, b = assemble_system(a, L, bc)

    solver = LUSolver(A, "petsc")
    solver.solve(y.vector(), b)

    J = assemble(0.5*inner(y-yd,y-yd)*dx)

    control = Control(u)
    rf = ReducedFunctional(J, control)

    problem = MoolaOptimizationProblem(rf)
    u_moola = moola.DolfinPrimalVector(u)

    box_constraints = BoxConstraints(U, lb, ub)
    moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    stepsize = QuasiArmijoGoldstein(gamma=0.5)
    stepsize = DemyanovRubinovOptimalStepSize()
    options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

    solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm, \
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

    sol = solver.solve()

    return sol["control_best"].data, sol["dual_gap"]

def test_convergence_rate():
    """Code verification for a one-dimensional initial value problem.

    dual_gap(u_h) should converge with rate h^2

    distance of u_h to true solution should converge with rate h

    canonical criticality measure should converge with rate h^3/2

    normal map-based criticality measure should converge with rate h
    """

    n_ref = 2**16
    ns = [2**n for n in range(6,13)]

    solutions = []

    gtol = 1e-14
    ftol = -np.inf

    for n in ns: 
        solution, dual_gap = solve_problem(n, n_ref, u_init=None, maxiter=100, gtol=gtol, ftol=ftol)
        solutions.append(solution)

    dual_gaps = []

    # perform one iteration to get access to dual_gap and criticality measures
    for i in range(np.size(ns)):
        solution, dual_gap = solve_problem(n_ref, n_ref, u_init=solutions[i], maxiter=0, gtol=gtol, ftol=ftol)

        dual_gaps.append(dual_gap)

    # Convergence dual gap
    rates = convergence_rates(dual_gaps, [1.0/n for n in ns])

    ndrop = 0
    x_vec = ns
    y_vec = dual_gaps
    X = np.ones((len(x_vec[ndrop::]), 2)); X[:, 1] = np.log(x_vec[ndrop::]) # design matrix
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(y_vec[ndrop::]), rcond=None)

    rate = x[1]
    constant = np.exp(x[0])

    fig, ax = plt.subplots()
    ax.plot([n for n in ns], dual_gaps)

    y_vec = constant*x_vec**rate
    ax.plot(x_vec, y_vec, color="black", linestyle="--", label=r"{}\cdot 10^{}".format(constant,rate))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("convergence_rates_bilinear.png")


    assert np.isclose(np.median(rates), 1.0, atol=0.0)

if __name__ == "__main__":

    test_convergence_rate()
