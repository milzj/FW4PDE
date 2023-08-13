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

def compute_yd(n, y_init=None):

    delta = 9.
    q = 1.0

    lb = Constant(-1.0)
    ub = Expression("1+0.1*sin(2*pi*x[0])", degree = 0)

    x0 = 2**(-delta)/3.0

    m = UnitIntervalMesh(n)
    mesh = UnitIntervalMesh(2*n)
    mesh.coordinates()[:] = np.concatenate((m.coordinates()-1, m.coordinates()[1:]))

    V = FunctionSpace(mesh, "CG", 1)

    y = Function(V)
    v = TestFunction(V)

    u = Expression('x[0] <= x0 ? lb : ub ', degree=0, x0=x0, lb=lb, ub=ub)

    F = (inner(grad(y), grad(v)) + y*abs(y)**3*v - u*v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    # initial value
    if y_init != None:
        y.interpolate(y_init)

    solve(F == 0, y, bc, solver_parameters={'newton_solver': {'linear_solver': 'petsc',
                "absolute_tolerance": 1e-13, "relative_tolerance": 1e-13}})

    z = Expression("(1-x[0]*x[0])*(x0-x[0])", degree = 3, x0=x0)
    w = Expression("6.0*x[0] - 2.0*x0", degree = 1, x0=x0)

    yd = Expression("w-4.0*pow(abs(y),3)*z+y", w = w, y = y, z = z, degree = 3)
    return yd, mesh, u, x0, z


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

    # homotopy
    yd, yd_mesh, true_solution, x0, z = compute_yd(3*2**6)
    yd, yd_mesh, true_solution, x0, z = compute_yd(n_ref, yd)
    # x0 should be a grid point

    beta = 0.0
    lb = Constant(-1.0)
    ub = Expression("1+0.1*sin(2*pi*x[0])", degree = 0)

    m = UnitIntervalMesh(n)
    mesh = UnitIntervalMesh(2*n)
    mesh.coordinates()[:] = np.concatenate((m.coordinates()-1, m.coordinates()[1:]))

    U = FunctionSpace(mesh, "DG", 0)
    V = FunctionSpace(mesh, "CG", 1)

    scaled_L1_norm = ScaledL1Norm(U,beta)

    u = Function(U)
    if u_init != None:
        u = project(u_init, U)
        u_vec = u.vector()[:]
        lb_vec = project(lb, U).vector()[:]
        ub_vec = project(ub, U).vector()[:]
        u_vec = np.clip(u_vec, lb_vec, ub_vec)
        u.vector()[:] = u_vec

    y = Function(V)
    v = TestFunction(V)

    F = (inner(grad(y), grad(v)) + y*abs(y)**3*v - u*v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    solve(F == 0, y, bc)

    J = assemble(0.5*inner(y-yd,y-yd)*dx)

    control = Control(u)
    rf = ReducedFunctional(J, control)

    problem = MoolaOptimizationProblem(rf)
    u_moola = moola.DolfinPrimalVector(u)

    box_constraints = BoxConstraints(U, lb, ub)
    moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    stepsize = DemyanovRubinovOptimalStepSize()
    stepsize = QuasiArmijoGoldstein()

    options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

    solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm, \
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

    sol = solver.solve()

    # compute criticality measure
    solution = sol["control_best"]
    problem.obj(solution)
    gradient = problem.obj.derivative(solution).primal()

    gradient_vec = gradient.data.vector()[:]
    solution_vec = solution.data.vector()[:]

    z_vec = project(z, U).vector()[:]

    x_vec = solution_vec - gradient_vec
    w_vec = np.clip(x_vec, -beta, beta)
    lb_vec = project(lb, U).vector()[:]
    ub_vec = project(ub, U).vector()[:]
    w_vec = np.clip(x_vec-w_vec, lb_vec, ub_vec)
    w = Function(U)
    w.vector()[:] = w_vec

    canonical_criticality_measure = errornorm(solution.data, w, degree_rise = 0, mesh=yd_mesh)
    normal_criticality_measure = -1.

    # normal map
    if discrete_gradient != None:
        dg = Function(U)
        dg.interpolate(discrete_gradient)

        v_vec = solution_vec - dg.vector()[:]
        w_vec = np.clip(v_vec, -beta, beta)
        prox_v_vec = np.clip(v_vec-w_vec, lb_vec, ub_vec)

        prox_v = Function(U)
        prox_v.vector()[:] = prox_v_vec
        prox_v_moola = moola.DolfinPrimalVector(prox_v)

        problem.obj(prox_v_moola)
        gradient = problem.obj.derivative(prox_v_moola).primal()

        w_vec = v_vec - prox_v_vec
        w = Function(U)
        # take minus
        w.vector()[:] = -w_vec

        normal_criticality_measure = errornorm(gradient.data, w, degree_rise = 0, mesh=yd_mesh)


    solution_error = errornormL1(true_solution, sol["control_best"].data, yd_mesh)

    return sol["control_best"].data, sol["dual_gap"], canonical_criticality_measure,\
               normal_criticality_measure, solution_error, gradient.data



def test_convergence_rate():
    """Code verification for a one-dimensional initial value problem.

    dual_gap(u_h) should converge with rate h

    distance of u_h to true solution should converge with rate h

    canonical criticality measure should converge with rate h

    normal map-based criticality measure should converge with rate h
    """

    n_ref = 3*2**13
    ns = [2**n for n in range(9,14)]

    solutions = []
    errors = []
    pg_errors = []
    gradients = []

    gtol = 1e-10
    ftol = -np.inf

    for n in ns:
        print(n)
        print("\n")
        solution, dual_gap, canonical_cm, normal_cm, solution_error, gradient \
            = solve_problem(n, n_ref, u_init=None, maxiter=1000, gtol=gtol, ftol=ftol)

        solutions.append(solution)
        errors.append(solution_error)
        gradients.append(gradient)

    dual_gaps = []
    canonical_criticality_measures = []
    normal_criticality_measures = []

    # perform one iteration to get access to dual_gap and criticality measures
    for i in range(np.size(ns)):
        solution, dual_gap, canonical_cm, normal_cm, solution_error, gradient \
            = solve_problem(n_ref, n_ref, u_init=solutions[i], maxiter=0, gtol=gtol, ftol=ftol, \
                discrete_gradient=gradients[i])

        dual_gaps.append(dual_gap)
        canonical_criticality_measures.append(canonical_cm)
        normal_criticality_measures.append(normal_cm)

    # Convergence dual gap
    rates = convergence_rates(dual_gaps, [1.0/n for n in ns])

    assert np.isclose(np.median(rates), 1.0, atol=0.2)

    X = np.ones((np.size(ns), 2)); X[:, 1] = np.log([1.0/n for n in ns])
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(dual_gaps), rcond=None)
    rate = x[1]
    constant = np.exp(x[0])

    assert np.isclose(rate, 1.0, atol=0.2)

    ndrop = 0
    x_vec = ns
    y_vec = dual_gaps

    fig, ax = plt.subplots()
    ax.plot([n for n in ns], dual_gaps)

    y_vec = constant*x_vec**(-rate)
    ax.plot(x_vec, y_vec, color="black", linestyle="--", label=r"{}\cdot 10^{}".format(constant,-rate))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("convergence_rates_semilinear.png")

    # Convergence of solutions
    rates = convergence_rates(errors, [1.0/n for n in ns])

    assert np.isclose(np.median(rates), 1.0, atol=0.2)

    X = np.ones((np.size(ns), 2)); X[:, 1] = np.log([1.0/n for n in ns])
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(errors), rcond=None)

    rate = x[1]
    constant = np.exp(x[0])

    assert np.isclose(rate, 1.0, atol=0.2)

    # Convergence of normal map based criticality measure
    rates = convergence_rates(normal_criticality_measures, [1.0/n for n in ns])

    assert np.isclose(np.median(rates), 1.0, atol=0.2)

    X = np.ones((np.size(ns), 2)); X[:, 1] = np.log([1.0/n for n in ns])
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(normal_criticality_measures), rcond=None)
    rate = x[1]
    constant = np.exp(x[0])

    assert np.isclose(rate, 1.0, atol=0.2)

    # Convergence of canonical criticality measure
    rates = convergence_rates(canonical_criticality_measures, [1.0/n for n in ns])

    assert np.isclose(np.median(rates), 1.0, atol=0.2)

    X = np.ones((np.size(ns), 2)); X[:, 1] = np.log([1.0/n for n in ns])
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(canonical_criticality_measures), rcond=None)
    rate = x[1]
    constant = np.exp(x[0])

    assert np.isclose(rate, 1.0, atol=0.2)




if __name__ == "__main__":

    test_convergence_rate()
