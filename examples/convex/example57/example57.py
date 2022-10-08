"""Implements Example 5.7 from
K. Kunisch and D. Walter. On fast convergence rates for generalized conditional gradient
methods with backtracking stepsize. arXiv:2109.15217v1
"""

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt
import numpy as np

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import QuasiArmijoGoldstein

lb = Constant(-30.0)
ub = Constant(30.0)

beta = 0.001

yd = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2*x[0])/6.0", degree = 1)

n = 32

maxiter = 1000
gtol = 1e-8
ftol = -np.inf
mesh = UnitSquareMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)


u = Function(U)

y = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(y), grad(v)) * dx
L = u*v*dx

bc = DirichletBC(V, Constant(0.0), "on_boundary")

A, b = assemble_system(a, L, bc)
solver = LUSolver(A, "petsc")

Y = Function(V)
solver.solve(Y.vector(), b)

J = assemble(0.5*inner(Y-yd,Y-yd)*dx)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

with stop_annotating():
    scaled_L1_norm = ScaledL1Norm(U,beta)
    box_constraints = BoxConstraints(U, lb, ub)
    moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    stepsize = QuasiArmijoGoldstein(gamma=0.75)

    options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol, "display": 2}

    solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,
            stepsize=stepsize, lmo=moola_box_lmo, options=options)

    sol = solver.solve()

    solution_final = sol["control_final"].data
    plot(solution_final)
    plt.savefig("solution.pdf")

    solution_final = sol["control_final"]
    obj = problem.obj
    obj(solution_final)
    gradient = obj.derivative(solution_final).primal()
    gradient_vec = gradient.data.vector()[:]
    np.savetxt("gradient_vec.out", gradient_vec)
    np.savetxt("solution_vec.out", solution_final.data.vector()[:])

