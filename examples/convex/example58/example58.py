"""Implements Example 5.8 from
K. Kunisch and D. Walter. On fast convergence rates for generalized conditional gradient
methods with backtracking stepsize. arXiv:2109.15217v1
"""

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt
import numpy as np

set_log_level(30)

import fw4pde

lb = Constant(-10.0)
ub = Expression('x[0] <= 0.25 ? 0 : -5.0+20.0*x[0]', degree=1)

beta = 0.002
yd = Expression("sin(4*pi*x[0])*cos(8*pi*x[1])*exp(2.0*x[0])", degree = 1)
g = Expression("10.0*cos(8*pi*x[0])*cos(8*pi*x[1])", degree = 1)

n = 256
maxiter = 1000
gtol = 1e-10
ftol = -np.inf
mesh = UnitSquareMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

scaled_L1_norm = fw4pde.problem.ScaledL1Norm(U,beta)

u = Function(U)
y = TrialFunction(V)
v = TestFunction(V)

bc = DirichletBC(V, 0.0, "on_boundary")
a = inner(grad(y), grad(v)) * dx
L = (u * v + g*v) * dx

A, b = assemble_system(a, L, bc)
solver = LUSolver(A, "petsc")

Y = Function(V)
solver.solve(Y.vector(), b)

J = assemble(0.5*inner(Y-yd,Y-yd)*dx)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

box_constraints = fw4pde.problem.BoxConstraints(U, lb, ub)
moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

stepsize = fw4pde.stepsize.QuasiArmijoGoldstein(alpha=0.5, gamma=0.75)

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm, 
        stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
c = plot(solution_final)
plt.colorbar(c)
plt.savefig("solution_final.pdf")
plt.close()

solution_best = sol["control_best"].data
c = plot(solution_best)
plt.colorbar(c)
plt.savefig("solution_best.pdf")

error = errornorm(solution_final, solution_best, degree_rise = 0)
print("Difference of best and final iterate={}".format(error))


solution_final = sol["control_final"]
obj = problem.obj
obj(solution_final)
gradient = obj.derivative(solution_final).primal()
gradient_vec = gradient.data.vector()[:]
np.savetxt("gradient_vec.out", gradient_vec)
np.savetxt("solution_vec.out", solution_final.data.vector()[:])

