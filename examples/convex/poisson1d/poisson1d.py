import numpy as np
from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

import fw4pde

lb = Constant(-1.0)
ub = Constant(1.0)

beta = .005
n = 2**10

maxiter = 1000
gtol = 1e-10
ftol = 1e-10
mesh = UnitIntervalMesh(n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

yd = Expression("exp(2*x[0])*sin(2.0*pi*x[0])/3.0", degree = 1)

scaled_L1_norm = fw4pde.problem.ScaledL1Norm(U,beta)

u = Function(U)
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) - u * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)

J = assemble(0.5*inner(y-yd,y-yd)*dx)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

box_constraints = fw4pde.problem.BoxConstraints(U, lb, ub)
moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

stepsize = fw4pde.stepsize.QuasiArmijoGoldstein()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.savefig("solution_final.png")
plt.close()

solution_best = sol["control_best"].data
plot(solution_best)
plt.savefig("solution_best.png")
plt.close()

