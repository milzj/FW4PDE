"""Implements a bilinear (nonconvex) control problem.

References:
----------

Michelle Vallejos. MGOPT with gradient projection method for solving bilinear
elliptic optimal control problems. Computing, 87(1-2):21–33, 2010.
"""
import numpy as np

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import QuasiArmijoGoldstein



lb = Constant(-4.0)
ub = Constant(4.0)

beta = 0.0001

f = Expression("sin(2.0*pi*x[0])*sin(2*pi*x[1])", degree = 1)
yd = Expression("1.0+f", f=f, degree = 1)

n = 128
maxiter = 1000
gtol = 1e-8
ftol = -np.inf
mesh = UnitSquareMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

scaled_L1_norm = ScaledL1Norm(U,beta)

u = Function(U)
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) - y * u * v - f*v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)

J = assemble(0.5*inner(y-yd,y-yd)*dx)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

box_constraints = BoxConstraints(U, lb, ub)
moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

stepsize = QuasiArmijoGoldstein()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
            stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.savefig("solution_final.pdf")

solution_best = sol["control_best"].data
plot(solution_best)
plt.savefig("solution_best.pdf")

error = errornorm(solution_final, solution_best, degree_rise = 0)
print("Difference of best and final iterate={}".format(error))
