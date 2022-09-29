import numpy as np
from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import QuasiArmijoGoldstein, DecreasingStepSize
from stepsize import DunnHarshbargerStepSize
from prox import prox_box_l1

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

scaled_L1_norm = ScaledL1Norm(U,beta)

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

box_constraints = BoxConstraints(U, lb, ub)
moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

stepsize = QuasiArmijoGoldstein()
#stepsize = DecreasingStepSize()
#stepsize = DunnHarshbargerStepSize()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm, stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.savefig("solution_final.pdf")

plt.close()
solution_best = sol["control_best"].data
plot(solution_best)
plt.savefig("solution_best.pdf")
plt.close()

error = errornorm(solution_final, solution_best, degree_rise = 0)
print("Difference of best and final iterate={}".format(error))

solution_final = sol["control_final"]
obj = problem.obj
obj(solution_final)
gradient = obj.derivative(solution_final).primal()

w = solution_final.data.vector()[:]-gradient.data.vector()[:]
prox_w = prox_box_l1(w, -1.0, 1.0, beta)
w = Function(U)
w.vector()[:] = prox_w

error = errornorm(w, solution_final.data, degree_rise = 0)
print("criticality measure={}".format(error))
solution = solution_final.data

set_working_tape(Tape())
nref = 10*2**12
meshref = UnitIntervalMesh(nref)

Uref = FunctionSpace(meshref, "DG", 0)
Vref = FunctionSpace(meshref, "CG", 1)

u = Function(Uref)
u = project(solution, Uref)
y = Function(Vref)
v = TestFunction(Vref)


bc = DirichletBC(Vref, 0.0, "on_boundary")
F = (inner(grad(y), grad(v)) - u * v) * dx

solve(F == 0, y, bc)

J = assemble(0.5*inner(y-yd,y-yd)*dx)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

obj = problem.obj
obj(u_moola)
gradient = obj.derivative(u_moola).primal()


w = u.vector()[:]-gradient.data.vector()[:]

prox_w = prox_box_l1(w, -1.0, 1.0, beta)
w = Function(Uref)
w.vector()[:] = prox_w

error = errornorm(w, u, degree_rise = 0)
print("criticality measure={}".format(error))
