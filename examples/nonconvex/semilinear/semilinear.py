import numpy as np

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import DemyanovRubinovOptimalStepSize


lb = Constant(-10.0)
ub = Constant(10.0)

beta = 0.1

yd = Expression("10.0*sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2*x[0])/6.0", degree = 2)
b = Expression("exp(2*x[0]*x[1])", degree=2)
kappa = Expression("x[1]*x[1]+0.05", degree = 2)

n = 256
maxiter = 1000
gtol = 1e-6
ftol = -np.inf
mesh = UnitSquareMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

scaled_L1_norm = ScaledL1Norm(U,beta)

u = Function(U)
y = Function(V)
v = TestFunction(V)


F = (kappa*inner(grad(y), grad(v)) + b*y**3 * v - u*v + Constant(1.0)*v) * dx
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

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
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
