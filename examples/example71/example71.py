"""Implements Example 7.1 from
G. Wachsmuth and D. Wachsmuth. Convergence and regularization results for opti-
mal control problems with sparsity functional. ESAIM Control. Optim. Calc. Var.,
17(3):858–886, 2011. doi:10.1051/cocv/2010027.
"""

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import QuasiArmijoGoldstein, DecreasingStepSize
from stepsize import DunnHarshbargerStepSize



lb = Constant(-30.0)
ub = Constant(30.0)

beta = 0.5
yd_expr = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2*x[0])/6.0", degree = 0)


n = 32
maxiter = 1000
gtol = 1e-8
ftol = 1e-8
mesh = UnitIntervalMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

yd = Function(V)
yd.interpolate(yd_expr)

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

linesearch = QuasiArmijoGoldstein(alpha=0.5, gamma=0.5)
#linesearch = DecreasingStepSize()
#linesearch = DunnHarshbargerStepSize()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm, linesearch=linesearch, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.savefig("example57_final.pdf")

solution_best = sol["control_best"].data
plot(solution_best)
plt.savefig("example57_best.pdf")

error = errornorm(solution_final, solution_best, degree_rise = 0)
print("Difference of best and final iterate={}".format(error))
