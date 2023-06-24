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

import fw4pde


from desired_state import example71_desired_state

lb = Constant(-30.0)
ub = Constant(30.0)

beta = 0.5
n = 500

yd_str, _, _ = example71_desired_state(n=n)
yd_expr = Expression(yd_str, degree=3)

maxiter = 1000
gtol = 1e-6
ftol = 1e-6
mesh = UnitIntervalMesh(n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

yd = yd_expr

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

stepsize = fw4pde.stepsize.DunnHarshbargerStepSize()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
            stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.gca().set_aspect(1.0/30)
plt.savefig("example57_final.pdf")
plt.close()

solution_best = sol["control_best"].data
plot(solution_best)
plt.gca().set_aspect(1.0/30)
plt.savefig("example57_best.pdf")

error = errornorm(solution_final, solution_best, degree_rise = 0)
print("Difference of best and final iterate={}".format(error))
