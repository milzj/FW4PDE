# The problem is inspired by that considered in
# https://opentidalfarm.readthedocs.io/en/latest/examples/headland-optimization/headland-optimization.html

import numpy as np

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import QuasiArmijoGoldstein, DecreasingStepSize
from stepsize import DunnHarshbargerStepSize, DunnScalingStepSize



lb = Constant(0.0)
ub = Constant(1.0)

beta = 7.5

kappa = Expression("x[1]*x[1]+0.005", degree = 2)
f = Expression("cos(2.0*pi*x[0]*x[1])*sin(2*pi*x[1]*x[0])", degree = 1)
yd = Expression('(0.25 < x[0] && x[0] < 0.75 && 0.25 < x[1] && x[1] < 0.75) ? -2.0 : 2.0', degree=0)
g = Expression("1.0+x[1]*x[1] + x[0]*x[0]", degree=0)

n = 64
maxiter = 1000
gtol = 1e-13
ftol = -np.inf
mesh = UnitSquareMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

scaled_L1_norm = ScaledL1Norm(U,beta)

u = Function(U)
y = Function(V)
v = TestFunction(V)

class FarmDomain(SubDomain):
    def inside(self, x, on_boundary):
        return (0.25 <= x[0] <= 0.75 and
                0.25  <= x[1] <= 0.75)

farm_domain = FarmDomain()
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(0)

farm_domain.mark(domains, 1)
dx = Measure("dx",domain=mesh, subdomain_data = domains)
plot(domains)
plt.savefig("domains.png")
plt.savefig("domains.pdf")
plt.close()

W = VectorFunctionSpace(mesh, "P", 1)
w = Function(W)

F = kappa*inner(grad(y), grad(v))*dx + g*abs(y)*y*u*v*dx(1) + f*v*dx
bc = DirichletBC(V, 0.0, "on_boundary")

solve(F == 0, y, bc)

J = -assemble(u*inner(y-yd,y-yd)**1.5*dx(1))

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

box_constraints = BoxConstraints(U, lb, ub)
moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

stepsize = QuasiArmijoGoldstein(gamma=0.8)
#stepsize = DunnScalingStepSize()
#stepsize = DecreasingStepSize()
#stepsize = DunnHarshbargerStepSize()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
c = plot(solution_final)
plt.colorbar(c)
plt.savefig("solution_final.pdf")
plt.savefig("solution_final.png")
plt.close()

solution_best = sol["control_best"].data
c = plot(solution_best)
plt.colorbar(c)
plt.savefig("solution_best.pdf")
plt.savefig("solution_best.png")
plt.close()

error = errornorm(solution_final, solution_best, degree_rise = 0)
print("Difference of best and final iterate={}".format(error))
