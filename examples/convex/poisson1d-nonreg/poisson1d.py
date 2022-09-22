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

beta = .0
n = 2**12

maxiter = 100

gtol = 1e-12
ftol = 1e-12
mesh = UnitIntervalMesh(n)

class Omega_0(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] <= np.pi/4.0

class Omega_1(SubDomain):
	def inside(self, x, on_boundary):
		return x[0] > np.pi/4.0

domains = MeshFunction('size_t', mesh, mesh.topology().dim()-1)

subdomain0 = Omega_0()
subdomain1 = Omega_1()
subdomain0.mark(domains, 0)
subdomain1.mark(domains, 1)

dx = Measure("dx",domain=mesh, subdomain_data = domains)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

yd1 = "-1.0/16.0*x[0]*(8.0-8.0*pi+pi*pi+ 8.0*x[0])"
yd2 = "-1.0/16.0*x[0]*(pi*pi-8.0*x[0])"

yd = Expression("x[0] <= pi/4 ? {}: {}".format(yd1, yd2), degree = 2, pi=np.pi)

yd1 = Expression("{}".format(yd1), degree = 2, pi=np.pi)
yd2 = Expression("{}".format(yd2), degree = 2, pi=np.pi)

scaled_L1_norm = ScaledL1Norm(U,beta)

u = Function(U)
u.interpolate(Expression("x[0] <= pi/4 ? 1.0: -1.0", degree = 0, pi=np.pi))
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) - u * v) * dx
bc = DirichletBC(V, 0.0, "on_boundary")
solve(F == 0, y, bc)

J = assemble(0.5*inner(y-yd,y-yd)*dx)
#J = assemble(0.5*inner(y-yd1,y-yd1)*dx(2) + 0.5*inner(y-yd2,y-yd2)*dx(1))

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

box_constraints = BoxConstraints(U, lb, ub)
moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

linesearch = QuasiArmijoGoldstein()
#linesearch = DecreasingStepSize()
#linesearch = DunnHarshbargerStepSize()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm, linesearch=linesearch, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.savefig("solution_final.pdf")

plt.close()
solution_best = sol["control_best"].data
plot(solution_best)
plt.savefig("solution_best.pdf")
plt.close()

plt.figure()
u.interpolate(Expression("x[0] <= pi/4 ? 1.0: -1.0", degree = 0, pi=np.pi))
solution_best = sol["control_best"].data
plot(u)
plt.savefig("solution.pdf")
plt.close()



error = errornorm(solution_final, solution_best, degree_rise = 0)
print("Difference of best and final iterate={}".format(error))

solution_final = sol["control_final"]
obj = problem.obj
obj(solution_final)
gradient = obj.derivative(solution_final).primal()
gradient_vec = gradient.data.vector()[:]
np.savetxt("gradient_vec.out", gradient_vec)
np.savetxt("solution_vec.out", solution_final.data.vector()[:])

