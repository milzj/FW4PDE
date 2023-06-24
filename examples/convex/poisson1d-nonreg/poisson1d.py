"""One-dimensional control problem

The solution u^* to the problem is a
bang-bang function. We have u^*(x) = 1
if x < pi/4 and u^*(x) = -1 if x > pi/4.

The desired state yd is given by S(u^*).
Therefore, yd is reachable.

The desired state is piecewise quadratic.
The second derivative is discontinuous at x=pi/4.

We use a standard equidistant grid. Hence,
pi/4 is not a grid point.

We implement yd using SubDomain.

References:
-----------
MiroK (2014): https://fenicsproject.org/qa/5230/defining-a-function-on-a-subdomain/
DGarciaLahuerta (2021): https://fenicsproject.discourse.group/t/subdomains-in-1d-mesh/7037

"""

import numpy as np
from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

import fw4pde

lb = Constant(-1.0)
ub = Constant(1.0)

beta = .0
n = 2**8

maxiter = 100

gtol = 1e-8
ftol = 1e-8
mesh = UnitIntervalMesh(n)

tol = 1e-13
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] <= np.pi/4.0 + tol

class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return x[0] + tol > np.pi/4.0

domains = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
domains.set_all(0)

subdomain0 = Omega_0()
subdomain1 = Omega_1()
subdomain0.mark(domains, 1)
subdomain1.mark(domains, 2)


dx = Measure("dx",domain=mesh, subdomain_data = domains)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

# https://www.wolframalpha.com/input?i=second+derivative+-1%2F16+x%288-8pi%2Bpi*pi%2B8x%29
yd1 = "-1.0/16.0*x[0]*(8.0-8.0*pi+pi*pi+8.0*x[0])"
# https://www.wolframalpha.com/input?i=second+derivative++-1%2F16+%28x-1%29%28pi*pi-8x%29
yd2 = "-1.0/16.0*(x[0]-1.0)*(pi*pi-8.0*x[0])"

yd = Expression("x[0] <= pi/4 ? {}: {}".format(yd1, yd2), degree = 1, pi=np.pi)

yd1 = Expression("{}".format(yd1), degree = 2, pi=np.pi)
yd2 = Expression("{}".format(yd2), degree = 2, pi=np.pi)

# Check if subdomain works
assert np.isclose(assemble(yd1*dx), assemble(yd1*dx(1)) + assemble(yd1*dx(2)), rtol=tol)

scaled_L1_norm = fw4pde.problem.ScaledL1Norm(U,beta)

u = Function(U)

w = TrialFunction(V)
v = TestFunction(V)

bc = DirichletBC(V, 0.0, "on_boundary")
a = inner(grad(w), grad(v)) * dx
L = u * v * dx

A, b = assemble_system(a, L, bc)
solver = LUSolver(A, "petsc")

y = Function(V)
solver.solve(y.vector(), b)

J = assemble(0.5*inner(y-yd1,y-yd1)*dx(1) + 0.5*inner(y-yd2,y-yd2)*dx(2))

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

box_constraints = fw4pde.problem.BoxConstraints(U, lb, ub)
moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

stepsize = fw4pde.stepsize.DemyanovRubinovOptimalStepSize()

options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.savefig("final_iterate.png")
plt.close()


plt.figure()
u.interpolate(Expression("x[0] <= pi/4 ? 1.0: -1.0", degree = 0, pi=np.pi))
solution_best = sol["control_best"].data
plot(u)
plt.savefig("solution.png")
plt.close()

solution_final = sol["control_final"]
obj = problem.obj
obj(solution_final)
gradient = obj.derivative(solution_final).primal()
gradient_vec = gradient.data.vector()[:]
np.savetxt("gradient_vec.out", gradient_vec)

