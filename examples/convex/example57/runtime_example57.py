"""Implements Example 5.7 from
K. Kunisch and D. Walter. On fast convergence rates for generalized conditional gradient
methods with backtracking stepsize. arXiv:2109.15217v1
"""

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

import numpy as np

set_log_level(30)

lb = Constant(-30.0)
ub = Constant(30.0)

beta = 0.000
yd_expr = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2*x[0])/6.0", degree = 0)


n = 256
maxiter = 5
gtol = 1e-8
ftol = 1e-8
mesh = UnitSquareMesh(n,n)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

yd = Function(U)
yd.interpolate(yd_expr)



u = Function(U)
# y = Function(V)
y = TrialFunction(V)
v = TestFunction(V)

#F = (inner(grad(y), grad(v)) - u * v) * dx
a = inner(grad(y), grad(v)) * dx
L = u*v*dx


bc = DirichletBC(V, Constant(0.0), "on_boundary")

A, b = assemble_system(a, L, bc)

solver = PETScKrylovSolver()
solver.set_operator(A)

Y = Function(V)
solver.solve(Y.vector(), b)

#problem = LinearVariationalProblem(a, L, Y, bc)
#solver = LinearVariationalSolver(problem)

parameters = {"linear_solver": "cg",
                                         "krylov_solver": {
                                             "absolute_tolerance": 1e-8
                                         },
                                         "preconditioner": "hypre_amg"
                                         }

parameters = {"linear_solver": "gmres",
                                         "krylov_solver": {
                                             "absolute_tolerance": 1e-3
                                         },
                                         "preconditioner": "ilu"
                                         }





#solver.parameters.update(parameters)
#solver.solve()

#solve(F == 0, y, bc, solver_parameters={"nonlinear_solver": "newton_solver"})
#solve(a == L, y, bc)

J = assemble(0.5*inner(Y-yd,Y-yd)*dx)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)



for i in range(100):
	print(i)
	with stop_annotating():
		print(rf(u))
	np.random.seed(i)
	u.vector()[:] = np.random.randn(U.dim())
