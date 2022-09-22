from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt
import numpy as np

set_log_level(30)


yd = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2*x[0])/6.0", degree = 1)



options_bfgs={'jtol': 0,
	'gtol': 1e-6,
        'Hinit': "default",
        'maxiter': 100,
         'mem_lim': 10}

options_newtoncg={'gtol': 1e-6,
                  'maxiter': 20,
                  'display': 3,
                  'ncg_hesstol': 0}


n = 32
mesh = UnitSquareMesh(n,n)
alpha = Constant(1e-3)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

u = Function(U)
y = TrialFunction(V)
v = TestFunction(V)

a = inner(grad(y), grad(v))*dx - y*u*v*dx
L = Constant(1.0)*v*dx

Y = Function(V)
bc = DirichletBC(V, Constant(0.0), "on_boundary")

problem = LinearVariationalProblem(a, L, Y, bc)
solver = LinearVariationalSolver(problem)
solver.solve()

#solve(F == 0, y, bc)

J = assemble(0.5*inner(Y-yd,Y-yd)*dx + 0.5*alpha*u**2*dx)

control = Control(u)
rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

#solver = moola.BFGS(problem, u_moola, options=options_bfgs)

solver = moola.NewtonCG(problem, u_moola, options=options_newtoncg)

sol = solver.solve()

solution = sol["control"].data
plot(solution)
plt.savefig("unconstrained_example57.pdf")

