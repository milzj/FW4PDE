from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt
import numpy as np

set_log_level(30)


yd_expr = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2*x[0])/6.0", degree = 1)
yd = yd_expr


options_bfgs={'jtol': 0,
	'gtol': 1e-6,
        'Hinit': "default",
        'maxiter': 100,
         'mem_lim': 10}

options_newtoncg={'gtol': 1e-6,
                  'maxiter': 20,
                  'display': 3,
                  'ncg_hesstol': 0}


n = 256
mesh = UnitSquareMesh(n,n)
alpha = Constant(1e-3)

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

u = Function(U)
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) - u * v) * dx

bc = DirichletBC(V, Constant(0.0), "on_boundary")

solve(F == 0, y, bc)

J = assemble(0.5*inner(y-yd,y-yd)*dx+alpha/2*u**2*dx)

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

