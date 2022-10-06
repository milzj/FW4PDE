"""Problem taken from Casas and Mateos (2021).
 
References:
----------
Eduardo Casas and Mariano Mateos. State error estimates for the nu-
merical approximation of sparse distributed control problems in the ab-
sence of Tikhonov regularization. Vietnam J. Math., 49(3):713–738, 2021.
doi:10.1007/s10013-021-00491-x.
"""
import numpy as np

from fenics import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import DemyanovRubinovOptimalStepSize

def compute_yd(n, y_init=None):

    delta = 9.
    q = 1.0

    alpha = -1.0
    beta = 1.0

    x0 = 2**(-delta)/3.0

    m = UnitIntervalMesh(n)
    mesh = UnitIntervalMesh(2*n)
    mesh.coordinates()[:] = np.concatenate((m.coordinates()-1, m.coordinates()[1:]))


    V = FunctionSpace(mesh, "CG", 1)

    y = Function(V)
    v = TestFunction(V)

    u = Expression('x[0] <= x0 ? alpha : beta ', degree=0, x0=x0, alpha=alpha, beta=beta)

    F = (inner(grad(y), grad(v)) + y*abs(y)**3*v - u*v) * dx
    bc = DirichletBC(V, 0.0, "on_boundary")

    # initial value
    if y_init != None:
        y.interpolate(y_init)

    solve(F == 0, y, bc, solver_parameters={'newton_solver': {'linear_solver': 'petsc',
                "absolute_tolerance": 1e-13, "relative_tolerance": 1e-13}})

    z = Expression("(1-x[0]*x[0])*(x0-x[0])", degree = 3, x0=x0)
    w = Expression("6.0*x[0] - 2.0*x0", degree = 1, x0=x0)

    yd = Expression("w-4.0*pow(abs(y),3)*z+y", w = w, y = y, z = z, degree = 3)
    return yd, mesh, u, x0


def errornormL1(u, uh, mesh=mesh):
    # TODO: Take difference first
    F = abs(u-uh)*dx(mesh, {'quadrature_degre': 5})
    f = assemble(F)

    return f


set_working_tape(Tape())

# Homotopy for yd
n = 3*2**6
yd, yd_mesh, true_solution, x0 = compute_yd(n)
n = 3*2**13
yd, yd_mesh, true_solution, x0 = compute_yd(n, yd)
assert (yd_mesh.coordinates() == x0).any()


beta = 0.0
lb = Constant(-1.0)
ub = Constant(1.0)

n = 2**13
maxiter = 1000
gtol = 1e-15
ftol = -np.inf

m = UnitIntervalMesh(n)
mesh = UnitIntervalMesh(2*n)
mesh.coordinates()[:] = np.concatenate((m.coordinates()-1, m.coordinates()[1:]))

U = FunctionSpace(mesh, "DG", 0)
V = FunctionSpace(mesh, "CG", 1)

scaled_L1_norm = ScaledL1Norm(U,beta)

u = Function(U)
y = Function(V)
v = TestFunction(V)

F = (inner(grad(y), grad(v)) + y*abs(y)**3*v - u*v) * dx
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

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,
                    stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

solution_final = sol["control_final"].data
plot(solution_final)
plt.savefig("solution.png")
plt.close()

