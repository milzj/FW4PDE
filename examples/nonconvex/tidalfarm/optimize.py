"""The code solves the optimization problem considered in https://zenodo.org/record/224251
but with free-slip boundary conditions for the shores.

The steady shallow water solver is taken from

https://github.com/OpenTidalFarm/OpenTidalFarm/blob/master/opentidalfarm/solvers/coupled_sw_solver.py

"""

import os
outdir = "output/"
if not os.path.exists(outdir):
   os.makedirs(outdir)
from datetime import datetime

import numpy as np
from dolfin import *
from dolfin_adjoint import *

from domain import *
from rectangle_domain import RectangularDomain


import moola
import matplotlib.pyplot as plt

set_log_level(30)

from algorithms import FrankWolfe, MoolaBoxLMO
from problem import ScaledL1Norm, BoxConstraints
from stepsize import QuasiArmijoGoldstein, DecreasingStepSize
from stepsize import DunnHarshbargerStepSize, DunnScalingStepSize
from stepsize import DemyanovRubinovOptimalStepSize
from stepsize import DemyanovRubinovAdaptiveStepSize
from stepsize import DecreasingAdaptiveStepSize

x_max = 2000
y_max = 1000
N = 100

domain = RectangularDomain(0, 0, x_max, y_max, nx=N, ny=N)
mesh = domain.mesh

class FarmDomain(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (0.375*x_max, 0.625*x_max)) and between(x[1], (0.35*y_max, 0.65*y_max))

domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(0)
farm_domain = FarmDomain()
farm_domain.mark(domains, 1)
site_dx = Measure("dx", domain=mesh, subdomain_data = domains)

# parameters
depth = Constant(50.0)
viscosity = Constant(5.0)
C_t = Constant(0.6)
A_t = Constant(314.15)
friction = Constant(0.0025)
g = Constant(9.81)
f_u = Constant((0, 0))
rho = 1025.0

# function spaces
V_h = VectorElement("CG", mesh.ufl_cell(), 2)
Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V_h*Q_h)

control_space = FunctionSpace(domain.mesh, "DG", 0)
control = Function(control_space)

initial_condition = Constant(("1e-7", "0.0", "0.0"))


# boundary conditions

from boundary_conditions import BoundaryConditionSet

bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2.0,0.0)), facet_id=1, bctype="strong_dirichlet")
bcs.add_bc("eta", Constant(0.0), facet_id=2, bctype="strong_dirichlet")
bcs.add_bc("u", facet_id=3, bctype="free_slip")

def generate_strong_bcs():

    facet_ids = domain.facet_ids

    # Generate velocity boundary conditions
    bcs_u = []
    for _, expr, facet_id, _ in bcs.filter("u", "strong_dirichlet"):
        bc = DirichletBC(W.sub(0), expr, facet_ids, facet_id)
        bcs_u.append(bc)

    # Generate free-surface boundary conditions
    bcs_eta = []
    for _, expr, facet_id, _ in bcs.filter("eta", "strong_dirichlet"):
        bc = DirichletBC(W.sub(1), expr, facet_ids, facet_id)
        bcs_eta.append(bc)

    return bcs_u + bcs_eta

strong_bcs = generate_strong_bcs()
ds = domain.ds

def steady_sw(control):

    # Get domain measures
    # Define test functions
    v, q = TestFunctions(W)
    state = Function(W)

    ic = project(initial_condition, W)
#    state.assign(ic, annotate=False)
    state.vector().axpy(1.0, ic.vector())

    u_mid, h_mid = split(state)

    H = h_mid + depth

    # The normal direction
    n = FacetNormal(domain.mesh)

    # Divergence term.
    Ct_mid = -H * inner(u_mid, grad(q)) * dx

    # The surface integral contribution from the divergence term
    bc_contr = -H * dot(u_mid, n) * q * ds

    for function_name, u_expr, facet_id, bctype in bcs.filter(function_name='u'):
        if bctype in ('free_slip'):
            # Subtract the divergence integral again
            bc_contr -= -H * dot(u_mid, n) * q * ds(facet_id)

    # don't integrate pressure gradient by parts (a.o. allows dg test function v)
    C_mid = g * inner(v, grad(h_mid)) * dx

    # Bottom friction
    norm_u_mid = inner(u_mid, u_mid)**0.5
    u_mag = dot(u_mid, u_mid)**0.5
    R_mid = friction / H * norm_u_mid * inner(u_mid, v) * dx
    #R_mid += (control * 0.5 * A_t * C_t / H) * u_mag * inner(u_mid, v) * site_dx(1)
    # R_mid appears to be implemented in https://zenodo.org/record/224251
    R_mid += (control / H) * u_mag * inner(u_mid, v) * site_dx(1)

    # Advection term
    Ad_mid = inner(dot(grad(u_mid), u_mid), v) * dx

    # Viscosity
    D_mid = viscosity * inner(2*sym(grad(u_mid)), grad(v)) * dx

    # Create the final form
    G_mid = C_mid + Ct_mid + R_mid

    # Add the advection term
    G_mid += Ad_mid

    # Add the viscosity term
    G_mid += D_mid

    # Add the source term
    G_mid -= inner(f_u, v) * dx
    F = G_mid - bc_contr

    solve(F == 0, state, bcs = strong_bcs, J=derivative(F, state))

    return state



state = steady_sw(control)
u, p = split(state)

beta = 1e-6*4800.0
lb = Constant(0.0)
ub = Constant(0.05890486225480863)

scaled_L1_norm = ScaledL1Norm(control_space,beta)

# power_functional = assemble(rho*0.5*C_t*A_t*control*inner(u,u)**1.5*site_dx(1))
# power functional appears to be implemented in https://zenodo.org/record/224251
power_functional = assemble(rho*control*inner(u,u)**1.5*site_dx(1))

ctrl = Control(control)
rf = ReducedFunctional(-1e-6*power_functional, ctrl)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(control)

box_constraints = BoxConstraints(control_space, lb, ub)
moola_box_lmo = MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

#stepsize = QuasiArmijoGoldstein(gamma=0.5)
stepsize = DecreasingStepSize()
#stepsize = DunnScalingStepSize()
#stepsize = DemyanovRubinovOptimalStepSize()
#stepsize = DemyanovRubinovAdaptiveStepSize()
#stepsize = DecreasingAdaptiveStepSize()

gtol= 1e-4
ftol = -np.inf
maxiter = 100
options = {"maxiter": maxiter, "gtol": gtol, "ftol": ftol}

solver = FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()


solution_final = sol["control_final"].data
c = plot(solution_final)
plt.colorbar(c)
plt.savefig("output/solution_final_N_{}.pdf".format(N))
plt.savefig("output/solution_final_N_{}.png".format(N))
plt.close()

solution_best = sol["control_best"].data
c = plot(solution_best)
plt.colorbar(c)
plt.savefig("output/solution_best_N_{}.pdf".format(N))
plt.savefig("output/solution_best_N_{}.png".format(N))
plt.close()


now = datetime.now().strftime("%d-%B-%Y-%H-%M-%S")

filename = outdir + now + "_solution_best_N={}.txt".format(N)
np.savetxt(filename, solution_best.vector()[:])
filename = outdir + now + "_solution_final_N={}.txt".format(N)
np.savetxt(filename, solution_final.vector()[:])


file = File(outdir + "/" + "solution" +  "_best_N={}".format(N) + ".pvd")
file << solution_best
file = File(outdir + "/" + "solution" +  "_final_N={}".format(N) + ".pvd")
file << solution_final
