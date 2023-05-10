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

from sw_solver import *

import moola
import matplotlib.pyplot as plt

set_log_level(30)

import fw4pde
from solver_options import SolverOptions

state = steady_sw(control)
u, p = split(state)

# sparsity parameter and box constraints
WtoMW = 1e-6
beta = WtoMW*4800.0
lb = Constant(0.0)
ub = Constant(0.059)

# Objective function
scaled_L1_norm = fw4pde.problem.ScaledL1Norm(control_space,beta)
power_functional = assemble(rho*control*inner(u,u)**1.5*site_dx(1))

ctrl = Control(control)
rf = ReducedFunctional(-WtoMW*power_functional, ctrl)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(control)

box_constraints = fw4pde.problem.BoxConstraints(control_space, lb, ub)
moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)


solver_options = SolverOptions()
options = solver_options.options
stepsize = solver_options.stepsize

solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

# Postprocessing: Plotting and saving

solution_final = sol["control_final"].data
plt.set_cmap("coolwarm")
c = plot(solution_final)
plt.colorbar(c)
plt.savefig("output/solution_final_n_{}.pdf".format(domain_parameters.n))
plt.savefig("output/solution_final_n_{}.png".format(domain_parameters.n))
plt.close()

solution_best = sol["control_best"].data
c = plot(solution_best)
plt.colorbar(c)
plt.savefig("output/solution_best_n_{}.pdf".format(domain_parameters.n))
plt.savefig("output/solution_best_n_{}.png".format(domain_parameters.n))
plt.close()

now = datetime.now().strftime("%d-%B-%Y-%H-%M-%S")

filename = outdir + now + "_solution_best_n={}.txt".format(domain_parameters.n)
np.savetxt(filename, solution_best.vector()[:])
filename = outdir + now + "_solution_final_n={}.txt".format(domain_parameters.n)
np.savetxt(filename, solution_final.vector()[:])


file = File(outdir + "/" + "solution" +  "_best_n={}".format(domain_parameters.n) + ".pvd")
file << solution_best
file = File(outdir + "/" + "solution" +  "_final_n={}".format(domain_parameters.n) + ".pvd")
file << solution_final
