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

from domain_farm import DomainFarm
from tidal_parameters import TidalParameters
from tidal_problem import TidalProblem

import moola
import matplotlib.pyplot as plt

set_log_level(30)

import fw4pde
from solver_options import SolverOptions


# Tidal farm problem
tidal_parameters = TidalParameters()
domain_farm = DomainFarm()
tidal_problem = TidalProblem(tidal_parameters, domain_farm)
control_space = tidal_problem.control_space
control = Function(control_space)

# Objective function
beta = tidal_problem.beta
lb = tidal_problem.lb
ub = tidal_problem.ub
scaled_L1_norm = fw4pde.problem.ScaledL1Norm(control_space,beta)
ctrl = Control(control)
J = tidal_problem(control)
rf = ReducedFunctional(J, ctrl)

# Optimization problem
problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(control)

box_constraints = fw4pde.problem.BoxConstraints(control_space, lb, ub)
moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

# Solver
solver_options = SolverOptions()
options = solver_options.options
stepsize = solver_options.stepsize

solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,\
                stepsize=stepsize, lmo=moola_box_lmo, options=options)

sol = solver.solve()

# Postprocessing: Plotting and saving
solution_final = sol["control_final"].data
n = tidal_problem.domain_mesh.n
plt.set_cmap("coolwarm")
c = plot(solution_final)
plt.colorbar(c)
plt.savefig("output/solution_final_n_{}.pdf".format(n))
plt.savefig("output/solution_final_n_{}.png".format(n))
plt.close()

solution_best = sol["control_best"].data
c = plot(solution_best)
plt.colorbar(c)
plt.savefig("output/solution_best_n_{}.pdf".format(n))
plt.savefig("output/solution_best_n_{}.png".format(n))
plt.close()

now = datetime.now().strftime("%d-%B-%Y-%H-%M-%S")

filename = outdir + now + "_solution_best_n={}.txt".format(n)
np.savetxt(filename, solution_best.vector()[:])
filename = outdir + now + "_solution_final_n={}.txt".format(n)
np.savetxt(filename, solution_final.vector()[:])


file = File(outdir + "/" + "solution" +  "_best_n={}".format(n) + ".pvd")
file << solution_best
file = File(outdir + "/" + "solution" +  "_final_n={}".format(n) + ".pvd")
file << solution_final


# Compute and print results (source: https://zenodo.org/record/224251)
site_dx = tidal_problem.domain_mesh.site_dx
cost_coefficient = tidal_problem.cost_coefficient
thrust_coefficient = tidal_parameters.thrust_coefficient.values()[0]
turbine_cross_section = tidal_parameters.turbine_cross_section.values()[0]

site_area = assemble(Constant(1.0)*site_dx(1))
power = -problem.obj(sol["control_best"])
total_friction = assemble(solution_best*site_dx(1))
cost = cost_coefficient * total_friction
friction = 0.5*thrust_coefficient*turbine_cross_section # see model_turbine.py https://zenodo.org/record/224251

print("="*40)
print("Site area (m^2): {}".format(site_area))
print("Cost coefficient: {}".format(cost_coefficient))
print("Total power: %e" % power)
print("Total cost: %e" % cost)
print("Total turbine friction: %e" % total_friction)
print("Average smeared turbine friction: %e" % (total_friction / site_area))
print("Total power / total friction: %e" % (power / total_friction))
print("Friction per discrete turbine: {}".format(friction))
print("Estimated number of discrete turbines: {}".format(total_friction/friction))
print("Estimated average power per turbine: {}".format(power / (total_friction/friction)))
