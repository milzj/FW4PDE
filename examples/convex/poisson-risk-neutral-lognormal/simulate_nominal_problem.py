import numpy as np

from dolfin import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

import fw4pde

import os

from random_poisson_problem import RandomPoissonProblem
from solver_options import SolverOptions
from random_field_options import RandomFieldOptions

outdir = "simulation_output/"
if not os.path.exists(outdir):
    os.makedirs(outdir)


n =  150

# Get random field options
rf_options = RandomFieldOptions().options
num_addends = rf_options["num_addends"]
num_random_vars = rf_options["num_rvs"]
len_scale = rf_options["len_scale"]
sample = np.zeros(num_random_vars)

poisson_problem = RandomPoissonProblem(n, num_addends, len_scale)
solver_options = SolverOptions().options

lb = poisson_problem.lb
ub = poisson_problem.ub
beta = poisson_problem.beta

u = poisson_problem.control
U = poisson_problem.control_space

control = Control(u)

J = poisson_problem(u, sample)

rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

with stop_annotating():
    scaled_L1_norm = fw4pde.problem.ScaledL1Norm(U,beta)
    box_constraints = fw4pde.problem.BoxConstraints(U, lb, ub)
    moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    stepsize = fw4pde.stepsize.QuasiArmijoGoldstein(gamma=0.75)


    solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,
            stepsize=stepsize, lmo=moola_box_lmo, options=solver_options)

    sol = solver.solve()

    solution_final = sol["control_final"].data
    c = plot(solution_final)
    plt.colorbar(c)
    plt.savefig(outdir + "nominal_solution_n={}.png".format(n))
    plt.savefig(outdir + "nominal_solution_n={}.pdf".format(n))
    plt.close()

    solution_final = sol["control_final"]
    obj = problem.obj
    obj(solution_final)
    gradient = obj.derivative(solution_final).primal()
    gradient_vec = gradient.data.vector()[:]
    np.savetxt(outdir + "nominal_gradient_vec_n={}.out".format(n), gradient_vec)
    np.savetxt(outdir + "nominal_solution_vec_n={}.out".format(n), solution_final.data.vector()[:])
