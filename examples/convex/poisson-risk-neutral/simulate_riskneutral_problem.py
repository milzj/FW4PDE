import numpy as np

from dolfin import *
from dolfin_adjoint import *
import moola
import matplotlib.pyplot as plt

set_log_level(30)

import fw4pde

import os

from random_poisson_problem import RandomPoissonProblem
from poisson_sampler import PoissonSampler
from solver_options import SolverOptions


outdir = "simulation_output/"
if not os.path.exists(outdir):
    os.makedirs(outdir)


n = 150
N = 150
std = 1.0

poisson_problem = RandomPoissonProblem(n)
solver_options = SolverOptions().options
sampler = PoissonSampler(std=std)

lb = poisson_problem.lb
ub = poisson_problem.ub
beta = poisson_problem.beta

u = poisson_problem.control
U = poisson_problem.control_space

control = Control(u)

# https://diego.assencio.com/?index=c34d06f4f4de2375658ed41f70177d59
J = 0
for i in range(N):
    sample = sampler.sample()
    j = poisson_problem(u, sample)
    J += 1.0/(i+1.0)*(j-J)

rf = ReducedFunctional(J, control)

problem = MoolaOptimizationProblem(rf)
u_moola = moola.DolfinPrimalVector(u)

with stop_annotating():
    scaled_L1_norm = fw4pde.problem.ScaledL1Norm(U,beta)
    box_constraints = fw4pde.problem.BoxConstraints(U, lb, ub)
    moola_box_lmo = fw4pde.algorithms.MoolaBoxLMO(box_constraints.lb, box_constraints.ub, beta)

    stepsize = fw4pde.stepsize.DemyanovRubinovOptimalStepSize()

    solver = fw4pde.algorithms.FrankWolfe(problem, initial_point=u_moola, nonsmooth_functional=scaled_L1_norm,
            stepsize=stepsize, lmo=moola_box_lmo, options=solver_options)

    sol = solver.solve()

    solution_final = sol["control_final"].data
    c = plot(solution_final)
    plt.colorbar(c)
    plt.savefig(outdir + "riskneutral_solution_n={}_N={}.png".format(n,N))
    plt.savefig(outdir + "riskneutral_solution_n={}_N={}.pdf".format(n,N))
    plt.close()

    solution_final = sol["control_final"]
    obj = problem.obj
    obj(solution_final)
    gradient = obj.derivative(solution_final).primal()
    gradient_vec = gradient.data.vector()[:]
    np.savetxt(outdir + "riskneutral_gradient_vec_n={}_N={}.out".format(n,N), gradient_vec)
    np.savetxt(outdir + "riskneutral_solution_vec_n={}_N={}.out".format(n,N), solution_final.data.vector()[:])
