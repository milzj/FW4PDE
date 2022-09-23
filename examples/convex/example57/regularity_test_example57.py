import fenics
import numpy as np

from stats import regularity_test

n = 64
beta = 0.001
mesh = fenics.UnitSquareMesh(n,n)
U = fenics.FunctionSpace(mesh, "DG", 0)
V = fenics.FunctionSpace(mesh, "CG", 1)
gradient = fenics.Function(U)
u = fenics.Function(U)

gradient_vec = np.loadtxt("gradient_vec.out")
u_vec = np.loadtxt("solution_vec.out")

gradient.vector()[:] = gradient_vec
u.vector()[:] = u_vec

regularity_test(gradient, beta,logspace_start=-11, logspace_stop=0, ndrop=3)

