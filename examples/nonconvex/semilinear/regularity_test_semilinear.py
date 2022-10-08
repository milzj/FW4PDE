import fenics
import numpy as np

from stats import regularity_test

n = 256
beta = 0.15
mesh = fenics.UnitSquareMesh(n,n)
U = fenics.FunctionSpace(mesh, "DG", 0)
V = fenics.FunctionSpace(mesh, "CG", 1)
gradient = fenics.Function(U)
u = fenics.Function(U)

gradient_vec = np.loadtxt("gradient_vec_n_256.out")

gradient.vector()[:] = gradient_vec

regularity_test(gradient, beta,logspace_start=-11, logspace_stop=0, ndrop=1)

