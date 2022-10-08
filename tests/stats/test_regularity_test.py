import pytest

import numpy as np
import fenics

from stats import regularity_test

def test_regularity_test():
    n = 64
    beta = 1e-2
    seed = 1234

    mesh = fenics.UnitSquareMesh(n,n)
    U = fenics.FunctionSpace(mesh, "DG", 0)

    gradient = fenics.Function(U)
    np.random.seed(seed)
    gradient.vector()[:] = beta + 1e-7*np.random.randn(U.dim())

    figure_name="test"
    assert regularity_test(gradient, beta, figure_name=figure_name)

