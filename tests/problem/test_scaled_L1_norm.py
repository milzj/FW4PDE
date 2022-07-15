import pytest

from dolfin import *
import numpy as np
from problem import ScaledL1Norm

"""
References:
-----------
Jan Blechta (2014), https://fenicsproject.org/qa/3329/ufl-l-1-norm-of-a-function/
"""

import dolfin
import numpy as np
from base import norm_L1

@pytest.mark.parametrize("seed", [1234, 12345, 123456])
@pytest.mark.parametrize("n", [64, 128, 356])
@pytest.mark.parametrize("beta", [0.0, 1e-3])
def test_scaled_L1_norm(seed, n, beta):

	atol = 1e-14
	mesh = UnitSquareMesh(dolfin.MPI.comm_self, n, n)

	W = FunctionSpace(mesh, "DG", 0)
	u = Function(W)
	w = Function(W)

	np.random.seed(seed)
	u_vec = np.random.randn(W.dim())
	u.vector().set_local(u_vec)
	scaled_L1_norm = ScaledL1Norm(W, beta)

	L1 = scaled_L1_norm(u)

	F = Constant(beta)*abs(u)*dx(None, {'quadrature_degre': 5})
	f = assemble(F)

	aerr = f-L1
	assert aerr < atol
	assert beta == scaled_L1_norm.beta
