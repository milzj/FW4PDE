import pytest

from dolfin import *
import numpy as np
from base import NormL1

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
def test_norm_L1_dim2(seed, n):

	rtol = 1e-13
	mesh = UnitSquareMesh(dolfin.MPI.comm_self, n, n)

	W = FunctionSpace(mesh, "DG", 0)
	u = Function(W)
	w = Function(W)

	np.random.seed(seed)
	u_vec = np.random.randn(W.dim())
	u.vector().set_local(u_vec)
	L1 = NormL1(W)(u)

	F = abs(u)*dx(None, {'quadrature_degre': 5})
	f = assemble(F)

	rel_err = (f-L1)/(1.0+f)
	assert rel_err < rtol

@pytest.mark.parametrize("seed", [1234, 12345, 123456])
@pytest.mark.parametrize("n", [64, 128, 356])
def test_norm_L1_dim1(seed, n):

	rtol = 1e-13
	mesh = UnitIntervalMesh(dolfin.MPI.comm_self, n)

	W = FunctionSpace(mesh, "DG", 0)
	u = Function(W)
	w = Function(W)

	np.random.seed(seed)
	u_vec = np.random.randn(W.dim())
	u.vector().set_local(u_vec)
	L1 = NormL1(W)(u)

	F = abs(u)*dx(None, {'quadrature_degre': 5})
	f = assemble(F)

	rel_err = (f-L1)/(1.0+f)
	assert rel_err < rtol

def test_raise_norm_L1_dim2():

	n = 4
	mesh = UnitSquareMesh(dolfin.MPI.comm_self, n, n)

	W = FunctionSpace(mesh, "DG", 1)
	u = Function(W)
	w = Function(W)

	with pytest.raises(Exception) as e_info:
		L1 = NormL1(W)(u)


def test_raise_norm_L1_dim1():

	n = 4
	mesh = UnitIntervalMesh(dolfin.MPI.comm_self, n)

	W = FunctionSpace(mesh, "DG", 1)
	u = Function(W)
	w = Function(W)

	with pytest.raises(Exception) as e_info:
		L1 = NormL1(W)(u)
