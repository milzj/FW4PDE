import pytest

import dolfin
import numpy as np
from problem import BoxConstraints


def test_box_constraints():

	lb = dolfin.Constant(-30.0)
	ub = dolfin.Constant(30.0)

	mesh = dolfin.UnitSquareMesh(6, 6)
	W = dolfin.FunctionSpace(mesh, "DG", 0)

	box_constraints = BoxConstraints(W, lb, ub)

	atol = 1e-15

	assert dolfin.errornorm(box_constraints._lb, lb, degree_rise = 0, mesh=mesh) < atol
	assert dolfin.errornorm(box_constraints._ub, ub, degree_rise = 0, mesh=mesh) < atol


	lb = dolfin.Constant(-10.0)
	ub = dolfin.Expression('x[0] <= 0.25 ? 0 : -5.0+20.0*x[0]', degree=0)

	box_constraints = BoxConstraints(W, lb, ub)

	atol = 1e-15

	assert dolfin.errornorm(box_constraints._lb, lb, degree_rise = 0, mesh=mesh) < atol
	assert dolfin.errornorm(box_constraints._ub, ub, degree_rise = 0, mesh=mesh) < atol
