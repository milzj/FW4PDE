import pytest

import dolfin
import numpy as np
from problem import BoxConstraints


def test_box_constraints():

	# Constant, finite bounds
	lb = dolfin.Constant(-30.0)
	ub = dolfin.Constant(30.0)

	mesh = dolfin.UnitSquareMesh(64, 64)
	W = dolfin.FunctionSpace(mesh, "DG", 0)

	box_constraints = BoxConstraints(W, lb, ub)

	atol = 1e-15

	assert dolfin.errornorm(box_constraints._lb, lb, degree_rise = 0, mesh=mesh) < atol
	assert dolfin.errornorm(box_constraints._ub, ub, degree_rise = 0, mesh=mesh) < atol

	assert dolfin.errornorm(lb, box_constraints._lb, degree_rise = 0, mesh=mesh) < atol
	assert dolfin.errornorm(ub, box_constraints._ub, degree_rise = 0, mesh=mesh) < atol

	# Infinite lower bound, finite upper bound
	lb = dolfin.Constant(-np.inf)
	ub = dolfin.Constant(30.0)

	mesh = dolfin.UnitSquareMesh(64, 64)
	W = dolfin.FunctionSpace(mesh, "DG", 0)

	box_constraints = BoxConstraints(W, lb, ub)

	assert (box_constraints.lb == lb.values()[0]).all()
	assert (box_constraints.ub == ub.values()[0]).all()

	# Projection v.s. interpolation

	ub = dolfin.Expression('x[0] <= 0.25 ? 0 : -5.0+20.0*x[0]', degree=0)
	u = dolfin.Function(W)
	u.interpolate(ub)

	assert dolfin.errornorm(dolfin.project(ub,W), u, degree_rise=0, mesh=mesh) < atol

	ub = dolfin.Constant(np.inf)
	u = dolfin.Function(W)
	u.interpolate(ub)
	v = dolfin.project(ub, W)

	assert (v.vector()[:] == u.vector()[:]).all()

	# Nonconstant upper bound

	lb = dolfin.Constant(-10.0)
	ub = dolfin.Expression('x[0] <= 0.25 ? 0 : -5.0+20.0*x[0]', degree=0)

	box_constraints = BoxConstraints(W, lb, ub)

	atol = 1e-15

	assert dolfin.errornorm(box_constraints._lb, lb, degree_rise = 0, mesh=mesh) < atol
	assert dolfin.errornorm(box_constraints._ub, ub, degree_rise = 0, mesh=mesh) < atol

	assert dolfin.errornorm(lb, box_constraints._lb, degree_rise = 0, mesh=mesh) < atol
	assert dolfin.errornorm(ub, box_constraints._ub, degree_rise = 0, mesh=mesh) < atol
