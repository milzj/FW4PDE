import pytest

import numpy as np
from algorithms import NumpyBoxLMO

@pytest.mark.parametrize("n", [64, 128, 256])
@pytest.mark.parametrize("seed", [1234, 12345, 123456])
@pytest.mark.parametrize("beta", [0.1, 1e-3, 1e-2, 1.0])
def test_numpy_box_lmo(n, seed, beta):
	"""
	The solution to min_x (g, x) + beta norm(x,l1) s.t. lb <= x <= ub
	satisfies the fixed point equation

	x = projection(u-projection(u, -beta, beta), lb, ub),

	where u = x - g.
	"""

	lb = -1.5*np.ones(n)
	ub = 5*np.ones(n)

	atol = 1e-6
	rtol = 1e-15

	numpy_box_lmo = NumpyBoxLMO(lb, ub, beta)

	np.random.seed(seed)
	gradient = np.random.randn(n)

	numpy_box_lmo.solve(gradient)
	solution = numpy_box_lmo.solution

	def l1_norm(v):
		return np.sum(np.abs(v))


	def projection(v, a, b):
		_v = np.minimum(v, b)
		return np.maximum(a, _v)

	u = solution - gradient
	u_projection = projection(u, -beta, beta)
	_solution = projection(u-u_projection, lb, ub)

	assert np.linalg.norm(solution-_solution) < atol
	assert np.linalg.norm(solution-_solution)/np.linalg.norm(solution) < rtol

	np.random.seed(seed+1)
	gradient = np.random.randn(n)

	numpy_box_lmo.solve(gradient)
	solution = numpy_box_lmo.solution

	u = solution - gradient
	u_projection = projection(u, -beta, beta)
	_solution = projection(u-u_projection, lb, ub)

	assert np.linalg.norm(solution-_solution) < atol
	assert np.linalg.norm(solution-_solution)/np.linalg.norm(solution) < rtol
