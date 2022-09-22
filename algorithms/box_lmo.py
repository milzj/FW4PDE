import numpy as np

class NumpyBoxLMO(object):
	"""
	See Lem. 5.6 in Ref. [1]

	References:
	----------
	[1] 	K. Kunisch and D. Walter, On fast convergence rates for generalized conditional
		gradient methods with backtracking stepsize, preprint,
		https://arxiv.org/abs/2109.15217, 2021
	"""

	def __init__(self, lb, ub, beta):

		n = lb.size
		self._gradient = np.zeros(n)
		self._solution = np.zeros(n)

		self._lb = lb
		self._ub = ub
		self._beta = beta

	@property
	def solution(self):
		return self._solution

	def solve(self, gradient):
		self._solve(gradient)

	def _solve(self, gradient):
		"""
		Note:
		----

		We implement the cases grad(x) > beta and grad(x) < -beta
		to allow for meaningful results when beta is zero.
		"""

		beta = self._beta
		lb = self._lb
		ub = self._ub

		self._solution *= 0.0
		self._gradient[:] = gradient[:]


		idx = self._gradient > beta
		self._solution[idx] = lb[idx]

		idx = self._gradient < -beta
		self._solution[idx] = ub[idx]



class MoolaBoxLMO(NumpyBoxLMO):

	def __init__(self, lb, ub, beta):

		super().__init__(lb, ub, beta)


	def solve(self, gradient, v):

		self._solve(gradient.data.vector().get_local())

		v.data.vector().set_local(self._solution)
		v.bump_version()
