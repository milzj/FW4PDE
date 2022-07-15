import numpy as np

class NormL1(object):
	"""

	See p. 137 in [1].

	References:
	----------
	[1] E. Casas, K. Kunisch, F. Tröltzsch, Optimal control of PDEs and FE-approximation,


	"""

	def __init__(self, function_space):

		import fenics

		if function_space.ufl_element() != fenics.FiniteElement('Discontinuous Lagrange', fenics.triangle, 0):
			raise TypeError("function_space={} should be DG0.".format(function_space))


		v = fenics.TestFunction(function_space)

		w = fenics.assemble(fenics.Constant(1.0)*v*fenics.dx)
		self._w = w
		self._control_coefficients = np.zeros(function_space.dim())


	def __call__(self, control):

		self._control_coefficients[:] = control.vector()[:]

		return np.dot(np.abs(self._control_coefficients), self._w)
