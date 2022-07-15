class QuasiArmijoGoldstein(object):

	def __init__(self, alpha=0.5, gamma=0.99):

		self._alpha = alpha
		self._gamma = gamma


	def linesearch(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, _u, obj_u):


		def update_control(s):
			_u.assign(u)
			_u.axpy(s, u_minus_v)

		def phi(s):
			update_control(s)
			return obj(_u) + nonsmooth_obj(_u.data)

		alpha = self._alpha
		gamma = self._gamma

		s = gamma

		phi_u = obj_u
		phi_u_new = phi(s)

		while alpha*s*dual_gap > phi_u - phi_u_new:

			s *= gamma
			phi_u_new = phi(s)

		# Accept step
		u.assign(u_)

		return s


