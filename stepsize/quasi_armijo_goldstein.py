class QuasiArmijoGoldstein(object):

	def __init__(self, alpha=0.5, gamma=0.99):

		self._alpha = alpha
		self._gamma = gamma

	def __str__(self):

		s = "Quasi-Armijo-Goldstein line search.\n"

		return s


	def do_linesearch(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration):

		u_new.zero()
		print("Line search 0")

		def update_control(s):
			if update_control.s_new != s:
				u_new.zero()
				u_new.axpy(1.0-s, u)
				u_new.axpy(s, v)
				update_control.s_new = s

		update_control.s_new = 0

		def phi(s):
			update_control(s)
			val = obj(u_new) + nonsmooth_obj(u_new.data)
			return val

		alpha = self._alpha
		gamma = self._gamma

		s = 1.0

		phi_u = obj_u + nonsmooth_obj(u.data)
		phi_u_new = phi(s)

		ls_calls = 0
		while  phi_u < phi_u_new + alpha*s*dual_gap and ls_calls < 1000:
			s *= gamma
			phi_u_new = phi(s)
			ls_calls += 1

		# Accept step
		u.assign(u_new)

		return s, ls_calls


