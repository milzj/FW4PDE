class QuasiArmijoGoldstein(object):

	def __init__(self, alpha=0.5, gamma=0.99):

		self._alpha = alpha
		self._gamma = gamma

	def __str__(self):

		s = "Quasi-Armijo-Goldstein line search.\n"

		return s


	def do_linesearch(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, _u, obj_u, iteration):

		u_new = u.copy()
		print("Line search 0")

		def update_control(s):
			if update_control.s_new != s:
				u_new.assign(u)
				u_new.axpy(-s, u_minus_v)
				update_control.s_new = s

		update_control.s_new = 0

		def phi(s):
			print("Line search 4")
			update_control(s)
			print("Line search 5")
			val = obj(u_new)
			print("Line search 6")
			val += nonsmooth_obj(u_new.data)
			print("Line search 7 {}".format(val))
			return val

		alpha = self._alpha
		gamma = self._gamma

		s = 1.0

		print("Line search 1")
		phi_u = obj_u + nonsmooth_obj(u.data)
		print("Line search 2 {}".format(phi_u))
		phi_u_new = phi(s)
		print("Line search 3 {}".format(phi_u_new))

		ls_calls = 0
		while alpha*s*dual_gap > phi_u - phi_u_new and ls_calls < 100:
			print("ls_call={}\n".format(ls_calls))
			print("ls_call={}\n".format(alpha*s*dual_gap))
			print("ls_call={}\n".format(phi_u-phi_u_new))
			s *= gamma
			phi_u_new = phi(s)
			ls_calls += 1

		# Accept step
		return s, u_new, ls_calls


