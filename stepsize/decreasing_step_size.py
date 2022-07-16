class DecreasingStepSize(object):

	def __str__(self):

		s = "Fixed step size.\n"

		return s

	def do_linesearch(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, _u, obj_u, iteration):

		u_new = u.copy().zero()

		s = 2.0/(iteration + 2.0)
		u_new.axpy(1-s, u)
		u_new.axpy(s, v)

		return s, u_new, 0



