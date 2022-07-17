class DecreasingStepSize(object):

	def __str__(self):

		s = "Fixed step size.\n"

		return s

	def do_linesearch(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration):

		s = 2.0/(iteration + 2.0)
		u_new.zero()
		u_new.axpy(1-s, u)
		u_new.axpy(s, v)

		u.assign(u_new)

		return s, 0



