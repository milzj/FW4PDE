class DemyanovRubinovOptimalStepSize(object):
	"""Step size rule 

	Computes step size s by minimizing the right-hand side of

	phi(s) <= phi(0) - dual_gap(u) s + (1/2)s^2 H(u)(d,d),

	over s in [0,1], where H(u)(d,d) is the 
	Hessian evaluated in the directions d = u-v and
	dual_gap(u) is the dual gap evaluated
	at the current iterate u. Moreover,
	phi(s) = f(u+s d) + g(u+s d).

	The step size rule assumes f be
	convex, twice differentiable, and
	quadratic.

	References:
	----------
	Demyanov, Vladimir F. and Rubinov, Aleksandr M.,
	Approximate methods in optimization problems, Elsevier, 
	New York, 1970

	"""

	def __str__(self):

		s = "Demyanov--Rubinov optimal step size.\n"

		return s

	def do_linesearch(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration):

		u_new.zero()

		dHd = obj.hessian(u)(u_minus_v).apply(u_minus_v)

		if dHd < 0.0:
			raise ValueError("H(u)(d,d) is negative.")

		s = min(1.0, dual_gap/dHd)

		u_new.axpy(1-s, u)
		u_new.axpy(s, v)

		u.assign(u_new)

		return s, 0



