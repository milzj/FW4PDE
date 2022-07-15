class FrankWolfe(object):

	def __init__(self, problem, nonsmooth_functional=None, initial_point=None,
			linesearch=None, options={}, callback={}):

		self.problem = problem
		self.nonsmooth_functional = nonsmooth_functional

		_options = self.default_options()
		_options.update(options)
		self.options = _options

		self.callback = callback

		self.data = {"control": initial_point, "iteration": 0}


	def default_options(self):

		options = {
			"gtol": 1e-4,
			"jtol": 1e-4,
			"maxiter": 200,
			"display": 2,
#			"line_search": "Quasi-Armijo-Goldstein",
#			"line_search_options": {"alpha": 0.5, "gamma": 0.99},
			"record": {"objective", "dual_gap"}
		}

		return options

	def __str__(self):

		s = "Franke-Wolfe method.\n"
		s += "Maximum iterations: {}\n".format(self.options["maxiter"])

		return s


	def solve(self):

		print(self.__str__())

		u = self.data["control"]
		v = u.copy()
		v.scale(0.0)

		u_minus_v = u.copy()
		u_minus_v.scale(0.0)

		_u = u.copy()

		obj = self.problem.obj
		nonsmooth_obj = self.nonsmooth_functional

		obj_u = obj(u)

		iteration = 0
		while True:

			derivative = obj.derivative(u)
			gradient = derivative.primal()

			# Compute direction


			# Update stats

			self.data.update({"control": u})

			# Check for convergence (w=u-v)
			u_minus_v.assign(u)
			u_minus_v.axpy(-1.0, v)

			dual_gap = derivative.apply(u_minus_v) + \
					nonsmooth_obj(u.data) - \
					nonsmooth_obj(v.data)

			print("dual_gap={}".format(dual_gap))

			if dual_gap <= self.options["gtol"]:
				break


			# Perform line search
			s = line_search(obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, _u, obj_u)
			obj_u = obj(u)


			iteration += 1



		return self.data
