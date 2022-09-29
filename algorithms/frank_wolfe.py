import numpy as np

"""

References:
-----------

[HJN15] Z. Harchaoui, A. Juditsky, and A. Nemirovski. Conditional gradient algorithms for
    norm-regularized smooth convex optimization. Math. Program., 152(1-2, Ser. A):75–
    112, 2015.

[KW21] 	K. Kunisch and D. Walter. On fast convergence rates for generalized conditional gradient
    methods with backtracking stepsize. arXiv:2109.15217v1

"""

class FrankWolfe(object):

    def __init__(self, problem, nonsmooth_functional=None, initial_point=None,
            stepsize=None, lmo=None, options={}, callback={}):

        self.problem = problem
        self.nonsmooth_functional = nonsmooth_functional
        self.stepsize = stepsize
        self.lmo = lmo

        _options = self.default_options()
        _options.update(options)
        self.options = _options

        self.callback = callback

        self.data = {"control_final": initial_point,
                "control_best": [],
                "objective_final": np.inf,
                "objective_best": np.inf,
                "objective_lower": -np.inf,
                "iteration": 0,
                "dual_gap": -np.inf
        }


    def default_options(self):

        options = {
            "gtol": 1e-4,
            "ftol": 1e-4,
            "maxiter": 200,
            "display": 2
        }

        return options

    def __str__(self):

        s = "Franke-Wolfe method.\n"
        s += "Maximum iterations: {}\n".format(self.options["maxiter"])
        s += "{}".format(self.stepsize.__str__())

        return s


    def solve(self):

        print(self.__str__())

        u = self.data["control_final"]
        v = u.copy().zero()

        u_minus_v = u.copy().zero()

        u_new = u.copy()

        obj = self.problem.obj
        nonsmooth_obj = self.nonsmooth_functional


        iteration = 0
        while True:
            print("iteration={}".format(iteration))

            obj_u = obj(u)
            derivative = obj.derivative(u)
            gradient = derivative.primal()

            # Compute direction
            self.lmo.solve(gradient, v)

            # Update stats

            self.data.update({"control_final": u})
            objective_final = obj_u + nonsmooth_obj(u.data)
            self.data.update({"objective_final": objective_final})
            # Update best function value
            objective_best = self.data["objective_best"]
            if objective_final < objective_best:
                self.data.update({"control_best": u})
                self.data.update({"objective_best": objective_final})
                objective_best = objective_final


            # Check for convergence (w=u-v)
            u_minus_v.assign(u)
            u_minus_v.axpy(-1.0, v)

            dual_gap = derivative.apply(u_minus_v) + \
                    nonsmooth_obj(u.data) - \
                    nonsmooth_obj(v.data)


            # Compute lower bound on true objective function value (see HJN15 eq. (16) and p. 5 in [KW21])
            objective_lower = self.data["objective_lower"]
            objective_lower = max(objective_final - dual_gap, objective_lower)
            self.data.update({"objective_lower": objective_lower})


            print("dual_gap={}".format(dual_gap))
            print("objective_lower={}".format(objective_lower))
            print("objective_best={}".format(objective_best))
            print("objective_final={}".format(objective_final))

            self.data.update({"dual_gap": dual_gap})

            if dual_gap <= self.options["gtol"]:
                print("dual_gap<={}".format(dual_gap))
                break

            best_minus_lower = objective_best - objective_lower
            print("objective_best - objective_lower<={}".format(best_minus_lower))

            # For convex objectives best_minus_lower is nonnegative
            if best_minus_lower >= 0.0 and best_minus_lower <= self.options["ftol"]:
                print("objective_best - objective_lower<={}".format(best_minus_lower))
                break

            if iteration >= self.options["maxiter"]:
                break



            # Perform line search
            s, ls_calls = self.stepsize.compute_step_size(obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration)

            print("ls_calls={}".format(ls_calls))
            print("\n")

            iteration += 1



        return self.data
