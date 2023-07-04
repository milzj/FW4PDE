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
                "control_best": initial_point,
                "objective_final": np.inf,
                "objective_best": np.inf,
                "objective_lower": -np.inf,
                "best_minus_lower": np.inf,
                "iteration": -1,
                "dual_gap": -np.inf,
                "ls_calls": -1,
                "step_size": -1.0,
                "status": 0
        }

        self.iter_heading()

    def default_options(self):

        options = {
            "gtol": 1e-4,
            "ftol": 1e-4,
            "maxiter": 200,
            "display": 3
        }

        return options

    @property
    def iter_status(self):
        keys =  ["iteration", "objective_final", "dual_gap", "best_minus_lower", "ls_calls", "step_size"]
        if self.data["iteration"] == 0:
            output = [self.data[k] for k in keys[0:-2]]
            return "{:<10d} {:<15e} {:<15e} {:<25e}".format(*output)
        else:
            output = [self.data[k] for k in keys if k in self.data]
            return "{:<10d} {:<15e} {:<15e} {:<25e} {:<10d} {:<15e}".format(*output)


    def display(self, text, print_level, iteration=-1):
        """

        print levels

        0 : no output
        1 : information at start and end of optimization
        2 : general information for every tenth optimization loop
        3 : general information for each optimization loop
        4 :
        """

        if iteration >= 0 and iteration % 10 == 0 and self.options["display"] > 1:
            print(self.__heading)

        if iteration % 10 == 0 and 2 == self.options["display"] and print_level > 1:
            print(text)

        elif 2 < print_level <= self.options["display"]:
            print(text)

    def __str__(self):

        s = "Franke--Wolfe method.\n"
        s += "Maximum number iterations: {}\n".format(self.options["maxiter"])
        s += "Step size: {}".format(self.stepsize.__str__())

        return s

    def iter_heading(self):

        headings = ["iter", "obj.val", "dual gap", "obj.val(best-lower)", "ls calls", "stepsize"]
        self.__heading = "{:10s} {:15s} {:15s} {:25s} {:10s} {:15s}".format(*headings)


    def check_convergence(self):

        data = self.data
        options = self.options

        status = 0

        if data["iteration"] >= options["maxiter"]:
            status = -1
        if data["dual_gap"] <= options["gtol"]:
            status = 1
        # For convex objectives best_minus_lower is nonnegative
        if 0.0 <= data["best_minus_lower"] <= options["ftol"]:
            status = 2

        self.data.update({"status": status})

        return status

    def termination_msg(self, status):

        msg = "Termination message: "
        if status == -1:
            msg += "Maximum number of iterations reached."
        elif status == 1:
            msg += "Dual gap <= tolerance."
        else:
            msg += "Difference of best and lower objective function values <= tolerance."

        return msg

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
            self.data.update({"iteration": iteration})

            obj_u = obj(u)
            derivative = obj.derivative(u)
            gradient = derivative.primal()

            # Compute direction
            self.lmo.solve(gradient, v)

            # Update stats
            self.data.update({"control_final": u})
            self.data.update({"gradient_final": gradient})
            objective_final = obj_u + nonsmooth_obj(u.data)
            self.data.update({"objective_final": objective_final})

            # Update best function value
            objective_best = self.data["objective_best"]

            if objective_final < objective_best:

                self.data.update({"control_best": u})
                self.data.update({"gradient_best": gradient})
                self.data.update({"objective_best": objective_final})
                objective_best = objective_final

            u_minus_v.assign(u)
            u_minus_v.axpy(-1.0, v)

            dual_gap = derivative.apply(u_minus_v) + \
                    nonsmooth_obj(u.data) - \
                    nonsmooth_obj(v.data)

            self.data.update({"dual_gap": dual_gap})

            # Compute lower bound on true objective function value (see HJN15 eq. (16) and p. 5 in [KW21])
            objective_lower = self.data["objective_lower"]
            objective_lower = max(objective_final - dual_gap, objective_lower)
            self.data.update({"objective_lower": objective_lower})
            self.data.update({"best_minus_lower": objective_best - objective_lower})

            self.display(self.iter_status, 3, iteration=iteration)
            status = self.check_convergence()

            if status != 0:
                break;

            # Perform line search
            s, ls_calls = self.stepsize.compute_step_size(obj, nonsmooth_obj, u, v, u_minus_v,\
                                    dual_gap, u_new, obj_u, iteration)

            self.data.update({"ls_calls": ls_calls})
            self.data.update({"step_size": s})

            iteration += 1

        print(self.termination_msg(status))

        return self.data
