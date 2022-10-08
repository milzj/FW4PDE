class DemyanovRubinovAdaptiveStepSize(object):
    """Step size rule

    TODO: Revise implementation and description

    Increases M until

    phi(s) <= phi(0) - dual_gap(u) s + (1/2)s^2 M norm(d)**2,

    where s in [0,1] is the minimizer of the right-hand side.
    After M (and s) satisfy the inequality, the step size s
    is choosen and M is divided by 2 to ensure that M<= 2L,
    where L is the Lipschitz constant of the gradient of f.

    M is an estimate of the Lipschitz constant of the derivative
    of f(u+ s d) about s.

    The method requires an initial value of M which should
    be <= than the Lipschitz constant of the gradient of f.

    TODO: Should L1 norm instead of L2 norm be used?
    Should we increase M until
    f(x+sd) <= f(x) + f'(x)sd + 1/2 s^2 M norm(d)^2?
    rather than the above approach?

    References:
    ----------
    Demyanov, Vladimir F. and Rubinov, Aleksandr M.,
    Approximate methods in optimization problems, Elsevier, 
    New York, 1970

    Nikita Doikov and Yurii Nesterov. Gradient regularization of Newton
    method with Bregman distances, 2021. URL: https://arxiv.org/abs/
    2112.02952, doi:10.48550/ARXIV.2112.02952.

    """
    def __init__(self, M=1.0, ls_max = 1000):

        self._M = M
        self._ls_max = ls_max

    def __str__(self):

        s = "Demyanov--Rubinov adaptive step size.\n"

        return s

    def compute_step_size(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration):

        d_norm = u_minus_v.norm()

        M = self._M
        ls_max = self._ls_max
        u_new.zero()

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

        dual_gap2 = dual_gap - nonsmooth_obj(u.data) + nonsmooth_obj(v.data)
        s = min(1.0, dual_gap/d_norm**2/M)

        phi_u = obj_u + nonsmooth_obj(u.data)
        phi_u_new = phi(s)

        ls_calls = 1
        while  phi_u_new > phi_u - s*dual_gap + 0.5*s**2*M*d_norm**2  and ls_calls < ls_max:

            M = 2*M
            s = min(1.0, dual_gap/d_norm**2/M)
            phi_u_new = phi(s)
            ls_calls += 1

        # Accept step
        u.assign(u_new)
        # Update M
        self._M = 0.5*M

        return s, ls_calls

