class DunnScalingStepSize(object):
    """

    Step size rule for smooth problems provided in
    eqns. (4.9) and (4.10) in Dunn (1979), but we assume
    that the stopping criterion has already been called.
    In other words, the first case in (4.9) in Dunn (1979)
    is not considered.

    eqns. (4.7B) and (5.1A) in Dunn (1979) may suggest choosing theta such that
    L theta**2 approx 1, where L is the Lipschitz constant
    of the objective function's gradient.

    References:
    ----------

    Dunn, J.C.: Rates of convergence for conditional gradient algorithms 
    near singular and nonsingular extremals.
    SIAM J. Control Optim. 17(2), 187–211 (1979)
    """

    def __init__(self, theta=1.0):

        self._theta = theta
        self._beta = 1.0

    def __str__(self):

        s = "Dunn step size.\n"

        return s

    def compute_step_size(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration):

        u_new.zero()

        u_minus_v_norm = u_minus_v.norm()
        theta = self._theta
        beta = self._beta
        q = theta**2*beta/u_minus_v_norm**2

        omega = 1.0
        if q < 1.0:
            omega = q

        self._beta = (1-omega)*beta+omega**2*u_minus_v_norm**2/(2*theta**2)

        if omega < 1.0:
            u_new.axpy(1-omega, u)
            u_new.axpy(omega, v)
        else:
            u_new.assign(v)

        u.assign(u_new)

        print(omega)
        return omega, 0



