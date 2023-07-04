class DunnHarshbargerStepSize(object):
    """

    References:
    ----------
    J. C. Dunn and S. Harshbarger. Conditional gradient algorithms with open loop step
    size rules. J. Math. Anal. Appl., 62(2):432–444, 1978. doi:10.1016/0022-247X(78)90137-3.

    """

    def __str__(self):

        s = "Dunn--Harshbarger step size.\n"

        return s

    def compute_step_size(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration):

        u_new.zero()

        s = 1.0
        for it in range(1, iteration+1):
            s = s - 0.5*s**2

        u_new.axpy(1-s, u)
        u_new.axpy(s, v)

        u.assign(u_new)

        return s, 0



