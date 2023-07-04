class DecreasingAdaptiveStepSize(object):

    def __str__(self):

        s = "Decreasing step size with 'adaptivity'.\n"

        return s

    def compute_step_size(self, obj, nonsmooth_obj, u, v, u_minus_v, dual_gap, u_new, obj_u, iteration):

        s = 2.0/(iteration + 2.0)
        u_new.zero()
        u_new.axpy(1-s, u)
        u_new.axpy(s, v)

        val_decreasing = obj(u_new) + nonsmooth_obj(u_new.data)
        val_step = obj(v) + nonsmooth_obj(v.data)

        if val_step <= val_decreasing:
            u.assign(v)
            s = 1.0
        else:
            u.assign(u_new)
            obj(u)

        return s, 0



