import numpy as np

def proj_box(v, lb, ub):
    """Compute projection onto a box.

    The function uses np.clip to compute
    the projection onto a box.

    Parameters:
    -----------

        v : nparray or float
            input array
        lb, ub : nparray or float
            lower and upper bounds

    Returns:
    --------
        projected_array : ndarray
            projection of `v` onto box `[lb,ub]`.
    """

    return np.clip(v, lb, ub)
