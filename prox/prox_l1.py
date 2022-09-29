import numpy as np

def prox_l1(v, lam):
    """Proximal operator of the l1 norm.

    prox_l1(v, lam) is the proximal operator
    of the l1-norm with parameter lam.

    Parameters:
    -----------
        v : nd.array, float
            input array

        lam : float
            parameter

    Returns:
    --------
        proximal_point : ndarray
            proximal point of the l1-norm with parameter lam.

    References:
    -----------

    F. Schaipp: https://github.com/fabian-sp/GGLasso/blob/master/gglasso/solver/ggl_helper.py#L13

    Example 3.2.8 in

    A. Milzarek, Numerical methods and second order theory for nonsmooth problems,
    Dissertation, TUM, Munich, http://mediatum.ub.tum.de/?id=1289514
    """
    return np.sign(v) * np.maximum(np.abs(v)-lam, 0.0)
