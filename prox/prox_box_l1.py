from .prox_l1 import prox_l1
from .proj_box import proj_box

def prox_box_l1(v, lb, ub, lam):
    """Compute proximal operator for box-constraints and l1-norm.

    The proximal operator is computed using a composition formula.

    Parameters:
    -----------
        v : ndarray or float
            input array
        lb, ub : ndarray or float
            lower and upper bounds
        lam : float
            parameter

    Returns:
    -------
        proximal_point : ndarray
            proximal point of input array

    References:
    -----------

    Example 3.2.9 in

    A. Milzarek, Numerical methods and second order theory for nonsmooth problems,
    Dissertation, TUM, Munich, http://mediatum.ub.tum.de/?id=1289514
    """

    w = prox_l1(v, lam)

    return proj_box(w, lb, ub)


