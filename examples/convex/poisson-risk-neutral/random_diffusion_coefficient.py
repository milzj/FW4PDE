import numpy as np

from dolfin import *
from dolfin_adjoint import *


class RandomDiffusionCoefficient(object):
    """Log-normal random diffusion coefficient.

    The coefficient is adapted from that used in

    Ahmad Ali, Ullmann, and Hinze (2017):
    https://epubs.siam.org/doi/abs/10.1137/16M109870X

    There the domain is (-1/2, 1/2)^2. We transform
    the coordinates to (0,1)^2 via x -> 2x+1/2.
    """

    def __init__(self, degree=1):

        kappa_str = ""
        kappa_str += "+0.84*cos(0.42*pi*x[0])*cos(0.42*pi*x[1])*a"
        kappa_str += "+0.45*cos(0.42*pi*x[0])*sin(1.17*pi*x[1])*b"
        kappa_str += "+0.45*sin(1.17*pi*x[0])*cos(0.42*pi*x[1])*c"
        kappa_str += "+0.25*sin(1.17*pi*x[0])*sin(1.17*pi*x[1])*d"


        kappa_str = kappa_str.replace("x[0]", "(2.0*x[0]+0.5)")
        kappa_str = kappa_str.replace("x[1]", "(2.0*x[1]+0.5)")


        self.kappa = Expression(kappa_str, a=0.0, b=0.0, c=0.0, d=0.0, pi=np.pi, degree=degree)

        self.exp_kappa = Expression("exp(kappa)", kappa=self.kappa, degree=degree)


    def sample(self, sample):

        self.kappa.a = sample[0]
        self.kappa.b = sample[1]
        self.kappa.c = sample[2]
        self.kappa.d = sample[3]

        return self.exp_kappa


