import numpy as np
from scipy import optimize
import itertools

import fenics


class LogNormalField(object):
    """Implements a lognormal random field

    The Gaussian random field is defined by a truncated KL
    expansion using the separable covariance operator
    considered in Example 7.56 in Lord, Powell, Shardlow (2014).

    References:
    ----------

    G. J. Lord, C. E. Powell, and T. Shardlow, An Introduction to Computational
    Stochastic PDEs, Cambridge Texts Appl. Math. 50, Cambridge University Press, Cam-
    bridge, 2014, https://doi.org/10.1017/CBO9781139017329

    """

    def __init__(self, function_space , num_addends = 10, len_scale=0.1):
        # cos cos, sin cos, cos sin, sin sin

        self.len_scale = len_scale
        self.num_addends = num_addends
        self.function_space = function_space

        self.odd_list = np.arange(0, 2*num_addends)[1::2]
        self.even_list = np.arange(0, 2*num_addends)[0::2]
        self.a = 0.5

        self.compute_roots()
        self.compute_addends()
        self.compute_2d_addends()


    def compute_roots(self):

        len_scale = self.len_scale
        a = self.a

        num_addends = self.num_addends
        odd_roots = np.zeros(num_addends)
        even_roots = np.zeros(num_addends)

        #fodd transformed by multiplying by cos(pi*x*a)
        fodd = lambda x: 1.0/len_scale*np.cos(np.pi*x*a) - np.pi*x*np.sin(np.pi*x*a)
        k = 0
        for i in self.even_list:
            if i == 0:
                root = optimize.brentq(fodd, 0, i+1)
            else:
                root = optimize.brentq(fodd, i, i+1)
            odd_roots[k] = root
            k += 1

        self.odd_roots = odd_roots

        #feven 1/l tan(xa) + x = 0 <=> 1/l sin(xa) + x cos(xa) = 0
        feven = lambda x: 1.0/len_scale*np.sin(np.pi*x*a) + np.pi*x*np.cos(np.pi*x*a)
        k = 0
        for i in self.odd_list:
            if i == 0:
                root = optimize.brentq(feven, 0, i+1)
            else:
                root = optimize.brentq(feven, i, i+1)
            even_roots[k] = root
            k += 1

        self.even_roots = even_roots

        # Tests
        atol = 1e-3
        fodd = lambda x: 1.0/len_scale-np.pi*x*np.tan(np.pi*x*a)
        for x in odd_roots:
            assert fodd(x) < atol, "Error in root computation for fodd"

        assert np.all(np.diff(odd_roots) > 0.0) == True

        feven = lambda x: 1.0/len_scale*np.tan(np.pi*x*a) + np.pi*x
        for x in even_roots:
            assert feven(x) < atol, "Error in root computation for feven"

        assert np.all(np.diff(even_roots) > 0.0) == True


    def compute_addends(self):

            a = self.a
            len_scale = self.len_scale
            num_addends = self.num_addends
            eigenfunctions_times_sqrt = []

            # odd
            k = 0
            for i in self.odd_list:
                omega = np.pi*self.odd_roots[k]
                A = 1.0/np.sqrt(a+np.sin(2*omega*a)/2/omega)
                nu = 2.0/len_scale/(omega**2+1/len_scale**2)
                eigenfunctions_times_sqrt.append("{}*cos({}*x)".format(A*np.sqrt(nu), omega))
                k+=1

            # even
            k = 0
            for i in self.even_list:
                omega = np.pi*self.even_roots[k]
                B = 1.0/np.sqrt(a-np.sin(2*omega*a)/2/omega)
                nu = 2.0/len_scale/(omega**2+1/len_scale**2)
                eigenfunctions_times_sqrt.append("{}*sin({}*x)".format(B*np.sqrt(nu), omega))
                k+=1

            self.eigenfunctions_times_sqrt = eigenfunctions_times_sqrt

    def compute_2d_addends(self):

        eigenfunctions_times_sqrt = self.eigenfunctions_times_sqrt

        products = list(itertools.product(eigenfunctions_times_sqrt, eigenfunctions_times_sqrt))

        addends = []

        v = fenics.Function(self.function_space)

        for i in products:
            a, b = i
            a = a.replace("x", "(x[0]-0.5)")
            b = b.replace("x", "(x[1]-0.5)")
            c = "{}*{}".format(a,b)
            v_expr = fenics.Expression(c, degree=1)
            v.interpolate(v_expr)
            addends.append(v.vector().get_local())

        self.addends = addends

    def sample(self, samples):

        addends = self.addends
        v = fenics.Function(self.function_space)
        w = fenics.Function(self.function_space)

#        field = np.zeros(len(addends[0]))

#        # Starting summation by smallest addend
#        for k in zip(addends[::-1], samples[::-1]):
#            a, b = k
#            field += a*b

        assert len(addends) == len(samples), "Number of samples differs from number of summands."

        field = np.sum([a*b for a, b in zip(addends, samples)], axis=0)

        v.vector()[:] = field
        w.interpolate(fenics.Expression("exp(v)", v=v, degree=1))

        return w

