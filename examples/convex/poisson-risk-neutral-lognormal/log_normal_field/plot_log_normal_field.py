import fenics

from gaussian_sampler import GaussianSampler
from truncated_gaussian_sampler import TruncatedGaussianSampler
from log_normal_field import LogNormalField

import matplotlib.pyplot as plt

import numpy as np

def plot_log_normal_field(outdir, n, num_addends):

    std = np.sqrt(2.0)
    sampler = TruncatedGaussianSampler(std=std)
    num_addends = num_addends
    num_rvs = (2*num_addends)**2
    N = 20

    mesh = fenics.UnitSquareMesh(n,n)
    U = fenics.FunctionSpace(mesh, "CG", 1)
    u = fenics.Function(U)
    len_scale = 0.1
    exp_kappa = LogNormalField(U, num_addends=num_addends, len_scale=len_scale)

    for i in range(N):

        sample = sampler.sample(num_rvs)
        exp_kappa_sample = exp_kappa.sample(sample)

        u.interpolate(exp_kappa_sample)

        c = fenics.plot(u)
        plt.colorbar(c)

        plt.title(r"Sample of $\kappa$ $(i={})$".format(i))
        filename = outdir + "exp_kappa" + "_sample=" + str(i)
        plt.savefig(filename)
        plt.close()


if __name__ == "__main__":

    import sys, os

    n = 64
    num_addends = 10

    outdir = "log_normal_field/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    plot_log_normal_field(outdir, n, num_addends)
