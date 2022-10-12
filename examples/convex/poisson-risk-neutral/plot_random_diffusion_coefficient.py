import fenics

from poisson_sampler import PoissonSampler
from random_diffusion_coefficient import RandomDiffusionCoefficient

import matplotlib.pyplot as plt


def plot_random_diffusion_coefficient(outdir, n, num_samples):

    sampler = PoissonSampler()
    exp_kappa = RandomDiffusionCoefficient()

    mesh = fenics.UnitSquareMesh(n,n)
    U = fenics.FunctionSpace(mesh, "CG", 1)
    u = fenics.Function(U)

    for i in range(num_samples):

        sample = sampler.sample()
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
    num_samples = 10

    outdir = "random_diffusion_coefficient/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)


    plot_random_diffusion_coefficient(outdir, n, num_samples)
