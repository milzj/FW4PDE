"""Sobol' sequence

Compute the convergence rate for integrating functions
using Sobol' sequences. We are interested in measuring
the effect of shifting an unscrambled Sobol' sequence
using the shift discussed in [2]_ on the integration
quality.

We perform simulations for two test functions:

(1) A test function considered in [2]_; here called
art_1.

(2) The test function "type_a" considered in [1]_.

References:
-----------

..  [1] Pamphile Tupui Roy,
    https://gist.github.com/tupui/fb6e219b1dd2316b7498ebce231bfff5, 2020

..  [2] Art B. Owen. On dropping the first Sobol’ point. In A. Keller, editor,
    Monte Carlo and quasi-Monte Carlo methods,
    Springer Proc. Math. Stat. 387, pages 71–86. Springer, Cham, 2022.
    doi:10.1007/978-3-030-98319-2_4.

"""

import os
from collections import namedtuple

import numpy as np
from scipy.stats import qmc
from scipy import stats

import matplotlib.pyplot as plt

ms = np.arange(4, 15)

path = 'sobol_convergence'
os.makedirs(path, exist_ok=True)

def art_1(sample):
    return np.sum(np.exp(sample) + 1.0 - np.exp(1.0), axis=1)

def type_a(sample, dim=30):
    # true value 1
    a = np.arange(1, dim + 1)
    f = 1.
    for i in range(dim):
        f *= (abs(4. * sample[:, i] - 2) + a[i]) / (1. + a[i])
    return f

def conv_method(sampler, func, dim, m, n_conv, mean):

    samples = [sampler(dim, m) for _ in range(n_conv)]
    samples = np.array(samples)

    evals = [np.mean(func(sample)) for sample in samples]
    squared_errors = (mean - np.array(evals)) ** 2
    rmse = np.mean(squared_errors) ** 0.5

    return rmse


def _sampler_rand(dim, m):

    return np.random.rand(2**m, dim)


def _sampler_sobol_scrambled(dim, m):

    sampler = qmc.Sobol(d=dim, scramble=True)
    q = sampler.random_base2(m=m)

    return q


def _sampler_sobol_shifted(dim, m):

    sampler = qmc.Sobol(d=dim, scramble=False)
    q = sampler.random_base2(m=m)
    q = q + 1/(2*2**m)
    assert np.all(q < 1.0), "Invalid shift of Sobol' sequence."

    return q


functions = namedtuple('functions', ['name', 'func', 'dim', 'mean'])

cases = [
    functions("Art 1", art_1, 10, 0.0),
    functions("Type A", type_a, 30, 1.0)
    ]

for case in cases:

    n_conv = 1000
    rmse_rand = []
    for m in ms:
        rmse = conv_method(_sampler_rand, case.func, case.dim, m, n_conv, case.mean)
        rmse_rand.append(rmse)

    n_conv = 1000
    rmse_sobol_scrambled = []
    for m in ms:
        rmse = conv_method(_sampler_sobol_scrambled, case.func, case.dim, m, n_conv, case.mean)
        rmse_sobol_scrambled.append(rmse)

    n_conv = 1
    rmse_sobol_shifted = []
    for m in ms:
        rmse = conv_method(_sampler_sobol_shifted, case.func, case.dim, m, n_conv, case.mean)
        rmse_sobol_shifted.append(rmse)

    fig, ax = plt.subplots()

    ax.plot(2**ms, rmse_sobol_scrambled, marker="o", label="Sobol' scrambled")
    ax.plot(2**ms, rmse_sobol_shifted, marker="s", label="Sobol' shifted")
    ax.plot(2**ms, rmse_rand, marker="s", label="rand")

    ax.set_xlabel("samples")
    ax.set_ylabel("RMSE")
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xticks(2**ms)
    ax.set_xticklabels([fr'$2^{{{m}}}$' for m in ms])

    ax.legend(loc="lower left")
    fig.tight_layout()
    func = case.name
    fig.savefig(os.path.join(path, f"sobol_integration_{func}.png"))

