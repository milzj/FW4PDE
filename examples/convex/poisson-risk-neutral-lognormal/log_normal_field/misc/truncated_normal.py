"""Truncated normal

We are interested in examining (i) the
empirical probability distribution
of truncated normal random variables
with large truncation interval (relative
to standard deviation); (ii) compare
the empirical probability distribution
of truncated normal random variables
generated using random samples with that
generated by a shifted Sobol' sequence
and the inverse transformation method; and
(iii) check whether the inverse transformation
method produces meaningful results.

The "shifted" Sobol' sequence is generated
using the shift discussed in [1]_.

The inverse transformation method for truncated
normal distributions may be inaccurate; see
Example 2.7 [2]_. The ppf implemented in [3]_
appears to produce meaningful results.


References:
-----------

..  [1] Art B. Owen. On dropping the first Sobol’ point. In A. Keller, editor,
    Monte Carlo and quasi-Monte Carlo methods,
    Springer Proc. Math. Stat. 387, pages 71–86. Springer, Cham, 2022.
    doi:10.1007/978-3-030-98319-2_4.

..  [2] Dirk P. Kroese. Monte Carlo Methods, lecture notes
    for graduate course on Monte Carlo methods given at the
    2011 Summer School of the Australian Mathematical Sciences Institute (AMSI),
    https://people.smp.uq.edu.au/DirkKroese/mccourse.pdf

..  [3] https://github.com/scipy/scipy/blob/v1.9.3/scipy/stats/_continuous_distns.py#L8108

"""

import os
path = 'sobol_convergence'
os.makedirs(path, exist_ok=True)


from scipy.stats import truncnorm
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc

fig, ax = plt.subplots(4, 1)

N = 100000
nbins = 50
sigma = 3.0
loc = 10.

# compare w/ the Notes in https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html
a = (-100.0 -loc)/sigma
b = (100.0-loc)/sigma

# random samples from truncated normal distribution
r = truncnorm.rvs(a, b, loc=loc, scale=sigma, size=N)
ax[0].hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=nbins)
ax[0].set_title("truncated normal variables")


# random samples from truncated distribution
r = loc + sigma*np.random.randn(N)
ax[1].hist(r, density=True, histtype='stepfilled', alpha=0.2, bins=nbins)
ax[1].set_title("normal variables")

# scrambled Sobol' sequence
sampler = qmc.Sobol(d=1, scramble=True, seed=1234)
m = 15
q = sampler.random_base2(m=m)
s = truncnorm.ppf(q, a, b, loc=loc, scale=sigma)
ax[2].hist(s, density=True, histtype='stepfilled', alpha=0.2, bins=nbins)
ax[2].set_title("truncated normal variables via ppf (w/ scrambled Sobol')")

# shifted Sobol' sequence
sampler = qmc.Sobol(d=1, scramble=False)
q = sampler.random_base2(m=m)
q = q + 1/(2*2**m)

assert np.all(q < 1.0), "Invalid shift of Sobol' sequence."

s = truncnorm.ppf(q, a, b, loc=loc, scale=sigma)
ax[3].hist(s, density=True, histtype='stepfilled', alpha=0.2, bins=nbins)
ax[3].set_title("truncated normal variables via ppf (w/ shifted Sobol')")

fig.tight_layout()
plt.savefig(os.path.join(path, "hist_truncated_normal.png"))
