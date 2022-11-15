import numpy as np
from scipy.stats import truncnorm
from scipy.stats import qmc

class ReferenceTruncatedGaussianSampler(object):

    def __init__(self, std=1.0, rv_range=[-100.0, 100.]):

        self.rv_range = rv_range
        self.std = std

    def reference_samples(self, base2_exp, num_rvs):

        std = self.std
        a, b = self.rv_range

        m = base2_exp

        sampler = qmc.Sobol(d=num_rvs, scramble=False)
        q = sampler.random_base2(m=m)
        q = q + 1/(2*2**m)

        assert np.all(q < 1.0), "Invalid shift of Sobol' sequence."

        s = truncnorm.ppf(q, a/std, b/std, loc=0, scale=std)
        return s



if __name__ == "__main__":

    sampler = ReferenceTruncatedGaussianSampler()

    sample = sampler.reference_samples(10, 9)
    print(sample.shape)
