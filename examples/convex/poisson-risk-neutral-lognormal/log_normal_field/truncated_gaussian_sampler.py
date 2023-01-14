import numpy as np
from scipy.stats import truncnorm

class TruncatedGaussianSampler(object):

    def __init__(self, std=1.0, rv_range=[-100.0, 100.]):

        self._seed = 1
        self.std = std
        self.rv_range = rv_range

    @property
    def seed(self):
        return self._seed

    def bump_seed(self):
        self._seed += 1


    def sample(self, num_rvs):

        a, b = self.rv_range
        std = self.std

        self.bump_seed()
        np.random.seed(self.seed)
        Z = truncnorm.rvs(a/std, b/std, loc=0.0, scale=std, size=num_rvs)

        return Z


if __name__ == "__main__":

    sampler = TruncatedGaussianSampler()

    sample = sampler.sample(4)
    print(sample)
