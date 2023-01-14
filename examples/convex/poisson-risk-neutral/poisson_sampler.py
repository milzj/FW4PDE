import numpy as np


class PoissonSampler(object):

    def __init__(self, std=1.0):

        self._seed = 1
        self.std = std

    @property
    def seed(self):
        return self._seed

    def bump_seed(self):
        self._seed += 1


    def sample(self):

        self.bump_seed()
        np.random.seed(self.seed)
        Z = np.random.randn(4)
        return self.std*Z


if __name__ == "__main__":

    sampler = PoissonSampler()

    sample = sampler.sample()
    print(sample)
