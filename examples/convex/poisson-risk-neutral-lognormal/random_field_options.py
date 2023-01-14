import numpy as np

class RandomFieldOptions(object):

    def __init__(self):

            num_addends = 16
            num_random_vars = (2*num_addends)**2

            self._options = {
                        "rv_range": [-100.0, 100.0],
                        "std": np.sqrt(2.0),
                        "len_scale": 0.1,
                        "num_addends": num_addends,
                        "num_rvs": num_random_vars
                        }

    @property
    def options(self):
        return self._options
