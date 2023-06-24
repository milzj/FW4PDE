from pathlib import Path
import sys

here = Path(__file__).parent
sys.path.insert(0, str(here.parent) + "/fw4pde")
sys.path.insert(0, str(here.parent) + "/misc")
sys.path.insert(0, str(here.parent) + "/prox")
sys.path.insert(0, str(here.parent) + "/stats")

import pytest
import numpy as np

def pytest_runtest_setup(item):
    np.random.seed(1234)
