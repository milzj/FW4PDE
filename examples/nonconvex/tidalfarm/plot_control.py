import numpy as np
from dolfin import *

from domain import *
from rectangle_domain import RectangularDomain

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

input_filename = "output/" + sys.argv[1]
u_vec = np.loadtxt(input_filename + ".txt")

x_max = 2000
y_max = 1000
N = 100

domain = RectangularDomain(0, 0, x_max, y_max, nx=N, ny=N)
mesh = domain.mesh

control_space = FunctionSpace(domain.mesh, "DG", 0)
control = Function(control_space)

control.vector()[:] = u_vec

plt.set_cmap("coolwarm")
c = plot(control)
plt.colorbar(c)
plt.savefig(input_filename + "_online_version" + ".pdf", bbox_inches="tight")
plt.savefig(input_filename + "_online_version" + ".png", bbox_inches="tight")

