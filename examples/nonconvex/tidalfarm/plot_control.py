import numpy as np
from dolfin import *

from domain import *
from domain_parameters import DomainParameters
from rectangle_domain import RectangularDomain

import matplotlib.pyplot as plt
# plt.rcParams['text.usetex'] = True

input_filename = "output/" + sys.argv[1]
u_vec = np.loadtxt(input_filename + ".txt")

domain_parameters = DomainParameters()
x_min = domain_parameters.x_min
x_max = domain_parameters.x_max
y_min = domain_parameters.y_min
y_max = domain_parameters.y_max
n = domain_parameters.n

domain = RectangularDomain(x_min, y_min, x_max, y_max, nx=n, ny=n)
mesh = domain.mesh

control_space = FunctionSpace(domain.mesh, "DG", 0)
control = Function(control_space)


scaling = 1
scaling = 100.0/max(u_vec)

control.vector()[:] = scaling*u_vec

plt.set_cmap("coolwarm")
c = plot(control)
plt.gca().set_xlabel("meters")
plt.gca().set_ylabel("meters")
cb = plt.colorbar(c, label="Turbine density", shrink=1, orientation="horizontal")
# https://stackoverflow.com/questions/34458949/matplotlib-colorbar-formatting
cb.ax.xaxis.set_major_formatter(plt.FuncFormatter('{:.0f}%'.format))

plt.savefig(input_filename + "_online_version" + ".pdf", bbox_inches="tight")
plt.savefig(input_filename + "_online_version" + ".png", bbox_inches="tight")

