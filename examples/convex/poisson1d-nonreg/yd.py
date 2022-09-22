import numpy as np
from fenics import *
import matplotlib.pyplot as plt


def convergence_rates(E_values, eps_values, show=True):
    from numpy import log
    r = []
    for i in range(1, len(eps_values)):
        r.append(log(E_values[i] / E_values[i - 1])
                 / log(eps_values[i] / eps_values[i - 1]))
    if show:
        print("Computed convergence rates: {}".format(r))
    return r


yd1 = "-1.0/16.0*x[0]*(8.0-8.0*pi+pi*pi+ 8.0*x[0])"
yd2 = "-1.0/16.0*x[0]*(pi*pi-8.0*x[0])"

yd = Expression("x[0] <= pi/4 ? {}: {}".format(yd1, yd2), degree = 1, pi=np.pi)

E_values = []
eps_values = []

for n in [2**8, 2**10, 2**12, 2**14, 2**16]:


	mesh = UnitIntervalMesh(n)
	V = FunctionSpace(mesh, "CG", 1)

	y = project(yd, V)
#	y = Function(V)
#	y.interpolate(yd)

	mesh = UnitIntervalMesh(2**2*n)
	err = errornorm(yd, y, mesh=mesh, degree_rise=0)
	print(err)

	eps_values.append(1/n)
	E_values.append(err**2)




convergence_rates(E_values, eps_values)

