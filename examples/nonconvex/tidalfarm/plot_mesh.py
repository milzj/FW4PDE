
from dolfin import *

import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

from rectangle_domain import RectangularDomain
from domain_parameters import DomainParameters


domain_parameters = DomainParameters()
x_min = domain_parameters.x_min
x_max = domain_parameters.x_max
y_min = domain_parameters.y_min
y_max = domain_parameters.y_max
n = 100

domain = RectangularDomain(x_min, y_min, x_max, y_max, nx=n, ny=n)
mesh = domain.mesh
plot(mesh, title="Rectangle")
plt.savefig("output/mesh_n={}.pdf".format(n))
plt.savefig("output/mesh_n={}.png".format(n))


class FarmDomain(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (0.375*x_max, 0.625*x_max)) and between(x[1], (0.35*y_max, 0.65*y_max))


farm_domain = FarmDomain()
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(0)

farm_domain.mark(domains, 1)
site_dx = Measure("dx", domain=mesh, subdomain_data = domains)
site_area = assemble(Constant(1.0)*site_dx(1))
print("Site area (m^2): {}".format(site_area))

plot(domains)
plt.savefig("output/domains_n={}.png".format(n))
plt.savefig("output/domains_n={}.pdf".format(n))
plt.close()
