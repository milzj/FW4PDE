
from dolfin import *
import matplotlib.pyplot as plt
from rectangle_domain import RectangularDomain


x_max = 2000
y_max = 1000
N = 20

domain = RectangularDomain(0, 0, x_max, y_max, nx=N, ny=N)
mesh = domain.mesh
plot(mesh, title="Rectangle")
plt.savefig("output/mesh.pdf")
plt.savefig("output/mesh.png")


class FarmDomain(SubDomain):
    def inside(self, x, on_boundary):
        return between(x[0], (0.375*x_max, 0.625*x_max)) and between(x[1], (0.35*y_max, 0.65*y_max))


farm_domain = FarmDomain()
domains = MeshFunction('size_t', mesh, mesh.topology().dim())
domains.set_all(0)


farm_domain.mark(domains, 1)
site_dx = Measure("dx", domain=mesh, subdomain_data = farm_domain)
plot(domains)
plt.savefig("output/domains.png")
plt.savefig("output/domains.pdf")
plt.close()
