from dolfin import *
from dolfin_adjoint import *

from base import RectangularDomain
from domain_parameters import DomainParameters
from farm_domain import FarmDomain

class DomainFarm(object):

    def __init__(self):

        # Computational domain
        domain_parameters = DomainParameters()
        x_min = domain_parameters.x_min
        x_max = domain_parameters.x_max
        y_min = domain_parameters.y_min
        y_max = domain_parameters.y_max
        n = domain_parameters.n

        domain = RectangularDomain(x_min, y_min, x_max, y_max, nx=n, ny=n)
        mesh = domain.mesh

        domains = MeshFunction('size_t', mesh, mesh.topology().dim())
        domains.set_all(0)
        farm_domain = FarmDomain(x_max=x_max, y_max=y_max)
        farm_domain.mark(domains, 1)
        site_dx = Measure("dx", domain=mesh, subdomain_data = domains)

        self.n = n
        self.mesh = mesh
        self.domain = domain
        self.site_dx = site_dx
