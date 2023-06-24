from dolfin import *
from dolfin_adjoint import *

from coupled_sw_solver import CoupledSWSolver


class TidalProblem(object):

    def __init__(self, tidal_parameters, domain_mesh):

        set_working_tape(Tape())
        self.tidal_parameters = tidal_parameters
        self.domain_mesh = domain_mesh

        self.coupled_sw_solver = CoupledSWSolver(tidal_parameters, domain_mesh)


        # control space
        mesh = self.coupled_sw_solver.mesh
        control_space = FunctionSpace(mesh, "DG", 0)
        self._control_space = control_space

        # sparsity parameter and box constraints
        WtoMW = 1e-6
        cost_coefficient = 4800.0
        beta = WtoMW*cost_coefficient
        lb = Constant(0.0)
        ub = Constant(0.059)

        self.WtoMW = WtoMW
        self.cost_coefficient = cost_coefficient
        self.beta = beta
        self.lb = lb
        self.ub = ub

    @property
    def control_space(self):
        return self._control_space


    def __call__(self, control):

        rho = self.tidal_parameters.water_density
        site_dx = self.domain_mesh.site_dx

        state = self.coupled_sw_solver.solve(control)
        u, p = split(state)

        WtoMW = self.WtoMW
        rho = self.tidal_parameters.water_density

        power_functional = assemble(rho*control*inner(u,u)**1.5*site_dx(1))
        return -WtoMW*power_functional
