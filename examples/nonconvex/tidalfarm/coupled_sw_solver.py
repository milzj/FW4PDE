from dolfin import *
from dolfin_adjoint import *

from base import BoundaryConditionSet

class CoupledSWSolver(object):

    def __init__(self, tidal_parameters, domain_farm):

        # Tidal parameters
        self.tidal_parameters = tidal_parameters

        # farm and domain
        self.mesh = domain_farm.mesh
        self.domain = domain_farm.domain
        self.site_dx = domain_farm.site_dx

        # function spaces
        mesh = domain_farm.mesh
        V_h = VectorElement("CG", mesh.ufl_cell(), 2)
        Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
        state_space = FunctionSpace(mesh, V_h*Q_h)

        self.state_space = state_space

        # boundary conditions
        bcs = BoundaryConditionSet()
        bcs.add_bc("u", Constant((2.0,0.0)), facet_id=1, bctype="strong_dirichlet")
        bcs.add_bc("eta", Constant(0.0), facet_id=2, bctype="strong_dirichlet")
        bcs.add_bc("u", facet_id=3, bctype="free_slip")

        self.bcs = bcs

        # Generate strong boundary conditions
        self.strong_bcs = self._generate_strong_bcs()

        # State initial condition
        self.ic = project(tidal_parameters.initial_condition, state_space)

    def _generate_strong_bcs(self):

        state_space = self.state_space
        facet_ids = self.domain.facet_ids
        bcs = self.bcs

        # Generate velocity boundary conditions
        bcs_u = []
        for _, expr, facet_id, _ in bcs.filter("u", "strong_dirichlet"):
            bc = DirichletBC(state_space.sub(0), expr, facet_ids, facet_id)
            bcs_u.append(bc)

        # Generate free-surface boundary conditions
        bcs_eta = []
        for _, expr, facet_id, _ in bcs.filter("eta", "strong_dirichlet"):
            bc = DirichletBC(state_space.sub(1), expr, facet_ids, facet_id)
            bcs_eta.append(bc)

        return bcs_u + bcs_eta


    def solve(self, control):

        # Get domain measures
        ds = self.domain.ds
        site_dx = self.site_dx

        # Get boundary conditions
        strong_bcs = self.strong_bcs
        bcs = self.bcs

        # Get mesh, state space, and ic
        mesh = self.mesh
        state_space = self.state_space
        ic = self.ic

        # Get source term
        f_u = self.tidal_parameters.source_term
        depth = self.tidal_parameters.depth
        g = self.tidal_parameters.gravity
        friction = self.tidal_parameters.bottom_friction
        viscosity = self.tidal_parameters.viscosity

        # Define test functions
        v, q = TestFunctions(state_space)
        state = Function(state_space)

        state.vector().axpy(1.0, ic.vector())

        u_mid, h_mid = split(state)

        # Define the water depth
        H = h_mid + depth

        # The normal direction
        n = FacetNormal(mesh)

        # Divergence term.
        Ct_mid = -H * inner(u_mid, grad(q)) * dx

        # The surface integral contribution from the divergence term
        bc_contr = -H * dot(u_mid, n) * q * ds

        for function_name, u_expr, facet_id, bctype in bcs.filter(function_name='u'):
            if bctype in ('free_slip'):
                # Subtract the divergence integral again
                bc_contr -= -H * dot(u_mid, n) * q * ds(facet_id)

        # don't integrate pressure gradient by parts (a.o. allows dg test function v)
        C_mid = g * inner(v, grad(h_mid)) * dx

        # Bottom friction
        norm_u_mid = inner(u_mid, u_mid)**0.5
        u_mag = dot(u_mid, u_mid)**0.5
        R_mid = friction / H * norm_u_mid * inner(u_mid, v) * dx

        # R_mid appears to be implemented in https://zenodo.org/record/224251
        R_mid += (control / H) * u_mag * inner(u_mid, v) * site_dx(1)

        # Advection term
        Ad_mid = inner(dot(grad(u_mid), u_mid), v) * dx

        # Viscosity
        D_mid = viscosity * inner(2*sym(grad(u_mid)), grad(v)) * dx

        # Create the final form
        G_mid = C_mid + Ct_mid + R_mid

        # Add the advection term
        G_mid += Ad_mid

        # Add the viscosity term
        G_mid += D_mid

        # Add the source term
        G_mid -= inner(f_u, v) * dx
        F = G_mid - bc_contr

        solve(F == 0, state, bcs = strong_bcs, J=derivative(F, state))

        return state




