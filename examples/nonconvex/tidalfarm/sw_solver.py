from dolfin import *
from dolfin_adjoint import *

from base import RectangularDomain
from base import BoundaryConditionSet
from tidal_parameters import TidalParameters
from domain_parameters import DomainParameters
from farm_domain import FarmDomain

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


# Parameters
parameters = TidalParameters()
depth = parameters.depth
viscosity = parameters.viscosity
C_t = parameters.thrust_coefficient
A_t = parameters.turbine_cross_section
friction = parameters.bottom_friction
g = parameters.gravity
rho = parameters.water_density
f_u = parameters.source_term
initial_condition = parameters.initial_condition

# function spaces
V_h = VectorElement("CG", mesh.ufl_cell(), 2)
Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V_h*Q_h)

control_space = FunctionSpace(mesh, "DG", 0)
control = Function(control_space)

# boundary conditions
bcs = BoundaryConditionSet()
bcs.add_bc("u", Constant((2.0,0.0)), facet_id=1, bctype="strong_dirichlet")
bcs.add_bc("eta", Constant(0.0), facet_id=2, bctype="strong_dirichlet")
bcs.add_bc("u", facet_id=3, bctype="free_slip")

def generate_strong_bcs():

    facet_ids = domain.facet_ids

    # Generate velocity boundary conditions
    bcs_u = []
    for _, expr, facet_id, _ in bcs.filter("u", "strong_dirichlet"):
        bc = DirichletBC(W.sub(0), expr, facet_ids, facet_id)
        bcs_u.append(bc)

    # Generate free-surface boundary conditions
    bcs_eta = []
    for _, expr, facet_id, _ in bcs.filter("eta", "strong_dirichlet"):
        bc = DirichletBC(W.sub(1), expr, facet_ids, facet_id)
        bcs_eta.append(bc)

    return bcs_u + bcs_eta

strong_bcs = generate_strong_bcs()
ds = domain.ds

def steady_sw(control):

    # Get domain measures
    # Define test functions
    v, q = TestFunctions(W)
    state = Function(W)

    ic = project(initial_condition, W)
#    state.assign(ic, annotate=False)
    state.vector().axpy(1.0, ic.vector())

    u_mid, h_mid = split(state)

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
    #R_mid += (control * 0.5 * A_t * C_t / H) * u_mag * inner(u_mid, v) * site_dx(1)
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
