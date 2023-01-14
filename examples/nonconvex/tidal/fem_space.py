from dolfin import *
from dolfin_adjoint import *

from domain import *
from rectangle_domain import RectangularDomain

depth = Constant(50.0)
viscosity = Constant(5.0)
C_t = Constant(0.6)
A_t = Constant(314.15)
friction = Constant(0.0025)

V_h = VectorElement("CG", mesh.ufl_cell(), 2)
Q_h = FiniteElement("CG", mesh.ufl_cell(), 1)
W = FunctionSpace(mesh, V_h*Q_h)

initial_condition = Constant(("1e-7", "0.0", "0.0"))

def steady_sw(control):

    # Get domain measures
    # Define test functions
    v, q = TestFunctions(W)
    state = Function(W)

    ic = project(initial_condition, W)
    state.assign(ic, annotate=False)

    u_mid, h_mid = split(state)

    H = h_mid + depth

    # The normal direction
    n = FacetNormal(self.mesh)

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
    R_mid = friction / H * norm_u_mid * inner(u_mid, v) * dx(mesh)
    R_mid += (control * 0.5 * A_t * C_t / H) * u_mag * inner(u_mid, v) * farm.site_dx

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

    solve(F == 0, state, bcs = strong_bcs)

    return state
