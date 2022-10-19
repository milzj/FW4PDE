from dolfin import *
from dolfin_adjoint import *

from log_normal_field import LogNormalField

class RandomPoissonProblem(object):

    def __init__(self, n, num_addends, len_scale):

        self.n = n
        self.beta = 0.0025

        self.lb = Constant(-1.)
        self.ub = Constant(1.)

        mesh = UnitSquareMesh(MPI.comm_self, n, n)

        V = FunctionSpace(mesh, "CG", 1)
        U = FunctionSpace(mesh, "DG", 0)

        self.V = V
        self.U = U

        # desired state taken from https://epubs.siam.org/doi/10.1137/S1052623498343131
        yd = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])*exp(2*x[0])/6.0", degree = 1)
        self.yd = yd

        self.bcs = DirichletBC(self.V, 0.0, "on_boundary")

        self.y_trial = TrialFunction(V)
        self.v_test = TestFunction(V)

        self.y = Function(V)
        self._kappa = Function(V)

        self.kappa = LogNormalField(V, num_addends=num_addends, len_scale = len_scale)

    @property
    def control_space(self):
            return self.U

    @property
    def control(self):
            return Function(self.U)

    def state(self, y, u, samples):

        kappa = self.kappa.sample(samples)
        _kappa = self._kappa
        _kappa.interpolate(kappa)
        bcs = self.bcs
        v_test = self.v_test
        y_trial = self.y_trial

        a = inner(_kappa*grad(y_trial), grad(v_test))*dx
        L = u*v_test*dx

        A, b = assemble_system(a, L, bcs)

        solver = LUSolver(A, "petsc")
        solver.solve(y.vector(), b)

    def __call__(self, u, sample):

        y = self.y
        y.vector().zero()

        yd = self.yd

        self.state(y, u, sample)

        return assemble(0.5*(y-yd)**2*dx)



if __name__ == "__main__":

    import numpy as np

    n = 8
    num_addends = 5
    len_scale = 0.1
    poisson_problem = RandomPoissonProblem(n, num_addends, len_scale)

    U = poisson_problem.control_space
    u = Function(U)
    u.interpolate(Expression("sin(pi*x[0]*x[1])", degree=0))

    np.random.seed(1234)
    sample = np.random.randn((2*num_addends)**2)
    print(poisson_problem(u, sample))
