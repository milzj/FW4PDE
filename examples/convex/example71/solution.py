import fenics
import matplotlib.pyplot as plt

def example71_solution(n=100):
    """Implements solution of Example 7.1 in Wachsmuth and Wachsmuth (2011)
    as fenics expression and function.

    References:
    -----------
    G. Wachsmuth and D. Wachsmuth: Convergence and regularization results for
    optimal control problems with sparsity functional, ESAIM: COCV 17 (2011) 858-886,
    https://doi.org/10.1051/cocv/2010027

    Parameters:
    ----------

    n: Int, optional
        number of cells

    Returns:
    --------

    solution_expr : fenics.Expression
        Solution as fenics expression

    solution_inter : fenics.Function (DG0 with n cells)
        Solution as DG0 function with n cells

    """

    a = -30.0
    b = 30.0

    solution_str = "(x[0] > 1.0/12 && x[0] < 5.0/12) ? a : " +\
            "(x[0] > 7.0/12 && x[0] < 11.0/12) ? b : 0.0"

    solution_expr = fenics.Expression(solution_str, a=a, b=b, degree=0)

    mesh = fenics.UnitIntervalMesh(n)
    U = fenics.FunctionSpace(mesh, "DG", 0)

    solution_inter = fenics.Function(U)
    solution_inter.interpolate(solution_expr)

    return solution_expr, solution_inter


if __name__ == "__main__":
    "Plot and save solution."

    import os

    dir = "plots"
    if not os.path.exists(dir):
        os.makedirs(dir)

    _, solution = example71_solution()

    plt.figure(figsize=(5,5))
    fenics.plot(solution)
    plt.gca().set_aspect(1.0/30)
    plt.tight_layout()
    plt.savefig(dir + "/" + "solution.pdf")
    plt.savefig(dir + "/" + "solution.png")
