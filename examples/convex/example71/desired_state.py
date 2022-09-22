import fenics
import matplotlib.pyplot as plt

def example71_desired_state(n=100):
	"""Implements desired state of Example 7.1 in Ref. [1].
	as fenics expression and function.

	Replaced +4pi^2sin(2pi x) by -4pi^2sin(2pi x) in yd computation (see p. 877 in Ref. [1]).

	References:
	-----------
	[1] G. Wachsmuth and D. Wachsmuth: Convergence and regularization results for
	optimal control problems with sparsity functional, ESAIM: COCV 17 (2011) 858-886,
	https://doi.org/10.1051/cocv/2010027

	Parameters:
	----------

	n: Int, optional
		number of cells

	Returns:
	--------

	yd_expr : fenics.Expression
		Solution as fenics expression

	yd_inter : fenics.Function (DG0 with n cells)
		Solution as DG0 function with n cells

	"""

	yd_str1 = "-5.0*x-4.0*pi*pi*sin(2.0*pi*x)"

	yd_str = "(x <= 1.0/12) ? 0.0 + yy : " +\
			"(x > 1.0/12 && x <= 5.0/12) ? 5.0/48 - 5.0/2*x + 15.0*x*x + yy :" +\
			"(x > 5.0/12 && x <= 7.0/12) ? -5.0/2 + 10.0*x + yy :" +\
			"(x > 7.0/12 && x <= 11.0/12) ? -365.0/48 + 55.0/2*x - 15.0*x*x + yy : 5.0 + yy"


	yd_str = yd_str.replace("yy", yd_str1)
	yd_str = yd_str.replace("x", "x[0]")


	yd_expr = fenics.Expression(yd_str, degree=0)

	mesh = fenics.UnitIntervalMesh(n)
	U = fenics.FunctionSpace(mesh, "DG", 0)

	yd_inter = fenics.Function(U)
	yd_inter.interpolate(yd_expr)

	return yd_str, yd_expr, yd_inter


if __name__ == "__main__":
	"Plot and save desired state."

	import os

	dir = "plots"
	if not os.path.exists(dir):
		os.makedirs(dir)

	_, _, desired_state = example71_desired_state(n=1000)

	plt.figure(figsize=(5,5))
	fenics.plot(desired_state)
	plt.gca().set_aspect(1.0/40)
	plt.tight_layout()
	plt.savefig(dir + "/" + "desired_state.pdf")
	plt.savefig(dir + "/" + "desired_state.png")
