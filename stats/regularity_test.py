import numpy as np
import fenics
import matplotlib.pyplot as plt

def regularity_test(gradient, beta, logspace_start=-11, logspace_stop=0, 
        figure_name="regularity_test", ndrop=1):
    """Verify growth condition/regularity condition of solution

    Computationally checks whether the Lebesgue measure of
    {x in D : |gradient(x)|-beta| <= eps} decays with a rate
    Ceps^c, where c and C are constants.

    The function estimates the constants c and C using least
    squares and creates a plot with the convergence rates and
    the least squares fit. 

    The Lebesgue measures are computed for a number of values
    of eps until the Lebesgue measure is zero.

    Parameters:
    -----------
    gradient : fenics.Function
        Gradient of smooth objective evaluated at solution
    beta : float64
        Sparsity parameter
    logspace_start, logspace_stop : int, optional
        
    figure_name : str, optional
        name of figure
    ndrop : int, optional
        number of data points to be excluded from least squares fit

    """
    U = gradient.function_space()

    w = fenics.Function(U)
    w_vec = -np.ones(U.dim())
    w_idx = -np.ones(U.dim())

    num = logspace_stop-logspace_start+1
    Epsilons = np.logspace(logspace_start, logspace_stop, num=num)
    measures =  []
    epsilons = []
    gradient_vec = gradient.vector()[:]

    # Compute measures
    num = 0
    for epsilon in Epsilons[::-1]:

        w_vec[:] = np.abs(gradient_vec) - beta
        w_idx[:] = 1.0*(np.abs(w_vec) <= epsilon)
        w.vector()[:] = w_idx

        measure = fenics.assemble(w*fenics.dx)

        if measure == 0:
            break

        measures.append(measure)
        epsilons.append(epsilon)
        num += 1


    # Least squares fit
    X = np.ones((num-ndrop, 2)); X[:, 1] = np.log(epsilons[ndrop::])
    x, residudals, rank, s = np.linalg.lstsq(X, np.log(measures[ndrop::]), rcond=None)

    rate = x[1]
    constant = np.exp(x[0])

    # Plotting
    base = 10
    xlabel = r"$\epsilon$"
    ylabel = r"$\mathrm{Meas}(||\nabla F(u^*)|-\beta| \leq \epsilon)$"

    fig, ax = plt.subplots()
    ax.plot(epsilons, measures, marker="o", linestyle="-.", color="black")

    X = epsilons
    Y = constant*X**rate
    Y = Y*Y[0] # -> log(Y) = log(Y) + log(Y[1])

    lsqs_label = r"${}\cdot {}^{}$".format(constant, base, "{"+ str(rate)+"}")
    ax.plot(X, Y, color="black", linestyle="--", label=lsqs_label)

    ax.set_xscale("log", base=base)
    ax.set_yscale("log", base=base)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.legend()

    plt.tight_layout()
    plt.savefig(figure_name + ".pdf")
    plt.savefig(figure_name + ".png")
    plt.close()

    return True


if __name__ == "__main__":

    import numpy as np
    import fenics

    n = 64
    beta = 1e-2
    seed = 1234

    mesh = fenics.UnitSquareMesh(n,n)
    U = fenics.FunctionSpace(mesh, "DG", 0)

    gradient = fenics.Function(U)
    np.random.seed(seed)

    gradient.vector()[:] = beta + beta*np.random.randn(U.dim())

    assert regularity_test(gradient, beta, ndrop=2)
