# FW4PDE: Frank--Wolfe algorithms for PDE-constrained optimization

The package implements conditional gradient methods for the solution 
of the PDE-constrained optimization problems

$$
	\min_{u \in U_{\text{ad}}}  J(S(u)) + \beta \\|u\\|_{L^1(D)},
$$

where $\beta \geq 0$, $S(u)$ is the solution to a potentially nonlinear PDE, and 
$U_{\text{ad}} = \\{ u \in L^2(D) : a \leq u \leq b \\}$. Here $a$, $b \in L^2(D)$
with $a \leq b$.

## Examples

Examples of convex PDE-constrained problems can be found in [convex](examples/convex)
and of potentially nonconvex ones in [nonconvex](examples/nonconvex).

## Dependencies

The following packages are required:

- [FEniCS](https://fenicsproject.org/)
- [dolfin-adjoint](http://www.dolfin-adjoint.org/)
- [moola](https://github.com/funsim/moola)

See [environment.yml](environment.yml) for a complete list of dependencies.

## References

* M. Besançon, A. Carderera, S. Pokutta (2022) [FrankWolfe.jl: A High-Performance and Flexible Toolbox for Frank–Wolfe Algorithms and Conditional Gradients](https://doi.org/10.1287/ijoc.2022.1191). INFORMS Journal on Computing 0(0). 

* Dunn, J.C.: [Rates of convergence for conditional gradient algorithms near singular and nonsingular extremals](https://doi.org/10.1137/0317015). SIAM J. Control Optim. 17(2), 187–211 (1979)

* Dunn, J.C.: [Convergence rates for conditional gradient sequences generated by implicit step length rules](https://doi.org/10.1137/0318035). SIAM J. Control Optim. 18(5), 473–487 (1980)

* J.C. Dunn, S. Harshbarger, [Conditional gradient algorithms with open loop step size rules](https://doi.org/10.1016/0022-247X(78)90137-3), Journal of Mathematical Analysis and Applications, Volume 62, Issue 2, February 1978, Pages 432-444

* Harchaoui, Z., Juditsky, A. and Nemirovski, A. [Conditional gradient algorithms for norm-regularized smooth convex optimization](https://doi.org/10.1007/s10107-014-0778-9). Math. Program. 152, 75–112 (2015). 

* K. Kunisch, D. Walter, On fast convergence rates for generalized conditional gradient methods with backtracking stepsize, preprint, https://arxiv.org/abs/2109.15217, 2021

A complete list of references is provided in [lib.md](misc/lit.md).
