import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import linregress

# Type enforcing:
from typing import Union, Callable, Tuple


# Function to generate a discretized matrix approximation of the 1D-Laplacian:
# With Dirichlet conditions and Natural ordering.
def one_dim_sparse_laplacian(m: int):
    return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], dtype='float64', shape=(m, m), format='lil')


def one_dim_generate_step_matrices(m: int, r: float, Neumann_BC: Union[Tuple[float, float], None]):
    # Discretized Laplacian in one dimension:
    Lap = one_dim_sparse_laplacian(m)

    if Neumann_BC is None:
        # Adapting the first and last row to Neumann BC:
        Lap[0, [0, 1]] = [0.0, 0.0]
        Lap[m - 1, [m - 2, m - 1]] = [0.0, 0.0]
    else:
        # Adapting the first and last row to Neumann BC:
        Lap[0, 1] = 2.0
        Lap[m - 1, m - 2] = 2.0

    r_half_Lap = (r / 2.0) * Lap

    # Find the LU-factorized version of the left-side matrix in Crank-Nicolson. Returns a callable:
    I_minus_r_Lap = spla.factorized((sp.identity(m, dtype='float64', format='lil') - r_half_Lap).tocsc())
    I_plus_r_Lap = (sp.identity(m, dtype='float64', format='lil') + r_half_Lap).tocsc()  # Turn into csc!

    return I_minus_r_Lap, I_plus_r_Lap


# Function taking a single step in time for a reaction-diffusion equation:
def one_dim_reaction_diffusion_step(u: np.ndarray, xs, I_minus_Lap: Callable, I_plus_Lap: sp.spmatrix, f: Callable,
                                    i: int, k: float, h: float, mu: float, Neumann_BC: Union[Tuple[float, float], None] = None):
    """
    Function returning a time step for u^(n). First find solution u_star to
    (I_minus_Lap)*u_star = (I_plus_Lap)*u + k*f(u). Then return u_star + (k/2.0)*(f(xs, u_star) - f(xs, u)).
    :param u: Current vector-represented function, u^(n), before a step is taken.
    :param I_minus_Lap: Left-side matrix, in LU-factorized callable format.
    :param I_plus_Lap: Right-side matrix, in sparse format.
    :param f: Reaction/Source term.
    :param i: Which time step we are on.
    :param k: Step size in time.
    :param h: Step size in space.
    :param mu: Diffusion constant.
    :return: Next step, u^(n+1).
    """
    M = len(u)
    t = i*k

    # Create the initial source vector:
    f_vec = np.array([f(xs[j], t, u[j]) for j in range(M)], dtype='float64')

    if Neumann_BC is None:
        f_vec[0] = 0.0
        f_vec[-1] = 0.0
        right_side_vector = I_plus_Lap.dot(u) + k * f_vec
    else:
        r_h = mu*k/h
        right_side_vector = I_plus_Lap.dot(u) + k * f_vec
        right_side_vector[0] -= 2*r_h*Neumann_BC[0]
        right_side_vector[-1] += 2*r_h*Neumann_BC[1]

    # Solves the linear system Ax = b, by calling I_minus_Lap(right_side_vector)
    u_star = I_minus_Lap(right_side_vector)

    # Create the intermediate source vector:
    f_vec_star = np.array([f(xs[j], t+k, u_star[j]) for j in range(M)], dtype='float64')

    if Neumann_BC is None:
        f_vec_star[0] = 0.0
        f_vec_star[-1] = 0.0

    return u_star + (k/2.0)*(f_vec_star - f_vec)


# Plan on making a reaction-diffusion solver, which can work for 1D- and 2D-problems.
def one_dim_reaction_diffusion_solver(u_init: np.ndarray, xs: np.ndarray, mu: float, f: Callable, N: int,
                                      T: float = 1.0, M: Union[int, None] = None,
                                      X: Union[float, Tuple[float, float]] = 1.0,
                                      Neumann_BC: Union[Tuple[float, float], None] = None):
    """
    Solving on the domain x in (0, X) or x in (X[0], X[1])
    :param u_init: Initial vector-values for u(x, 0). u_init[0] and u_init[-1] used as Dirichlet boundary values.
    :param xs: Discretization of solution domain.
    :param mu: Diffusion coefficient.
    :param f: Source term. f(x, t, u).
    :param N: Number of temporal discretization points.
    :param T: End time for solution.
    :param M: Number of spatial discretization points.
    :param X: End point in space for solution. Assumed start at x=0 unless otherwise specified.
    :param Neumann_BC: Variable indicating whether or not to use Neumann boundary conditions.
    :return: (N, M)-np.ndarray storing the solution at every time- and space-point.
    """

    # Finding Spatial specifications:
    if M is None:
        M = len(u_init)

    if isinstance(X, float):
        h = (X - 0.0)/(M-1)
    else:
        h = (X[1] - X[0])/(M-1)

    k = (T - 0.0)/(N-1)
    r = mu*k/(h*h)

    I_minus_Lap, I_plus_Lap = one_dim_generate_step_matrices(M, r, Neumann_BC)

    u_storage = np.zeros((N, M), dtype='float64')
    u_storage[0, :] = np.copy(u_init)

    for i in range(1, N):
        # Subtract 1 from i to start from time-step 0 (t_0), instead of time-step 1 (t_1).
        u_init = one_dim_reaction_diffusion_step(u_init, xs, I_minus_Lap, I_plus_Lap, f, i-1, k, h, mu, Neumann_BC)
        u_storage[i, :] = np.copy(u_init)

    return u_storage


def one_dim_diffusion_reaction_convergence(u_exact: Callable, mu: float, f: Callable,
                                           Neumann_BC : Union[None, Tuple[float, float]] = None,
                                           M: int = 200, X : Tuple[float, float] = (0.0, 1.0), T: float = 1.0):
    xs = np.linspace(X[0], X[1], M)
    exact_solution_T = u_exact(xs, T)

    u_init = u_exact(xs, 0.0)

    Ns = np.logspace(1.0, 2.2, 10, dtype=int)
    sup_errors = np.zeros(len(Ns))

    for i, N in enumerate(Ns):
        solution_array = one_dim_reaction_diffusion_solver(u_init, xs, mu, f, N=N, T=T, M=M, X=X, Neumann_BC=Neumann_BC)
        sup_errors[i] = np.max(np.abs(exact_solution_T - solution_array[-1, :]))

    print(f"Sup-errors: {sup_errors}\n")
    Ns -= 1  # To get the precise value of 'h' when taking the reciprocal.
    log_log_slope = linregress(np.log10(1/Ns), np.log10(sup_errors))[0]

    fig, ax = plt.subplots(1, 1)

    ax.plot(np.log10(1/Ns), np.log10(sup_errors), label=f"Sup-error(N): Slope {log_log_slope:.3f}")
    ax.set_xlabel("log(h = 1/N)")
    ax.set_ylabel("log(Sup|u_ex - u_num|)")
    ax.legend(loc="best")
    plt.show()


def one_dim_dirichlet_convergence():
    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.sin(np.pi*x) - t*x*(1.0 - x)
    mu = 1.0
    f = lambda x, t, u: 1.0*(-x * (1.0 - x) - 2*t)

    one_dim_diffusion_reaction_convergence(u_exact=u_exact, mu=mu, f=f)


def test_one_dim_dirichlet():
    M, N = 200, 200
    X, T = (0.0, 1.0), 5.0

    xs = np.linspace(X[0], X[1], M)

    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.sin(np.pi*x) - t*x*(1.0 - x)

    exact_solution_T = u_exact(xs, T)

    u_init = u_exact(xs, 0.0)
    mu = 1.0
    f = lambda x, t, u: - 1.0 * x * (1.0 - x) - 2*t

    solution_array = one_dim_reaction_diffusion_solver(u_init, xs, mu, f, N=N, T=T, M=M, X=X)

    sup_error = np.max(np.abs(exact_solution_T - solution_array[-1, :]))
    print(f"Sup error is: {sup_error:.3e}")

    fig, axis = plt.subplots(1, 1)

    axis.plot(xs, solution_array[-1, :], label="U_num")
    axis.plot(xs, exact_solution_T, label="U_exact")

    axis.legend(loc="best")
    axis.set_xlim(X)
    plt.show()


def one_dim_neumann_convergence():
    def u_exact(x, t):
        return t**3 + x*(1-x)

    def f(x, t, u): return 3*t**2 + 1.0
    mu = 0.5

    one_dim_diffusion_reaction_convergence(u_exact, mu, f, Neumann_BC=(1.0, -1.0))


def test_one_dim_neumann():

    def u_exact(x, t):
        return t**3 + x*(1-x)

    def f(x, t, u): return 3*t**2 + 1.0

    M, N = 200, 25
    X, T = (0.0, 1.0), 1.0

    mu = 0.5
    xs = np.linspace(X[0], X[1], M)

    u_init = u_exact(xs, 0.0)
    exact_solution_T = u_exact(xs, T)
    neumann_bc = (1.0, -1.0)

    solution_array = one_dim_reaction_diffusion_solver(u_init, xs, mu, f, N=N, T=T, M=M, X=X, Neumann_BC=neumann_bc)

    sup_error = np.max(np.abs(exact_solution_T - solution_array[-1, :]))
    print(f"Sup error is: {sup_error:.3e}")

    fig, axis = plt.subplots(1, 1)

    axis.plot(xs, solution_array[-1, :], label="U_num", alpha=0.8)
    axis.plot(xs, exact_solution_T, label="U_exact", alpha=0.8)

    axis.legend(loc="best")
    axis.set_xlim(X)
    plt.show()


if __name__ == '__main__':
    # one_dim_dirichlet_convergence()
    # test_one_dim_dirichlet()

    one_dim_neumann_convergence()
    test_one_dim_neumann()
