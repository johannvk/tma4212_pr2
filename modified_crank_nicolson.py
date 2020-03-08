import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import linregress

# Type enforcing:
from typing import Union, Callable, Tuple


# Function to generate a discretized matrix approximation of the 2D-Laplacian:
# With Dirichlet conditions and Natural ordering.
def two_dim_laplacian(n: int):
    pass


# Function to generate a discretized matrix approximation of the 1D-Laplacian:
# With Dirichlet conditions and Natural ordering.
def one_dim_sparse_laplacian(n):
    return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], dtype='float64', shape=(n, n), format='lil')


# Function taking a single step in time for a reaction-diffusion equation:
def one_dim_reaction_diffusion_step(u: np.ndarray, xs, I_minus_Lap: Callable, I_plus_Lap: sp.spmatrix, f: Callable,
                                    i: int, k: float):
    """
    Function returning a time step for u^(n). First find solution u_star to
    (I_minus_Lap)*u_star = (I_plus_Lap)*u + k*f(u). Then return u_star + (k/2.0)*(f(xs, u_star) - f(xs, u)).
    :param u: Current vector-represented function, u^(n), before a step is taken.
    :param I_minus_Lap: Left-side matrix, in LU-factorized callable format.
    :param I_plus_Lap: Right-side matrix, in sparse format.
    :param f: Reaction/Source term.
    :param i: Which time step we are on.
    :param k: Step size in time.
    :return: Next step, u^(n+1).
    """
    M = len(u)
    t = i*k

    # Create the initial source vector:
    f_vec = np.zeros(M, dtype='float64')
    f_vec[1:-1] = np.array([f(xs[j], t, u[j]) for j in range(1, len(u)-1)])

    right_side_vector = I_plus_Lap.dot(u) + k*f_vec

    # Solves the linear system Ax = b, by calling I_minus_Lap(right_side_vector)
    u_star = I_minus_Lap(right_side_vector)

    # Create the intermediate source vector:
    f_vec_star = np.zeros(M, dtype='float64')
    f_vec_star[1:-1] = np.array([f(xs[j], t+k, u_star[j]) for j in range(1, len(u)-1)])

    return u_star + (k/2.0)*(f_vec_star - f_vec)


# Plan on making a reaction-diffusion solver, which can work for 1D- and 2D-problems.
def one_dim_reaction_diffusion_solver(u_init: np.ndarray, xs: np.ndarray, mu: float, f: Callable, N: int,
                                      T: float = 1.0, M: Union[int, None] = None,
                                      X: Union[float, Tuple[float, float]] = 1.0):
    """
    Solving on the domain [0, 1]
    :param u_init: Initial vector-values for u(x, 0). u_init[0] and u_init[-1] used as boundary values.
    :param M: Number of spatial discretization points.
    :param N: Number of temporal discretization points.
    :param X; End point in space for solution. Assumed start at x=0.
    :param T: End time for solution.
    :return: (N, M)-np.ndarray storing the solution at every time- and space-point.
    """

    if M is None:
        M = len(u_init)

    if isinstance(X, float):
        h = (X - 0.0)/(M-1)
    else:
        h = (X[1] - X[0])/(M-1)

    k = (T - 0.0)/(N-1)

    r = mu*k/(h*h)

    # Constructing the M by M Discrete Laplacian matrix with zeros in the first and last row.
    Lap = one_dim_sparse_laplacian(M)
    Lap[0, [0, 1]] = [0.0, 0.0]
    Lap[M-1, [M-2, M-1]] = [0.0, 0.0]

    r_half_Lap = (r/2.0)*Lap

    # Find the LU-factorized version of the left-side matrix in Crank-Nicolson. Returns a callable:
    I_minus_r_Lap = spla.factorized((sp.identity(M, dtype='float64', format='lil') - r_half_Lap).tocsc())
    I_plus_r_Lap = sp.identity(M, dtype='float64', format='lil') + r_half_Lap

    u_storage = np.zeros((N, M), dtype='float64')
    u_storage[0, :] = np.copy(u_init)

    for i in range(1, N):
        u_init = one_dim_reaction_diffusion_step(u_init, xs, I_minus_r_Lap, I_plus_r_Lap, f, i, k)
        u_storage[i, :] = np.copy(u_init)

    return u_storage


def one_dim_diffusion_reaction_convergence():
    M = 100
    X, T = (0.0, 1.0), 0.1

    xs = np.linspace(X[0], X[1], M)

    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.sin(np.pi*x) - t*x*(1.0 - x)

    exact_solution_T = u_exact(xs, T)

    u_init = np.sin(np.pi * xs)
    mu = 1.0
    f = lambda x, t, u: 1.0*(-x * (1.0 - x) - 2*t)

    Ns = np.logspace(1.0, 2.2, 10, dtype=int)
    sup_errors = np.zeros(len(Ns))
    for i, N in enumerate(Ns):
        solution_array = one_dim_reaction_diffusion_solver(u_init, xs, mu, f, N=N, T=T, M=M, X=X)
        sup_errors[i] = np.max(np.abs(exact_solution_T - solution_array[-1, :]))

    Ns -= 1  # To get the precise value of 'h' when taking the reciprocal.
    log_log_slope = linregress(np.log10(1/Ns), np.log10(sup_errors))[0]

    fig, ax = plt.subplots(1, 1)

    ax.plot(np.log10(1/Ns), np.log10(sup_errors), label=f"Sup-error(N): Slope {log_log_slope:.3f}")
    ax.set_xlabel("log(h = 1/N)")
    ax.set_ylabel("log(Sup|u_ex - u_num|)")
    ax.legend(loc="best")

    plt.show()


def test_one_dim_diffusion_reaction():
    M, N = 200, 200
    X, T = (0.0, 1.0), 0.5

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


if __name__ == '__main__':
    # test_one_dim_diffusion_reaction()
    one_dim_diffusion_reaction_convergence()
