import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Type enforcing:
from typing import Union, Callable


# Function taking a single step in time for a reaction-diffusion equation:
def one_dim_reaction_diffusion_step(u: np.ndarray, I_minus_Lap: Callable, I_plus_Lap: sp.spmatrix, f: Callable,
                                    k: float):
    """
    [Currently one dimensional!]
    Function returning the solution to (I_minus_Lap)*u_star = (I_plus_Lap)*u + k*f(u).
    :param u: Current vector-represented function before a step is taken. Represents the M points from 0 to 1.
    :param I_minus_Lap: Left-side matrix, in LU-factorized callable format.
    :param I_plus_Lap: Right-side matrix, in sparse format.
    :param f: Reaction/Source term.
    :param mu: Diffusion coefficient.
    :param h: Step size in space.
    :param k: Step size in time.
    :return: Next step u_star.
    """
    M = len(u)

    # Create the initial source vector:
    f_vec = np.zeros(M, dtype='float64')
    f_vec[1:-1] = np.array([f(u_i) for u_i in u[1:-1]])

    right_side_vector = I_plus_Lap.dot(u) + f_vec

    # Solves the linear system Ax = b, by calling I_minus_Lap(right_side_vector)
    u_star = I_minus_Lap(right_side_vector)

    # Create the intermediate source vector:
    f_vec_star = np.zeros(M, dtype='float64')
    f_vec_star[1:-1] = np.array([f(u_i) for u_i in u_star[1:-1]])

    u_next = u_star + (k/2.0)*(f_vec_star - f_vec)
    return u_next


# Function to generate a discretized matrix approximation of the 2D-Laplacian:
# With Dirichlet conditions and Natural ordering.
def two_dim_laplacian(n: int):
    pass


# Function to generate a discretized matrix approximation of the 1D-Laplacian:
# With Dirichlet conditions and Natural ordering.
def one_dim_sparse_laplacian(n):
    return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], dtype='float64', shape=(n, n), format='lil')


# Plan on making a reaction-diffusion solver, which can work for 1D- and 2D-problems.
def one_dim_reaction_diffusion_solver(u_init: np.ndarray, mu: float, f: Callable,
                                              N: int, T: float = 1.0, M: Union[int, None] = None, X: float = 1.0):
    """
    Solving on the domain [0, 1]
    :param u_init: Initial vector-values for u(x, 0). u_init[0] and u_init[-1] used as boundary values.
    :param M: Number of spatial discretization points.
    :param N: Number of temporal discretization points.
    :param X; End point in space for solution. Assumed start at x=0.
    :param T: End time for solution.
    :param twoD:
    :return: (N, M)-np.ndarray storing the solution at every time- and space-point.
    """

    if M is None:
        M = len(u_init)

    k = (T - 0.0)/N
    h = (X - 0.0)/M

    r = mu*k/(h*h)

    # Constructing the M by M Discrete laplacian matrix with zeros on the first and last rows.
    Lap = one_dim_sparse_laplacian(M)
    Lap[0, [0, 1]] = [0.0, 0.0]
    Lap[M-1, [M-2, M-1]] = [0.0, 0.0]

    r_half_Lap = (r/2.0)*Lap

    # Find the LU-factorized version of the left-side matrix in Crank-Nicolson. Returns a callable:
    I_minus_r_Lap = spla.factorized((sp.identity(M, dtype='float64', format='lil') - r_half_Lap).tocsc())
    I_plus_r_Lap = sp.identity(M, dtype='float64', format='lil') + r_half_Lap

    u_storage = np.zeros((M, N), dtype='float64')
    u_storage[0, :] = np.copy(u_init)

    for i in range(1, N):
        u_init = one_dim_reaction_diffusion_step(u_init, I_minus_r_Lap, I_plus_r_Lap, f, k)
        u_storage[i, :] = np.copy(u_init)

    return u_storage


def test_one_dim_diffusion_reaction():
    M, N = 100, 100
    mu = 0.01

    u_init = 1.0*np.ones(M)
    u_init[-1] = 5.0
    f = lambda x: -1.5*(x - 0.5)

    solution_array = one_dim_reaction_diffusion_solver(u_init, mu, f, N, M)
    # print(f"Solution array:\n{solution_array}")

    fig, axis = plt.subplots(1, 1)
    xs = np.linspace(0.0, 1.0, M)

    axis.plot(xs, solution_array[10, :])
    axis.set_ylim(0.0, 5.2)
    plt.show()


if __name__ == '__main__':
    test_one_dim_diffusion_reaction()
