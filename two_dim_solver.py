# Numerical Libraries:
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Graphical Libraries:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm

# Type enforcing:
from typing import Union, Callable, Tuple, Iterable


def two_dim_laplace_neumann(M: int, format: str = 'coo', dtype: str = 'float64'):
    """
    Building the discretized Laplacian in two dimensions with block matrix sparse tools.
    Built in Neumann boundary conditions. Need compensation for this in the solver-step.
    :param M: Number of spatial discretization points in each spatial dimension.
    :param format: Storage format for the sparse matrices.
    :param dtype: Data type in the sparse matrices.
    :return: Discretized two dimensional Laplacian, with Neumann boundary conditions.
    """
    inner_data = [[1.0] * (M - 2) + [2.0], -4.0, [2.0] + [1.0] * (M - 2)]
    inner_diag = sp.diags(inner_data, offsets=[-1, 0, 1], format=format, dtype=dtype)
    I_m = sp.identity(M, format=format, dtype=dtype)

    # Rows of matrices:
    # Initializing with the top row:
    rows = [[inner_diag, 2*I_m] + (M - 2) * [None]]

    # Adding the middle rows (1, m-2):
    for i in range(1, M - 1):
        row = [None] * M
        row[i-1:i+2] = I_m, inner_diag, I_m
        rows.append(row)

    # Adding the bottom row:
    rows.append((M - 2) * [None] + [2 * I_m, inner_diag])
    return sp.bmat(rows, format=format, dtype=dtype)


class DiffusionReactionSolver2D:

    def __init__(self, u_init: np.ndarray, domain: Tuple[np.ndarray, np.ndarray], mu: float, f: Callable, N: int,
                 T: float = 1.0, Neumann_BC = None, store_result: bool = True):
        """
        Initializer function for
        :param u_init: Initial Values for the distribution, in (M, M)-np.ndarray.
        :param domain:
        :param mu:
        :param f:
        :param N:
        :param T:
        :param Neumann_BC:
        :param M: Number of spatial discretization points in each spatial dimension.
        :param r: Composite diffusion- and steps-size parameter.
        :return:
        """
        # Need the step sizes in both spatial dimensions to be equal.
        assert np.max(domain[0]) == np.max(domain[1])
        assert np.min(domain[0]) == np.min(domain[1])

        # Need a square domain, with equal number of points in each spatial dimension.
        assert u_init.shape[0] == u_init.shape[1]

        if Neumann_BC is None:  # No boundary conditions supplied, assumes zero derivatives at the boundaries.
            self.Neumann_BC = [lambda *args: 0.0] * 4
        else:
            self.Neumann_BC = [np.vectorize(func) for func in Neumann_BC]

        # Storing domain parameters:
        self.M = u_init.shape[0]
        self.N = N
        self.T = T
        self.X, self.Y = domain

        # Storing the initial state of the function, and making storage for the steps taken.
        self.u_init = u_init
        self.u_storage = np.zeros((self.N, self.M, self.M), dtype='float64')

        # Step size in space and time respectively:
        self.h = (np.max(domain[0]) - np.min(domain[0])) / (self.M - 1)
        self.k = (self.T - 0.0) / (self.N - 1)

        # Retaining the Reaction function and Diffusion coefficient within the Class:
        self.f = np.vectorize(f)
        self.mu = mu
        self.mu_k_h = (self.mu, self.k, self.h)  # Can be removed later.

        # Getting the composite diffusion/step-size parameter r:
        self.r = self.mu * self.k / (self.h**2)

        # Generatimg the Left-hand and Right-hand side matrices for doing implicit steps:
        self.I_minus_Lap, self.I_plus_Lap = self.generate_two_dim_step_matrices()

        # Setting the boundary boolean arrays:
        self.boundaries = [np.full((self.M, self.M), False, dtype=bool), np.full((self.M, self.M), False, dtype=bool),
                           np.full((self.M, self.M), False, dtype=bool), np.full((self.M, self.M), False, dtype=bool)]
        self.boundaries[0][1:self.M - 1, self.M - 1] = True  # Eastern boundary.
        self.boundaries[1][self.M - 1, 1:self.M - 1] = True  # Northern boundary.
        self.boundaries[2][1:self.M - 1, 0] = True  # Western boundary.
        self.boundaries[3][0, 1:self.M - 1] = True  # Southern boundary.

    def generate_two_dim_step_matrices(self):
        """
        Generates the required step matrices for doing a single step in the diffusion-reaction solver.
        Assumes Neumann boundary conditions.
        :return: I_minus_Lap: Callable, LU-factorized Implicit-solver matrix.
                 I_plus_Lap: Sparse matrix, for generating right side vector.
        """
        Lap_h = two_dim_laplace_neumann(self.M, format='csc')
        I_m = sp.identity(self.M * self.M, dtype='float64', format='csc')

        I_minus_Lap = spla.factorized(I_m - (self.r / 2.0) * Lap_h)
        I_plus_Lap = I_m + (self.r / 2.0) * Lap_h

        return I_minus_Lap, I_plus_Lap

    def generate_right_side_vector(self, u_n: np.ndarray, n: int):
        """
        Generating the right-hand-side vector for the Implicit solve in the Diffusion-reaction scheme.
        :param U_n: Current solution at timestep n, an (M*M,)-np.ndarray.
        :param M: Number of spatial discretization points in each spatial dimension.
        :param n: Current time step. From 0 to N-1.
        :param X: (M, M)-np.ndarray storing the X-values for the domain in a meshgrid-format.
        :param Y: (M, M)-np.ndarray storing the Y-values for the domain in a meshgrid-format.
        :param I_plus_Lap: Right hand side explicit part of the diffusion step.
        :param f: Reaction term. Callable function as a function of (x, y, t, u).
        :param bc_funcs: Boundary condition functions. Ordered {East: 0, North: 1, West: 2, South: 3}.
                         Also have to accept the arguments as (x, y, t, u).
        :param mu_k_h: Tuple storing (mu, k, h). Diffusion coefficient, step size in time and space respectively.
        :return: (M*M,)-np.ndarray Right-hand-side vector used for the Implicit solve.
        """
        #  Reshape u_n to appropriate dimensions:
        u_n = u_n.reshape((self.M, self.M), order='C')
        t_n = n * self.k

        # Boolean masks for boundary indices: East: 0, North: 1, West: 2, South: 3.

        # Initializing the right-hand-side vector:
        f_vec = self.k * self.f(self.X, self.Y, t_n, u_n)
        mult_bc = (2 * self.mu * self.k / self.h)
        for i, boundary in enumerate(self.boundaries):
            f_vec[boundary] += mult_bc * self.Neumann_BC[i](self.X[boundary], self.Y[boundary], t_n, u_n[boundary])

        # WhaT if we treat the corners as simply lying on the x-axis?
        # Handle the corner cases: Not working perfectly. Would like to average out the two nearest points.
        corners = [(self.M - 1, self.M - 1), (0, self.M - 1), (0, 0), (self.M - 1, 0)]
        mult_corner = np.sqrt(2.0) * self.mu * self.k / self.h

        for i, (xi, yi) in enumerate(corners):
            f_vec[xi, yi] += mult_corner * (self.Neumann_BC[i](self.X[xi, yi], self.Y[xi, yi], t_n, u_n[xi, yi]) +
                                            self.Neumann_BC[(i + 1) % 4](self.X[xi, yi], self.Y[xi, yi], t_n, u_n[xi, yi]))

        return self.I_plus_Lap.dot(u_n.ravel(order='C')) + f_vec.ravel(order='C')

    def two_dim_reaction_diffusion_step(self, u_n: np.ndarray, n: int, rhs: np.ndarray):
        t_n = n * self.k
        u_star = self.I_minus_Lap(rhs)

        f_vec = self.f(self.X, self.Y, t_n, u_n.reshape((self.M, self.M), order='C')).ravel(order='C')
        f_vec_star = self.f(self.X, self.Y, t_n + self.k, u_star.reshape(self.M, self.M, order='C')).ravel(order='C')

        return u_star + (self.k / 2.0) * (f_vec_star - f_vec)

    def execute(self):
        self.u_storage[0, :, :] = np.copy(self.u_init)
        u_n = np.copy(self.u_init).ravel(order='C')
        for n in range(0, self.N - 1):
            # Prepare the right hand side vector:
            rhs = self.generate_right_side_vector(u_n, n)
            u_n = self.two_dim_reaction_diffusion_step(u_n, n, rhs)

            self.u_storage[n + 1, :, :] = np.copy(u_n.reshape(self.M, self.M, order='C'))

        return self.u_storage


# def generate_right_side_vector(U_n: np.ndarray, M: int, n: int, X: np.ndarray, Y: np.ndarray, I_plus_Lap: np.ndarray,
#                                f: Callable, bc_funcs: Tuple[Callable, Callable, Callable, Callable],
#                                mu_k_h: Tuple[float, float, float]):
#     """
#     Generating the right-hand-side vector for the Implicit solve in the Diffusion-reaction scheme.
#     :param U_n: Current solution at timestep n, an (M*M,)-np.ndarray.
#     :param M: Number of spatial discretization points in each spatial dimension.
#     :param n: Current time step. From 0 to N-1.
#     :param X: (M, M)-np.ndarray storing the X-values for the domain in a meshgrid-format.
#     :param Y: (M, M)-np.ndarray storing the Y-values for the domain in a meshgrid-format.
#     :param I_plus_Lap: Right hand side explicit part of the diffusion step.
#     :param f: Reaction term. Callable function as a function of (x, y, t, u).
#     :param bc_funcs: Boundary condition functions. Ordered {East: 0, North: 1, West: 2, South: 3}.
#                      Also have to accept the arguments as (x, y, t, u).
#     :param mu_k_h: Tuple storing (mu, k, h). Diffusion coefficient, step size in time and space respectively.
#     :return: (M*M,)-np.ndarray Right-hand-side vector used for the Implicit solve.
#     """
#     #  Think that U_n is passed around as a long ass 1D-array.
#     U_n = U_n.reshape((M, M), order='C')
#     mu, k, h = mu_k_h
#     t_n = n*k
#
#     # Boolean masks for boundary indices: East: 0, North: 1, West: 2, South: 3.
#     boundaries = [np.full((M, M), False, dtype=bool), np.full((M, M), False, dtype=bool),
#                   np.full((M, M), False, dtype=bool), np.full((M, M), False, dtype=bool)]
#
#     boundaries[0][1:M - 1, M - 1] = True  # Eastern boundary.
#     boundaries[1][M - 1, 1:M - 1] = True  # Northern boundary.
#     boundaries[2][1:M - 1, 0] = True  # Western boundary.
#     boundaries[3][0, 1:M - 1] = True  # Southern boundary.
#
#     # Initializing the right-hand-side vector:
#     f_vec = k*f(X, Y, t_n, U_n)
#     mult_bc = (2*mu*k/h)
#     for i, boundary in enumerate(boundaries):
#         f_vec[boundary] += mult_bc*bc_funcs[i](X[boundary], Y[boundary], t_n, U_n[boundary])
#
#     # WhaT if we treat the corners as simply lying on the x-axis?
#     # Handle the corner cases: Not working perfectly. Would like to average out the two nearest points.
#     corners = [(M-1, M-1), (0, M-1), (0, 0), (M-1, 0)]
#     mult_corner = np.sqrt(2.0)*mu*k/h
#     for i, (xi, yi) in enumerate(corners):
#         f_vec[xi, yi] += mult_corner*(bc_funcs[i](X[xi, yi], Y[xi, yi], t_n, U_n[xi, yi]) +
#                                       bc_funcs[(i+1) % 4](X[xi, yi], Y[xi, yi], t_n, U_n[xi, yi]))
#
#     ret_value = I_plus_Lap.dot(U_n.ravel(order='C')) + f_vec.ravel(order='C')
#     return ret_value
#
#
# def two_dim_reaction_diffusion_step(U_n, I_minus_Lap: Callable, rhs: np.ndarray, X, Y,
#                                     f: Callable, time_step: int, k: float, M: int):
#     t_n = time_step*k
#     U_star = I_minus_Lap(rhs)
#
#     f_vec = f(X, Y, t_n, U_n.reshape(M, M, order='C')).ravel(order='C')
#     f_vec_star = f(X, Y, t_n + k, U_star.reshape(M, M, order='C')).ravel(order='C')
#
#     return U_star + (k/2.0)*(f_vec_star - f_vec)
#
#
# def two_dim_reaction_diffusion_solver(u_init: np.ndarray, domain: Tuple[np.ndarray, np.ndarray], mu: float, f: Callable,
#                                       N: int, T: float = 1.0, Neumann_BC=None):
#     """
#
#     :param u_init: Initial Values for the distribution, in (M, M)-np.ndarray.
#     :param domain:
#     :param mu:
#     :param f:
#     :param N:
#     :param T:
#     :param Neumann_BC:
#     :return:
#     """
#     # Need the step sizes in both spatial dimensions to be equal.
#     assert np.max(domain[0]) == np.max(domain[1])
#     assert np.min(domain[0]) == np.min(domain[1])
#
#     # Need a square domain, with equal number of points in each spatial dimension.
#     assert u_init.shape[0] == u_init.shape[1]
#
#     if Neumann_BC is None:  # No boundary conditions supplied, assumes zero derivatives at the boundaries.
#         Neumann_BC = [lambda *args: 0.0] * 4
#
#     M = u_init.shape[0]
#
#     X, Y = domain
#     h = (np.max(domain[0]) - np.min(domain[0]))/(M - 1)
#     k = (T - 0.0) / (N - 1)
#     mu_k_h = (mu, k, h)
#
#     r = mu*k/(h*h)
#     I_minus_Lap, I_plus_Lap = generate_two_dim_step_matrices(M, r)
#
#     u_storage = np.zeros((N, M, M), dtype='float64')
#     u_storage[0, :, :] = np.copy(u_init)
#
#     u_n = np.copy(u_init).ravel(order='C')
#     for n in range(0, N-1):
#
#         # Prepare the right hand side vector:
#         rhs = generate_right_side_vector(u_n, M, n, X, Y, I_plus_Lap, f, Neumann_BC, mu_k_h)
#         u_n = two_dim_reaction_diffusion_step(u_n, I_minus_Lap, rhs, X, Y, f, n, k, M)
#
#         u_storage[n+1, :, :] = np.copy(u_n.reshape(M, M, order='C'))
#
#     return u_storage

# TODO: SETT ALT INN I EN "Solver Class".


# def two_dim_test():
#     # Make a simple test!
#     L, T = 2.0, 2.0
#     M, N = 50, 50
#     mu = 0.5
#
#     def u_exact(x, y, t):
#         return x**2*(x - L)**2*y**2*(y - L)**2 + t**2
#
#     def f(x, y, t, u):
#         kx = x**2*(x-L)**2
#         ky = y**2*(y-L)**2
#
#         Cx = ky*((x - L)**2 + 4*np.sqrt(kx) + x**2)
#         Cy = kx*((y - L)**2 + 4*np.sqrt(ky) + y**2)
#
#         return 2*t - (Cx + Cy)
#
#     def f1(x, y, t, u):
#         return 0.0*x*y*t*u
#
#     X, Y = np.meshgrid(np.linspace(0.0, L, M), np.linspace(0.0, L, M))
#     u_init = u_exact(X, Y, 0.0)
#
#     U_final = two_dim_reaction_diffusion_solver(u_init=u_init, domain=(X, Y), mu=mu, f=f, N=N, T=T)
#
#     u_test = u_exact(X, Y, 0.0)
#     f_test = f(X, Y, 0.0, T)
#
#     fig = plt.figure()
#     fig.suptitle("Numerical solution, t=T.")
#     ax = fig.gca(projection='3d')
#
#     # ax.plot_surface(X, Y, u_test, cmap=cm.coolwarm)  # Surface-plot
#     ax.plot_surface(X, Y, U_final[-1, :, :], cmap=cm.coolwarm, alpha=0.5)  # Surface-plot
#     ax.plot_surface(X, Y, f_test, cmap=cm.plasma, alpha=0.5)  # Surface-plot
#
#     plt.xlabel('x', fontsize=12)
#     plt.ylabel('y', fontsize=12)
#     ax.set_zlabel("$U_{i, j}$", fontsize=12)
#     # ax.set_zlim(0.0, 1.0)
#
#     plt.show()
#
#     pass


def test_2():

    L, T = 1.0, 2.0
    M, N = 50, 100

    def u_exact(x, y, t):
        return t ** 3 + y * (1 - y)

    def f(x, y, t, u):
        return 3 * t ** 2 + 1.0
    mu = 0.5

    X, Y = np.meshgrid(np.linspace(0.0, L, M), np.linspace(0.0, L, M))

    u_init = u_exact(X, Y, 0.0)

    def BC_E(x, y, t, u):
        return 0.0

    def BC_N(x, y, t, u):
        return -1.0

    def BC_W(x, y, t, u):
        return 0.0

    def BC_S(x, y, t, u):
        return -1.0

    boundary_funcs = [np.vectorize(func) for func in (BC_E, BC_N, BC_W, BC_S)]
    solver = DiffusionReactionSolver2D(u_init, (X, Y), mu, f, N, T, boundary_funcs)

    U_final = solver.execute()

    u_test = u_exact(X, Y, T)
    f_test = f(X, Y, 0.0, T)

    fig = plt.figure()
    fig.suptitle("Numerical solution, t=T.")
    ax = fig.gca(projection='3d')

    # ax.plot_surface(X, Y, u_test, cmap=cm.coolwarm)  # Surface-plot
    ax.plot_surface(X, Y, U_final[-1, :, :], cmap=cm.coolwarm, alpha=0.9)  # Surface-plot
    # ax.plot_surface(X, Y, f_test, cmap=cm.plasma, alpha=0.5)  # Surface-plot

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    ax.set_zlabel("$U_{i, j}$", fontsize=12)
    # ax.set_zlim(0.0, 1.0)
    ax.legend()
    plt.show()

    pass


if __name__ == '__main__':
    # two_dim_test()
    test_2()

# two_dim_step_matrices(4, 0.2)
# two_dim_laplace_neumann(4)
