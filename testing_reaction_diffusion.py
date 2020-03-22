# Numerical Libraries:
import numpy as np

# Graphical libraries:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm

import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.stats import linregress

# Type enforcing:
from typing import Union, Callable, Tuple

from solvers import DiffusionReactionSolver1D, DiffusionReactionSolver2D

# font = {'family' : 'normal',
#         'weight' : 'bold',
#         'size'   : 22}
#
# matplotlib.rc('font', **font)
plt.rcParams.update({'font.size': 15})


def one_dim_diffusion_reaction_convergence(u_exact: Callable, mu: float, f: Callable,
                                           Neumann_BC : Union[None, Tuple[float, float]] = None,
                                           M: int = 100, X : Tuple[float, float] = (0.0, 1.0),
                                           N: int = 100, T: float = 1.0):
    xs = np.linspace(X[0], X[1], M)
    exact_solution_T = u_exact(xs, T)

    u_init = u_exact(xs, 0.0)

    Ns = np.logspace(1.0, 2.2, 10, dtype=int)
    Ms = np.logspace(1.0, 2.2, 10, dtype=int)
    time_sup_errors = np.zeros(len(Ns))
    space_sup_errors = np.zeros(len(Ms))

    for i, n in enumerate(Ns):
        RD_solver_N = DiffusionReactionSolver1D(u_init, xs, mu, f, N=n, T=T, X=X, Neumann_BC=Neumann_BC)
        solution_array_n = RD_solver_N.execute()
        time_sup_errors[i] = np.max(np.abs(exact_solution_T - solution_array_n[-1, :]))

    for i, m in enumerate(Ms):
        m_xs = np.linspace(X[0], X[1], m)
        m_u_init = u_exact(m_xs, 0.0)
        RD_solver_M = DiffusionReactionSolver1D(m_u_init, m_xs, mu, f, N=N, T=T, X=X, Neumann_BC=Neumann_BC)
        solution_array_m = RD_solver_M.execute()
        sup_m_error = np.max(np.abs(u_exact(m_xs, T) - solution_array_m[-1, :]))
        space_sup_errors[i] = sup_m_error

    print(f"Space sup-errors: {space_sup_errors}\n")
    print(f"Time sup-errors: {time_sup_errors}\n")

    Ns -= 1  # To get the precise value of 'h' when taking the reciprocal.
    time_slope = linregress(np.log10(1 / Ns), np.log10(time_sup_errors))[0]

    Ms -= 1
    space_slope = linregress(np.log10(1/Ms), np.log10(space_sup_errors))[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.set_title("Convergence Plot, Neumann Boundary")
    ax.plot(np.log10(1/Ns), np.log10(time_sup_errors), linewidth=2.5, label=f"Decreasing k, Slope {time_slope:.3f}, h={(X[1]-X[0])/(M-1):.2e}")
    ax.plot(np.log10(1/Ms), np.log10(space_sup_errors), linewidth=2.5, label=f"Decrasing h, Slope {space_slope:.3f}, k={T/(N-1):.2e}")

    ax.set_xlabel("log(step length)")
    ax.set_ylabel("log(Sup|u - U|)")
    ax.legend(loc="best", fontsize=12)
    plt.tight_layout()
    plt.show()


def test_one_dim_dirichlet_convergence():
    print("Testing one-dimensional Dirichlet boundary convergence.\n")

    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.sin(np.pi*x) - t*x*(1.0 - x)
    mu = 1.0
    f = lambda x, t, u: -x * (1.0 - x) - 2*t

    one_dim_diffusion_reaction_convergence(u_exact=u_exact, mu=mu, f=f)


def test_one_dim_dirichlet():
    print("Testing one-dimensional Neumann boundary conditions.\n")
    M, N = 160, 200
    X, T = (0.0, 1.0), 0.1

    xs = np.linspace(X[0], X[1], M)

    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.sin(np.pi*x) - t*x*(1.0 - x)

    exact_solution_T = u_exact(xs, T)

    u_init = u_exact(xs, 0.0)
    mu = 1.0
    f = lambda x, t, u: - 1.0 * x * (1.0 - x) - 2*t

    RD_solver = DiffusionReactionSolver1D(u_init, xs, mu, f, N=N, T=T, X=X)
    solution_array = RD_solver.execute()

    sup_error = np.max(np.abs(exact_solution_T - solution_array[-1, :]))
    print(f"Sup error is: {sup_error:.3e}")

    fig, axis = plt.subplots(1, 1)

    axis.plot(xs, solution_array[-1, :], label="U_num")
    axis.plot(xs, exact_solution_T, label="U_exact")

    axis.legend(loc="best")
    axis.set_xlim(X)
    plt.show()


def test_one_dim_neumann_convergence():
    print("Testing one-dimensional Neumann Convergence.\n")

    # def u_exact(x, t):
    #     return t**2 + x*(1-x)
    #
    # def f(x, t, u): return 2*t + 1.0
    # mu = 0.5
    # neumann_bc = (1.0, -1.0)

    x0, L = 0.0, 2.0
    X, T = (x0, L), 12.0

    mu = 0.5

    def u_exact(x, t):
        return np.sin(x) + np.cos(t)

    def f(x, t, u):
        return -np.sin(t) + mu * np.sin(x)

    neumann_bc = (np.cos(x0), np.cos(L))

    one_dim_diffusion_reaction_convergence(u_exact, mu, f, Neumann_BC=neumann_bc, X=X, T=T)


def test_one_dim_neumann():
    print("Testing one-dimensional Neumann boundary conditions.\n")
    x0, L = 0.0, 2.0
    M, N = 20, 20
    X, T = (x0, L), 1.0

    mu = 0.5

    def u_exact(x, t):
        return np.sin(x) + np.cos(t)

    def f(x, t, u):
        return -np.sin(t) + mu*np.sin(x)

    # def u_exact(x, t):
    #     return t**3 + x*(1-x)
    #
    # def f(x, t, u): return 3*t**2 + 1.0

    xs = np.linspace(X[0], X[1], M)

    u_init = u_exact(xs, 0.0)
    exact_solution_T = u_exact(xs, T)
    neumann_bc = (np.cos(x0), np.cos(L))

    RD_solver_Neumann = DiffusionReactionSolver1D(u_init, xs, mu, f, N=N, T=T, X=X, Neumann_BC=neumann_bc)
    solution_array = RD_solver_Neumann.execute()

    sup_error = np.max(np.abs(exact_solution_T - solution_array[-1, :]))
    print(f"Sup error is: {sup_error:.3e}")

    fig, axis = plt.subplots(1, 1)

    axis.plot(xs, solution_array[-1, :], label="U_num", alpha=0.8)
    axis.plot(xs, exact_solution_T, label="U_exact", alpha=0.8)

    axis.legend(loc="best")
    axis.set_xlim(X)
    plt.show()


def test_1_two_dim_neumann():
    print("Testing two dimensional Neumann boundary conditions.\n")
    L, T = 3.0, 1.0
    M, N = 100, 100
    xs, ys = np.linspace(0.0, L, M), np.linspace(0.0, L, M)
    X, Y = np.meshgrid(xs, ys)

    def u_exact(x, y, t):
        return t**3 + 2*y*(L - y) + x*(L - x)

    mu = 1.0

    def f(x, y, t, *args):
        return 3*t**2 - mu*(-6)

    u_init = u_exact(X, Y, 0.0)
    bc_funcs = [lambda *args: -L, lambda *args: -2.00*L, lambda *args: -L, lambda *args: -2*L]

    solver = DiffusionReactionSolver2D(u_init, (X, Y), f, mu, N, T, bc_funcs)
    u_num = solver.execute()

    u_test = u_exact(X, Y, T)

    fig = plt.figure()
    fig.suptitle("Numerical solution, t=T.")
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, u_test, cmap=cm.plasma, alpha=0.9)  # Surface-plot
    ax.plot_surface(X, Y, u_num[-1, :, :], cmap=cm.coolwarm, alpha=0.9)  # Surface-plot

    errors = np.abs(u_test - u_num[-1, :, :])
    sup_error = np.max(errors)
    sup_error_loc = np.unravel_index(np.argmax(errors), errors.shape)
    print(f"The position with the sup-error is: {sup_error_loc}")
    print(f"Then sup-error is: {sup_error:.3e}")

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    ax.set_zlabel("$U_{i, j}$", fontsize=12)

    plt.show()


def test_2_two_dim_neumann():
    print("Testing two dimensional Neumann boundary conditions.\n")
    L, T = 1.0, 1.0
    M, N = 200, 100

    def u_exact(x, y, t):
        return t ** 3 + x * (1 - x)

    def f(x, y, t, u):
        return 3 * t ** 2 + 1.0
    mu = 0.5

    X, Y = np.meshgrid(np.linspace(0.0, L, M), np.linspace(0.0, L, M))

    u_init = u_exact(X, Y, 0.0)

    def BC_E(x, y, t, u):
        return -1.0

    def BC_N(x, y, t, u):
        return 0.0

    def BC_W(x, y, t, u):
        return -1.0

    def BC_S(x, y, t, u):
        return 0.0

    boundary_funcs = [np.vectorize(func) for func in (BC_E, BC_N, BC_W, BC_S)]
    solver = DiffusionReactionSolver2D(u_init, (X, Y), f, mu, N, T, boundary_funcs)

    u_num = solver.execute()

    u_test = u_exact(X, Y, T)

    errors = np.abs(u_test - u_num[-1, :, :])
    sup_error = np.max(errors)
    sup_error_loc = np.unravel_index(np.argmax(errors), errors.shape)
    print(f"The position with the sup-error is: {sup_error_loc}")
    print(f"Then sup-error is: {sup_error:.3e}")

    fig = plt.figure()
    fig.suptitle("Numerical solution, t=T.")
    ax = fig.gca(projection='3d')

    ax.plot_surface(X, Y, u_test, cmap=cm.plasma, alpha=0.5)  # Surface-plot
    ax.plot_surface(X, Y, u_num[-1, :, :], cmap=cm.coolwarm, alpha=0.5)  # Surface-plot

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    ax.set_zlabel("$U_{i, j}$", fontsize=12)

    plt.show()


def test_batch():
    # test_one_dim_dirichlet_convergence()
    # test_one_dim_dirichlet()

    test_one_dim_neumann_convergence()
    # test_one_dim_neumann()

    # test_1_two_dim_neumann()
    # test_2_two_dim_neumann()


if __name__ == '__main__':
    test_batch()
