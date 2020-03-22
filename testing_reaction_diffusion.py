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


def one_dim_diffusion_reaction_convergence(u_exact: Callable, mu: float, f: Callable,
                                           Neumann_BC : Union[None, Tuple[float, float]] = None,
                                           M: int = 200, X : Tuple[float, float] = (0.0, 1.0), T: float = 1.0):
    xs = np.linspace(X[0], X[1], M)
    exact_solution_T = u_exact(xs, T)

    u_init = u_exact(xs, 0.0)

    Ns = np.logspace(1.0, 2.2, 10, dtype=int)
    sup_errors = np.zeros(len(Ns))

    for i, N in enumerate(Ns):
        RD_solver_N = DiffusionReactionSolver1D(u_init, xs, mu, f, N=N, T=T, X=X, Neumann_BC=Neumann_BC)
        solution_array = RD_solver_N.execute()
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


def test_one_dim_dirichlet_convergence():
    print("Testing one-dimensional Dirichlet boundary convergence.\n")

    def u_exact(x, t):
        return np.exp(-np.pi**2*t)*np.sin(np.pi*x) - t*x*(1.0 - x)
    mu = 1.0
    f = lambda x, t, u: 1.0*(-x * (1.0 - x) - 2*t)

    one_dim_diffusion_reaction_convergence(u_exact=u_exact, mu=mu, f=f)


def test_one_dim_dirichlet():
    print("Testing one-dimensional Neumann boundary conditions.\n")
    M, N = 200, 200
    X, T = (0.0, 1.0), 5.0

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
    def u_exact(x, t):
        return t**3 + x*(1-x)

    def f(x, t, u): return 3*t**2 + 1.0
    mu = 0.5

    one_dim_diffusion_reaction_convergence(u_exact, mu, f, Neumann_BC=(1.0, -1.0))


def test_one_dim_neumann():
    print("Testing one-dimensional Neumann boundary conditions.\n")

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
    test_one_dim_dirichlet_convergence()
    test_one_dim_dirichlet()

    test_one_dim_neumann_convergence()
    test_one_dim_neumann()

    test_1_two_dim_neumann()
    test_2_two_dim_neumann()


if __name__ == '__main__':
    test_batch()
