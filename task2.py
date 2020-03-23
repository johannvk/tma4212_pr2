# Numerical Libraries:
import numpy as np
from scipy.integrate import solve_ivp

# Plotting libraries:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm

# Type enforcing:
from typing import Union, Callable, Tuple, Iterable

# Solver:
from solvers import SIR_Model, SIR_Animation


def SIR_ode_sys(t: float, y: np.ndarray, beta: float, gamma: float):
    S, I, R = y[0], y[1], y[2]
    return np.array([-beta*S*I, beta*S*I - gamma*I, gamma*I])


def SIR_ode_epidemic():

    # Problem definitions:
    beta = 1.8
    gamma = 0.5

    y0 = np.array([0.9, 0.1, 0.0])
    t_span = [0.0, 15.0]
    num_evaluations = 100
    t_evaluations = np.linspace(t_span[0], t_span[1], num_evaluations)

    # Let the solver itself progress and step forward in time as it decides:
    SIR_solution = solve_ivp(SIR_ode_sys, t_span, y0, dense_output=True, args=(beta, gamma))

    print(f"Shape of sol.y: {SIR_solution.y.shape}")

    # Afterwards interpolate the function using the function evaluations found already:
    SIR_func = SIR_solution.sol

    # Stack the solutions coming out in shape (3,) into shape (3, num_evaluations):
    interp_ys = np.stack([SIR_func(t) for t in t_evaluations], axis=1)

    print(f"ys.shape:\n{interp_ys.shape}")

    # Plotting code:
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    ax.plot(t_evaluations, interp_ys[0, :], 'r-', label="S")
    ax.plot(t_evaluations, interp_ys[1, :], 'k-', label="I")
    ax.plot(t_evaluations, interp_ys[2, :], 'b-', label="R")

    ax.legend(loc='best')
    plt.show()


def test_two_dim_SIR():

    M, N = 50, 80
    L, T = 5.0, 2.0
    xs, ys = np.linspace(0.0, L, M), np.linspace(0.0, L, M)
    X, Y = np.meshgrid(xs, ys)

    S_init = np.ones((M, M))*0.3

    init_indices = np.full((M, M), False, dtype=bool)
    # Setting inner square to True:
    init_indices[(1.0 < X) & (X < 3.0) & (2.0 < Y) & (Y < 4.0)] = True

    S_init[init_indices] += np.abs(np.sin(np.pi*np.sqrt(X[init_indices]**2 + Y[init_indices]**2)))

    I_init = np.zeros((M, M))
    I_init[:, :] = 0.2

    model = SIR_Model(S_init, I_init, mu_S_I=(0.05, 0.5), beta=2.0, gamma=0.3, domain=(X, Y), N=N, T=T)
    animate_model = SIR_Animation(model)
    animate_model.play_animation()

    # model.execute()
    #
    # S_final = model.S_solver.u_n
    # I_final = model.I_solver.u_n
    #
    # # Plotting code:
    # fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
    # fig.suptitle("Numerical solution, t=T.")
    #
    # axes[0].plot_surface(X, Y, S_final, cmap=cm.coolwarm, alpha=0.9)  # Surface-plot
    # axes[0].set_title("Final Susceptible")
    #
    # axes[1].plot_surface(X, Y, I_final, cmap=cm.coolwarm, alpha=0.9)  # Surface-plot
    # axes[1].set_title("Final Infected")
    #
    # axes[0].set_xlabel('X')
    # axes[0].set_ylabel('Y')
    #
    # axes[1].set_xlabel('X')
    # axes[1].set_ylabel('Y')
    #
    # plt.tight_layout()
    # plt.show()


if __name__ == '__main__':
    # SIR_ode_epidemic()
    test_two_dim_SIR()
