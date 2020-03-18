# Numerical Libraries:
import numpy as np
from scipy.integrate import solve_ivp

# Plotting libraries:
import matplotlib.pyplot as plt
from matplotlib import cm

# Type enforcing:
from typing import Union, Callable, Tuple, Iterable

# Solver:
from two_dim_solver import DiffusionReactionSolver2D


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


class SIR_Model:
    # Framework for the two interlocked reaction diffusion solvers.
    def __init__(self, S_init: np.ndarray, I_init: np.ndarray, mu_S_I: Tuple[float, float], beta: Union[float, Callable],
                 gamma: Union[float, Callable], domain: Tuple[np.ndarray, np.ndarray], N: int = 100, T: float = 1.0):
        """
        :param S_init:
        :param I_init:
        :param mu_I_S:
        :param beta:
        :param gamma:
        :param domain:
        :param N:
        :param T:
        """

        self.mu_S, self.mu_I = mu_S_I
        if not callable(beta):
            self.beta = np.vectorize(lambda *args: beta)
        else:
            self.beta = np.vectorize(beta)

        if not callable(gamma):
            self.gamma = np.vectorize(lambda *args: gamma)
        else:
            self.gamma = np.vectorize(gamma)

        self.X, self.Y = domain
        self.N = N
        self.T = T

        # Solver for the Susceptible population. Modify the reaction-vector function:
        self.S_solver = DiffusionReactionSolver2D(u_init=S_init, domain=domain, mu=self.mu_S, N=self.N, T=self.T)
        self.S_solver.generate_reaction_vector = self.S_generate_reaction_vector

        # Solver for the Infected population. Modify the reaction-vector function:
        self.I_solver = DiffusionReactionSolver2D(u_init=I_init, domain=domain, mu=self.mu_I, N=self.N, T=self.T)
        self.I_solver.generate_reaction_vector = self.I_generate_reaction_vector

    def S_generate_reaction_vector(self, *args):
        """
        :return: The recution in Susceptible population at each point in the domain.
        """
        return -self.beta(self.X, self.Y) * self.S_solver.u_n * self.I_solver.u_n

    def I_generate_reaction_vector(self, *args):
        """
        :return: The recution in Susceptible population at each point in the domain.
        """
        return (self.beta(self.X, self.Y) * self.S_solver.u_n - self.gamma(self.X, self.Y)) * self.I_solver.u_n

    def execute(self):
        for n in range(0, self.N - 1):
            self.S_solver.u_n = self.S_solver.two_dim_reaction_diffusion_step(n)
            self.I_solver.u_n = self.I_solver.two_dim_reaction_diffusion_step(n)


def test_two_dim_SIR():

    M, N = 50, 100
    L, T = 5.0, 5.0
    xs, ys = np.linspace(0.0, L, M), np.linspace(0.0, L, M)
    X, Y = np.meshgrid(xs, ys)

    S_init = np.zeros((M, M))
    S_init[int(M/6):int(5*M/6), int(M/6):int(5*M/6)] = 2.0

    I_init = np.zeros((M, M))
    I_init[int(2*M/4):int(3*M/4), int(2*M/4):int(3*M/4)] = 0.2*(X[int(2*M/4):int(3*M/4), int(2*M/4):int(3*M/4)]*(L - X[int(2*M/4):int(3*M/4), int(2*M/4):int(3*M/4)]))

    model = SIR_Model(S_init, I_init, mu_S_I=(0.2, 0.2), beta=2.0, gamma=0.3, domain=(X, Y), N=N, T=T)
    model.execute()

    S_final = model.S_solver.u_n
    I_final = model.I_solver.u_n

    # Plotting code:
    fig, axes = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
    fig.suptitle("Numerical solution, t=T.")

    axes[0].plot_surface(X, Y, S_final, cmap=cm.coolwarm, alpha=0.9)  # Surface-plot
    axes[0].set_title("Final Susceptible")

    axes[1].plot_surface(X, Y, I_final, cmap=cm.coolwarm, alpha=0.9)  # Surface-plot
    axes[1].set_title("Final Infected")

    axes[0].set_xlabel('X')
    axes[0].set_ylabel('Y')

    axes[1].set_xlabel('X')
    axes[1].set_ylabel('Y')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    SIR_ode_epidemic()
    test_two_dim_SIR()
