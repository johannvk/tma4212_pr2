import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp


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


if __name__ == '__main__':
    SIR_ode_epidemic()
