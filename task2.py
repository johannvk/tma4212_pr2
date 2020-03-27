# Numerical Libraries:
import numpy as np
from scipy.integrate import solve_ivp

# Plotting libraries:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # For 3-d plot
from matplotlib import cm
import matplotlib.gridspec as gs

# Type enforcing:
from typing import Union, Callable, Tuple, Iterable

# Solver:
from solvers import SIModel, SIAnimation


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


def display_two_dim_SI():
    h_optimal = 0.1
    k_optimal = 0.1
    L, T = 8.0, 1.0

    domain_length = 2*abs(L)
    M = int(domain_length/h_optimal) + 2
    N = int(T/k_optimal) + 2
    print("Simulating single source model.")
    print(f"M: {M}\tN: {N}\n")

    xs, ys = np.linspace(-L, L, M), np.linspace(-L, L, M)
    X, Y = np.meshgrid(xs, ys)

    middle_index = int(M/2)

    S_init = np.ones((M, M))*1.0
    S_init[middle_index-10:middle_index+11, middle_index-10:middle_index+11] -= 0.3

    I_init = np.ones((M, M))*0.0
    I_init[middle_index-10:middle_index+11, middle_index-10:middle_index+11] = 0.3

    model = SIModel(S_init, I_init, mu_S_I=(0.1, 0.2), beta=2.5, gamma=0.5, domain=(X, Y), N=N, T=T, store=False)
    S_final, I_final = model.execute()

    # Plotting code:
    fig = plt.figure(figsize=(10, 8))
    gridspec = gs.GridSpec(1, 3, width_ratios=[1, 1, 0.05])
    zlim = (0.0, 1.1*np.maximum(np.max(S_final), np.max(I_final)))
    initial_view = {'elev': 15, 'azim': 35}

    S_ax = fig.add_subplot(gridspec[0], projection='3d', zlim=zlim,
                           xlabel="X", ylabel="Y")
    I_ax = fig.add_subplot(gridspec[1], projection='3d', zlim=zlim,
                           xlabel="X", ylabel="Y")
    color_ax = fig.add_subplot(gridspec[2])

    # Adding single color-bar:
    mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
    # mappable.set_array(S_ax)
    mappable.set_clim(*zlim)
    plt.colorbar(mappable, cax=color_ax, orientation="vertical")

    plot_args = {'rstride': 1, 'cstride': 1, 'linewidth': 0.01, 'cmap': mappable.cmap, 'norm': mappable.norm,
                 'antialiased': True, 'shade': True}

    S_ax.plot_surface(X, Y, S_init, **plot_args)
    I_ax.plot_surface(X, Y, I_init, **plot_args)

    S_ax.view_init(**initial_view)
    I_ax.view_init(**initial_view)

    S_ax.set_title("S: Susceptible Population", pad=20, fontsize=16)
    I_ax.set_title("I: Infected Population", pad=20, fontsize=16)

    time_text = plt.figtext(x=0.02, y=0.94, s=f"T: {0}")

    plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.99])
    plt.show()


def gaussian(X, Y, means=np.array([0.0, 0.0]), variance=np.array([1.0, 1.0])):
    assert(np.all(variance > 0))
    normalization = 1.0/(2*np.pi*np.prod(variance))
    return normalization*np.exp(-0.5*(((X - means[0])/variance[0])**2 + ((Y - means[1])/variance[1])**2))


def two_city_model():
    h_optimal = 0.2
    k_optimal = 0.1
    L, T = 20.0, 20.0

    domain_length = 2*abs(L)
    M = int(domain_length/h_optimal) + 2
    N = int(T/k_optimal) + 2
    print("Simulating two city model.")
    print(f"M: {M}\tN: {N}\n")

    xs, ys = np.linspace(-L, L, M), np.linspace(-L, L, M)
    X, Y = np.meshgrid(xs, ys)

    city_1 = np.array([-7, 0.0])
    city_2 = np.array([7, 0.0])
    std_dev = 1.5
    variance = np.array([1.0, 1.0])*std_dev**2

    S_init = np.ones((M, M))*0.0
    S_init += 10.0*gaussian(X, Y, means=city_1, variance=variance)
    S_init += 10.0*gaussian(X, Y, means=city_2, variance=variance)

    peak_height = np.max(S_init)
    # I_init = np.ones((M, M))*0.0

    # Outbreak centered at city 2:
    oubreak_center = city_2
    I_init = 0.5*gaussian(X, Y, means=city_2, variance=np.array([1.0, 1.0]))
    S_init -= I_init

    beta = 10.0/peak_height
    gamma = 0.1*beta  # /peak_height
    model = SIModel(S_init, I_init, mu_S_I=(0.1, 0.5), beta=beta, gamma=gamma, domain=(X, Y), N=N, T=T, store=False)
    print(f"beta: {beta}")
    print(f"gamma: {gamma}\n")

    init_view = {'elev': 20, 'azim': 90}
    z_limits = (np.array([0.0, 1.1*peak_height]), np.array([0.0, 1.1*np.max(I_init)]))
    animate_model = SIAnimation(model, zlims=z_limits, azim_rotation=0.5, initial_view=init_view)

    filename = f"two_cities_one_city_outbreak_L{int(L)}_T{int(T)}_beta{int(beta)}_gamma{int(gamma)}_var{int(std_dev**2)}"
    last_frame = animate_model.play_animation(save=True, filename=filename, as_gif=True)
    np.save("lastframes//" + filename + "_last_frames.npz", last_frame)
    print("Done displaying/saving the animation!")


def single_source_model():
    h_optimal = 0.2
    k_optimal = 0.1
    L, T = 20.0, 20.0

    domain_length = 2*abs(L)
    M = int(domain_length/h_optimal) + 2
    N = int(T/k_optimal) + 2
    print("Simulating single source model.")
    print(f"M: {M}\tN: {N}\n")

    xs, ys = np.linspace(-L, L, M), np.linspace(-L, L, M)
    X, Y = np.meshgrid(xs, ys)

    middle_index = int(M/2)

    S_init = np.ones((M, M))*1.0
    S_init[middle_index-10:middle_index+11, middle_index-10:middle_index+11] -= 0.3

    I_init = np.ones((M, M))*0.0
    I_init[middle_index-10:middle_index+11, middle_index-10:middle_index+11] = 0.3

    model = SIModel(S_init, I_init, mu_S_I=(0.1, 0.2), beta=1.5, gamma=0.5, domain=(X, Y), N=N, T=T, store=False)

    z_limits = (np.array([0.0, 1.1]), np.array([0.0, 1.1]))
    init_view = {'elev': 15, 'azim': 0}
    animate_model = SIAnimation(model, zlims=z_limits, azim_rotation=0.5, initial_view=init_view)
    filename = f"single_source_L{int(L)}_T{int(T)}_2"

    last_frame = animate_model.play_animation(save=True, filename=filename, as_gif=True)
    np.save("lastframes//" + filename + "_last_frames.npz", last_frame)
    print("Done displaying/saving the animation!")


if __name__ == '__main__':
    SIR_ode_epidemic()
    display_two_dim_SI()
    # two_city_model()
    # single_source_model()
