# Numerical Libraries:
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Graphical Libraries:
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.cm as cm
import matplotlib.gridspec as gs
# import ffmpeg

# Type enforcing:
from typing import Union, Callable, Tuple, Iterable

# Set the path for the mp4-animation-saver:
plt.rcParams['animation.ffmpeg_path'] = r'C:\FFmpeg\bin\ffmpeg.exe'


def one_dim_sparse_laplacian(m: int):
    """
    Function to generate a discretized matrix approximation of the 1D-Laplacian.
    :param m: Dimension of the square matrix.
    :return: An (m, m) sparse matrix.
    """
    return sp.diags([1.0, -2.0, 1.0], [-1, 0, 1], dtype='float64', shape=(m, m), format='lil')


def two_dim_sparse_neumann_laplacian(M: int, format: str = 'coo', dtype: str = 'float64'):
    """
    Building the discretized Laplacian with Neumann boundary conditions in two dimensions
    with block matrix sparse tools. Need specification for this in the solver-step.
    :param M: Number of spatial discretization points in each spatial dimension.
    :param format: Storage format for the sparse matrices.
    :param dtype: Data type in the sparse matrices.
    :return: Discretized two dimensional Laplacian, with Neumann boundary conditions.
    """
    inner_data = [[1.0] * (M - 2) + [2.0], -4.0, [2.0] + [1.0] * (M - 2)]
    inner_diag = sp.diags(inner_data, offsets=[-1, 0, 1], format=format, dtype=dtype)
    I_m = sp.identity(M, format=format, dtype=dtype)

    # Rows of matrices.
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


class DiffusionReactionSolver1D:

    def __init__(self, u_init: np.ndarray, xs: np.ndarray, mu: float, f: Callable, N: int, T: float = 1.0,
                 X: Union[float, Tuple[float, float]] = 1.0, Neumann_BC: Union[Tuple[float, float], None] = None):
        """
        Solving on the domain x in (0, X) or x in (X[0], X[1])
        :param u_init: Initial vector-values for u(x, 0). u_init[0] and u_init[-1] used as Dirichlet boundary values.
        :param xs: Discretization of solution domain.
        :param mu: Diffusion coefficient.
        :param f: Source term. f(x, t, u).
        :param N: Number of temporal discretization points.
        :param T: End time for solution.
        :param X: End point in space for solution. Assumed start at x=0 unless otherwise specified.
        :param Neumann_BC: Variable indicating whether or not to use Neumann boundary conditions.
        """
        # Storing spatial domain specifications:
        self.M = len(u_init)
        self.xs = xs
        if isinstance(X, float):
            self.h = (X - 0.0)/(self.M-1)
        else:
            self.h = (X[1] - X[0])/(self.M-1)

        # Storing temporal domain specifications:
        self.N = N
        self.T = T
        self.k = (self.T - 0.0)/(self.N-1)

        # Storing problem specifications:
        self.mu = mu
        self.f = np.vectorize(f)
        self.Neumann_BC = Neumann_BC
        self.mu = mu
        self.r = self.mu*self.k/(self.h*self.h)

        # Generating step matrices:
        self.I_minus_Lap, self.I_plus_Lap = self.one_dim_generate_step_matrices()

        # Preparing storage of the solution:
        self.u_n = np.copy(u_init)
        self.u_storage = np.zeros((self.N, self.M), dtype='float64')
        self.u_storage[0, :] = np.copy(u_init)

    def one_dim_generate_step_matrices(self):
        # Discretized Laplacian in one dimension:
        Lap = one_dim_sparse_laplacian(self.M)

        if self.Neumann_BC is None:
            # Adapting the first and last row to Neumann BC:
            Lap[0, [0, 1]] = [0.0, 0.0]
            Lap[self.M - 1, [self.M - 2, self.M - 1]] = [0.0, 0.0]
        else:
            # Adapting the first and last row to Neumann BC:
            Lap[0, 1] = 2.0
            Lap[self.M - 1, self.M - 2] = 2.0

        r_half_Lap = (self.r / 2.0) * Lap

        # Find the LU-factorized version of the left-side matrix in Crank-Nicolson. Returns a callable:
        I_minus_r_Lap = spla.factorized((sp.identity(self.M, dtype='float64', format='lil') - r_half_Lap).tocsc())
        I_plus_r_Lap = (sp.identity(self.M, dtype='float64', format='lil') + r_half_Lap).tocsc()  # Turn into csc

        return I_minus_r_Lap, I_plus_r_Lap

    def one_dim_reaction_diffusion_step(self, u: np.ndarray, n: int):
        """
        Function returning a time step for u^(n). First find solution u_star to
        (I_minus_Lap)*u_star = (I_plus_Lap)*u + k*f(u). Then return u_star + (k/2.0)*(f(xs, u_star) - f(xs, u)).
        :param u: Current vector-represented function, u^(n), before a step is taken.
        :param n: Which time step we are on.
        :return: Next step, u^(n+1).
        """
        t = n * self.k

        # Create the initial source vector:
        f_vec = np.array([self.f(self.xs[j], t, u[j]) for j in range(self.M)], dtype='float64')

        if self.Neumann_BC is None:
            f_vec[0] = 0.0
            f_vec[-1] = 0.0
            right_side_vector = self.I_plus_Lap.dot(u) + self.k * f_vec
        else:
            r_h = self.mu * self.k / self.h
            right_side_vector = self.I_plus_Lap.dot(u) + self.k * f_vec
            right_side_vector[0] -= 2 * r_h * self.Neumann_BC[0]
            right_side_vector[-1] += 2 * r_h * self.Neumann_BC[1]

        # Solves the linear system Ax = b, by calling I_minus_Lap(right_side_vector)
        u_star = self.I_minus_Lap(right_side_vector)

        # Create the intermediate source vector:
        f_vec_star = np.array([self.f(self.xs[j], t + self.k, u_star[j]) for j in range(self.M)], dtype='float64')

        if self.Neumann_BC is None:
            f_vec_star[0] = 0.0
            f_vec_star[-1] = 0.0

        return u_star + (self.k / 2.0) * (f_vec_star - f_vec)

    def execute(self):
        """
        Step the solver forward in time through the specified number of steps.
        :return: (N, M)-np.ndarray storing the solution at every time- and space-point.
        """
        for n in range(0, self.N-1):
            # Subtract 1 from i to start from time-step 0 (t_0), instead of time-step 1 (t_1).
            self.u_n = self.one_dim_reaction_diffusion_step(self.u_n, n)
            self.u_storage[n+1, :] = np.copy(self.u_n)

        return self.u_storage


class DiffusionReactionSolver2D:

    def __init__(self, u_init: np.ndarray, domain: Tuple[np.ndarray, np.ndarray], f: Union[Callable, None] = None,
                 mu: float = 1.0, N: int = 100, T: float = 1.0, Neumann_BC = None, store=True, *args):
        """
        Initializer function for
        :param u_init: Initial Values for the distribution, in (M, M)-np.ndarray.
        :param domain: Tuple[np.ndarray, np.ndarray] with the X-domain and Y-domain.
        :param mu: Diffusion coefficient.
        :param f: Reaction function/Source term.
        :param N: Number of temporal discretization points.
        :param T: Final time for simulation.
        :param Neumann_BC: functions specifying the Neumann Boundary conditions.
        :param *args: Any parameters to pass into the Reaction function/Source term.
        :return:
        """
        # Need the step sizes in both spatial dimensions to be equal.
        assert np.max(domain[0]) == np.max(domain[1])
        assert np.min(domain[0]) == np.min(domain[1])

        # Need a square domain, with equal number of points in each spatial dimension.
        assert u_init.shape[0] == u_init.shape[1]

        if Neumann_BC is None:  # No boundary conditions supplied, assumes zero derivatives at the boundaries.
            self.Neumann_BC = [np.vectorize(lambda *args: 0.0)] * 4
        else:
            self.Neumann_BC = [np.vectorize(func) for func in Neumann_BC]

        # Storing domain parameters:
        # M: Number of spatial discretization points in each spatial dimension.
        self.M = u_init.shape[0]
        # N: Number of temporal discretization points.
        self.N = N
        # End time of simulation.
        self.T = T
        # Domain stored in a np.meshgrid format.
        self.X, self.Y = domain

        # Retaining an (M, M)-np.ndarray for access to current values of u at time step n.
        self.u_n = np.copy(u_init)

        # Storing the initial state of the function, and making storage for the steps taken.
        self.store = store
        if self.store:
            self.u_storage = np.zeros((self.N, self.M, self.M), dtype='float64')
            self.u_storage[0, :, :] = np.copy(u_init)

        # Step size in space and time respectively:
        self.h = (np.max(domain[0]) - np.min(domain[0])) / (self.M - 1)
        self.k = (self.T - 0.0) / (self.N - 1)

        # Storing the Reaction function and Diffusion coefficient within the Class:
        if callable(f):
            self.f = np.vectorize(lambda x, y, t, u: f(x, y, t, u, *args))
        else:
            self.f = np.vectorize(lambda *args: 0.0)
        self.mu = mu

        # Getting the composite diffusion/step-size parameter r:
        self.r = self.mu * self.k / (self.h**2)

        # Generating the Left-hand and Right-hand side matrices for doing implicit steps:
        self.I_minus_Lap, self.I_plus_Lap = self.generate_two_dim_step_matrices()

        # Setting the boundary boolean arrays:
        self.boundaries = [np.full((self.M, self.M), False, dtype=bool), np.full((self.M, self.M), False, dtype=bool),
                           np.full((self.M, self.M), False, dtype=bool), np.full((self.M, self.M), False, dtype=bool)]
        self.boundaries[0][1:self.M - 1, self.M - 1] = True  # Eastern boundary.
        self.boundaries[1][self.M - 1, 1:self.M - 1] = True  # Northern boundary.
        self.boundaries[2][1:self.M - 1, 0] = True  # Western boundary.
        self.boundaries[3][0, 1:self.M - 1] = True  # Southern boundary.
        self.corners = [(self.M - 1, self.M - 1), (0, self.M - 1), (0, 0), (self.M - 1, 0)]  # Corner indices.

    def generate_two_dim_step_matrices(self):
        """
        Generates the required step matrices for doing a single step in the diffusion-reaction solver.
        Assumes Neumann boundary conditions.
        :return: I_minus_Lap: Callable, LU-factorized Implicit-solver matrix.
                 I_plus_Lap: Sparse matrix, for generating right side vector.
        """
        Lap_h = two_dim_sparse_neumann_laplacian(self.M, format='csc')
        I_m = sp.identity(self.M * self.M, dtype='float64', format='csc')

        I_minus_Lap = spla.factorized(I_m - (self.r / 2.0) * Lap_h)
        I_plus_Lap = I_m + (self.r / 2.0) * Lap_h

        return I_minus_Lap, I_plus_Lap

    def generate_reaction_vector(self, u: np.ndarray, n: int):
        """
        Function to generate the reaction term vector.
        :param u: (M, M)-np.ndarray. Current values of solution at time step n.
        :param n: Time step.
        :return: (M, M)-np.ndarray with k*f(x, y, t, u).
        """
        return self.f(self.X, self.Y, n*self.k, u)

    def generate_right_side_vector(self, n: int):
        """
        Generating the right-hand-side vector for the Implicit solve in the Diffusion-reaction scheme.
        :param u_n: Current solution at timestep n, an (M, M)-np.ndarray.
        :param M: Number of spatial discretization points in each spatial dimension.
        :param n: Current time step. From 0 to N-1.
        :param X: (M, M)-np.ndarray storing the X-values for the domain in a meshgrid-format.
        :param Y: (M, M)-np.ndarray storing the Y-values for the domain in a meshgrid-format.
        :param I_plus_Lap: Right hand side explicit part of the diffusion step.
        :param f: Reaction term. Callable function as a function of (x, y, t, u).
        :param bc_funcs: Boundary condition functions. Ordered {East: 0, North: 1, West: 2, South: 3}.
                         Also have to accept the arguments as (x, y, t, u).
        :return: (M*M,)-np.ndarray Right-hand-side vector used for the Implicit solve.
        """
        # Initializing the right-hand-side vector:
        f_vec = self.k * self.generate_reaction_vector(self.u_n, n)

        # Current time:
        t_n = n * self.k

        # Boundary condition multiplier:
        mult_bc = 2 * self.mu * self.k / self.h

        # Boolean masks for boundary indices: East: 0, North: 1, West: 2, South: 3.
        for i, boundary in enumerate(self.boundaries):
            f_vec[boundary] += mult_bc * self.Neumann_BC[i](self.X[boundary], self.Y[boundary], t_n, self.u_n[boundary])

        for i, (xi, yi) in enumerate(self.corners):
            corner_i = self.Neumann_BC[i](self.X[xi, yi], self.Y[xi, yi], t_n, self.u_n[xi, yi])
            corner_i_plus_1 = self.Neumann_BC[(i + 1) % 4](self.X[xi, yi], self.Y[xi, yi], t_n, self.u_n[xi, yi])
            f_vec[xi, yi] += mult_bc * (corner_i + corner_i_plus_1)

        return self.I_plus_Lap.dot(self.u_n.ravel(order='C')) + f_vec.ravel(order='C')

    def two_dim_reaction_diffusion_step(self, n: int):
        # Prepare the right hand side vector:
        rhs = self.generate_right_side_vector(n)  # .ravel(order='C')
        u_star = self.I_minus_Lap(rhs).reshape((self.M, self.M), order='C')

        f_vec = self.generate_reaction_vector(self.u_n, n)
        f_vec_star = self.generate_reaction_vector(u_star, n + 1)

        return u_star + (self.k / 2.0) * (f_vec_star - f_vec)

    def execute(self):
        if self.store:
            for n in range(0, self.N - 1):
                self.u_n = self.two_dim_reaction_diffusion_step(n)
                self.u_storage[n + 1, :, :] = np.copy(self.u_n)
            return self.u_storage
        else:
            for n in range(0, self.N - 1):
                self.u_n = self.two_dim_reaction_diffusion_step(n)
            return self.u_n


class SIModel:
    # Framework for the two interlocked reaction diffusion solvers.
    def __init__(self, S_init: np.ndarray, I_init: np.ndarray, mu_S_I: Tuple[float, float],
                 beta: Union[float, Callable], gamma: Union[float, Callable], domain: Tuple[np.ndarray, np.ndarray],
                 N: int = 100, T: float = 1.0, store=True):
        """
        :param S_init:
        :param I_init:
        :param mu_I_S:
        :param beta:
        :param gamma:
        :param domain:
        :param N:
        :param T:
        :param store: Whether or not to store the values of the arrays as they are computed.
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
        self.M = S_init.shape[0]
        self.N = N
        self.T = T
        self.store = store

        # Solver for the Susceptible population. Modify the reaction-vector function:
        self.S_solver = DiffusionReactionSolver2D(u_init=S_init, domain=domain, mu=self.mu_S, N=self.N, T=self.T,
                                                  store=self.store)
        self.S_solver.generate_reaction_vector = self.S_generate_reaction_vector

        # Solver for the Infected population. Modify the reaction-vector function:
        self.I_solver = DiffusionReactionSolver2D(u_init=I_init, domain=domain, mu=self.mu_I, N=self.N, T=self.T,
                                                  store=self.store)
        self.I_solver.generate_reaction_vector = self.I_generate_reaction_vector

        # Storage and execution status:
        if self.store:
            self.S_storage = np.zeros((self.N, self.M, self.M), dtype='float64')
            self.S_storage[0, :, :] = np.copy(S_init)
            self.I_storage = np.zeros((self.N, self.M, self.M), dtype='float64')
            self.I_storage[0, :, :] = np.copy(I_init)

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

    def perform_single_step(self, n: int):
        self.S_solver.u_n = self.S_solver.two_dim_reaction_diffusion_step(n)
        self.I_solver.u_n = self.I_solver.two_dim_reaction_diffusion_step(n)

    def execute(self):
        if self.store:
            for n in range(0, self.N - 1):
                self.perform_single_step(n)
                self.S_storage[n+1, :, :] = np.copy(self.S_solver.u_n)
                self.I_storage[n+1, :, :] = np.copy(self.I_solver.u_n)
            return self.S_storage, self.I_storage
        else:
            for n in range(0, self.N - 1):
                self.perform_single_step(n)
            return self.S_solver.u_n, self.I_solver.u_n


class SIAnimation:

    def __init__(self, model: SIModel, zlim: np.ndarray, initial_view=None, elev_rotation=0.0, azim_rotation=1.0):
        self.frames = model.N
        self.X, self.Y = model.X, model.Y
        self.model = model
        self.n = 0
        self.fps = 25
        self.zlim = zlim

        if initial_view is None:
            self.initial_view = {'elev': 30, 'azim': 0}
        else:
            self.initial_view = initial_view
        self.rotation = 1.0
        self.elev_rotation = elev_rotation
        self.azim_rotation = azim_rotation

    def play_animation(self, save=False, filename='test_anim', as_gif=False):

        fig = plt.figure(figsize=(10, 8))
        gridspec = gs.GridSpec(1, 3, width_ratios=[1, 1, 0.05])

        S_ax = fig.add_subplot(gridspec[0], projection='3d', zlim=self.zlim,
                               xlabel="X", ylabel="Y")
        I_ax = fig.add_subplot(gridspec[1], projection='3d', zlim=self.zlim,
                               xlabel="X", ylabel="Y")
        color_ax = fig.add_subplot(gridspec[2])

        # Old way of generating figure and subplots:
        # fig, (S_ax, I_ax, color_ax) = plt.subplots(1, 2, figsize=(10, 6),
        #                                            gridspec_kw={'nrows': 1, 'ncols': 3, 'width_ratios': [1, 1, 0.05]},
        #                                            subplot_kw={'projection': '3d', 'zlim': self.zlim})

        S_ax.set_title("S: Susceptible Population", pad=20, fontsize=16)
        I_ax.set_title("I: Infected Population", pad=20, fontsize=16)

        # Adding single color-bar:
        mappable = plt.cm.ScalarMappable(cmap=plt.cm.plasma)
        mappable.set_array(self.model.I_solver.u_n)
        mappable.set_clim(0.0, np.maximum(np.max(self.model.S_solver.u_n), np.max(self.model.I_solver.u_n)))
        plt.colorbar(mappable, cax=color_ax, orientation="vertical")

        # Tighten up the entire figure:
        plt.tight_layout(rect=[0.0, 0.02, 1.0, 0.99])

        # Defining plot arguments to be sent to the
        plot_args = {'rstride': 1, 'cstride': 1, 'linewidth': 0.01, 'cmap': mappable.cmap, 'norm': mappable.norm,
                     'antialiased': True, 'shade': True}

        axes = [S_ax.plot_surface(self.X, self.Y, self.model.S_solver.u_n, **plot_args),
                I_ax.plot_surface(self.X, self.Y, self.model.I_solver.u_n, **plot_args)]

        S_ax.view_init(**self.initial_view)
        time_text = plt.figtext(x=0.02, y=0.94, s=f"T: {0 * self.model.S_solver.k:.3f}")

        I_ax.view_init(**self.initial_view)

        def update_plot(i, axes, plot_args):
            self.model.perform_single_step(i)
            time_text.set_text(f"T: {i * self.model.S_solver.k:.3f}")

            axes[0].remove()
            axes[0] = S_ax.plot_surface(self.X, self.Y, self.model.S_solver.u_n, **plot_args)
            S_ax.elev += self.elev_rotation
            S_ax.azim += self.azim_rotation

            axes[1].remove()
            axes[1] = I_ax.plot_surface(self.X, self.Y, self.model.I_solver.u_n, **plot_args)
            I_ax.elev += self.elev_rotation
            I_ax.azim += self.azim_rotation
            return time_text,

        anim_obj = anim.FuncAnimation(fig=fig, func=update_plot, frames=self.frames, fargs=(axes, plot_args),
                                      interval=int(1000.0/self.fps))

        # Works if you have 'ffmpeg' installed:
        # my_writer = anim.FFMpegWriter(fps=self.fps, codec='libx264',
        #                               extra_args=['-b', '864k', '-tune', 'animation'])
        # anim_obj.save("animations\\test5.mp4", writer=my_writer)

        # Otherwise:
        if save and as_gif:
            print("Saving animation as .gif.")
            anim_obj.save(filename=filename+'.gif', writer='imagemagick')
        elif save:
            print("Saving animation as html.")
            html_writer = anim.HTMLWriter(fps=self.fps)
            anim_obj.save(filename=filename + '.html', writer=html_writer)
        else:
            plt.show()
        return np.array([self.model.S_solver.u_n, self.model.I_solver.u_n])


if __name__ == '__main__':
    print("ran solvers.py")
    pass
