import numpy as np
import setup
import matplotlib.pyplot as plt
import advection_schemes
import Observers
import os
import sys

# This script evolves the linear advection equation in time, with variable initial conditions and numerical schemes

if __name__ == "__main__":
    """This function is sets the initial conditions and evolves everything in time."""

    savingpath = "Plots/"
    advection_scheme_key_list = ("FTCS", "BTCS", "CTCS", "FTBS","LaxWendroff")

    # Setting initial values
    x_min = 0
    x_max = 20
    grid_points = 200
    time_steps = 350

    # Courant Parameter
    #c = np.float(sys.argv[1])
    c = 0.7

    # Key word for the initial value. Possible arguments have to match input of initial conditions function
    init_curve = "gauss"
    # some parameter for the initial condition, e.g. the center of a gaussian curve, if None, center is automatically
    parameter_initial_cond = None

    dx = (x_max - x_min) / grid_points
    x_grid = np.arange(x_min, x_max, dx)

    # Calling setup function to get initial condition
    u_init = setup.initial_condition(x_grid, init_curve, parameter_initial_cond)

    # Observers which monitor quantities of interest
    ErrorObserver = Observers.ErrorObserver(time_steps)
    MomentObserver = Observers.MomentObserver(time_steps)

    # Define a function for the analytical solution at different time steps, we can give this function later to
    # compare numerics to theory

    def analytical_solution(t):
        analytical = setup.initial_condition((x_grid - c * t * dx) % (x_max - x_min), init_curve,
                                             parameter_initial_cond)
        return analytical

    u_analytic_sol = analytical_solution(time_steps)

    fig, axs = plt.subplots(3, 1, figsize=(14, 10))

    # Evolving and Plotting. Could be over another list
    # If you want something different, than what this function does, implement it here "by hand"

    for advection_scheme_key in advection_scheme_key_list:
        # Iterate over all advection schemes
        # Call the time evolution, together with the Observers, which measure the quantities of interest while the
        # simulation is running
        u_num_sol = advection_schemes.time_evolution(u_init, time_steps, c, advection_scheme_key, analytical_solution,
                                                     [ErrorObserver, MomentObserver])

        # Plot the solution at last time step, the two error norms and the mass over time step
        axs[0].plot(x_grid, u_num_sol, label=advection_scheme_key)
        axs[1].semilogy(ErrorObserver.linf_array, label="$l_\inf$, " + advection_scheme_key)
        axs[1].semilogy(ErrorObserver.l2_array, label="$l_2$, " + advection_scheme_key)
        axs[2].plot(MomentObserver.mass_array * dx, label=advection_scheme_key)

    axs[0].plot(x_grid, u_analytic_sol, "--", color="black", label="analytical")

    # Plot Settings
    axs[0].set_ylim(-0.1, 1.1)
    axs[1].set_ylim(0.01, 10)
    axs[2].set_ylim(np.sqrt(np.pi) - 0.01, np.sqrt(np.pi) + 0.03)

    axs[0].set_xlabel("x")
    axs[0].set_ylabel("u")

    axs[1].set_xlabel("# time steps")
    axs[1].set_ylabel("Error")

    axs[2].set_xlabel("# time steps")
    axs[2].set_ylabel("Moment")

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    axs[0].set_title("c={}, dx={}".format(c, dx))

    if not os.path.exists(savingpath):
        os.makedirs(savingpath)
    # Save Plot
    #plt.savefig(savingpath + "c{}dx{}.pdf".format(c, dx), bbox_inches="tight")
    plt.show()
