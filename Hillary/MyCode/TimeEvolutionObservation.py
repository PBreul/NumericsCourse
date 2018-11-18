import numpy as np
import setup
import matplotlib.pyplot as plt
import advection_schemes
import Observers
import os
import time
import sys

# This script evolves the linear advection equation in time, with variable initial conditions and numerical schemes
# It can monitor errors and mass as they evolve over time

if __name__ == "__main__":
    """This function is sets the initial conditions and evolves everything in time."""

    savingpath = "Plots/"
    advection_scheme_key_list = ("FTCS", "BTCS", "CTCS", "FTBS", "LaxWendroff", "SemiLagrangien")
    color_dic = {"FTCS": "C0", "BTCS": "C1", "CTCS": "C2", "FTBS": "C3", "LaxWendroff": "C4", "SemiLagrangien": "C5"}

    # Setting initial values
    x_min = 0
    x_max = 20

    grid_points = np.int(sys.argv[1])
    # grid_points = 250

    time_steps = np.int(sys.argv[2])
    # time_steps = 250

    # Courant Parameter
    c = np.float(sys.argv[3])
    # c = 0.2

    # Key word for the initial value. Possible arguments have to match input of initial conditions function
    # init_curve = "step"
    init_curve = sys.argv[4]

    # some parameter for the initial condition, e.g. the center of a gaussian curve, if None, center is automatically
    parameter_initial_cond = None

    # advection schemes which will be used to solve the equation
    used_advection_schemes = ("FTBS", "BTCS", "LaxWendroff")

    dx = (x_max - x_min) / grid_points
    x_grid = np.arange(x_min, x_max, dx)

    # Calling setup function to get initial condition
    u_init = setup.initial_condition(x_grid, init_curve, parameter_initial_cond)

    # Observers which monitor quantities of interest
    ErrorObserver = Observers.ErrorObserver(time_steps)
    MomentObserver = Observers.MomentObserver(time_steps, dx)

    # Define a function for the analytical solution at different time steps, we can give this function later to
    # compare numerics to theory

    def analytical_solution(t):
        analytical = setup.initial_condition((x_grid - c * t * dx) % (x_max - x_min), init_curve,
                                             parameter_initial_cond)
        return analytical


    u_analytic_sol = analytical_solution(time_steps)

    plt.rcParams.update({'font.size': 14})
    fig, ((axs0, axs1), (axs2, axs3)) = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))

    # Evolving and Plotting. Could be over another list
    # If you want something different, than what this function does, implement it here "by hand"

    for advection_scheme_key in used_advection_schemes:
        # Iterate over all advection schemes
        # Call the time evolution, together with the Observers, which measure the quantities of interest while the
        # simulation is running

        plot_color = color_dic[advection_scheme_key]

        t1 = time.time()
        u_num_sol = advection_schemes.time_evolution(u_init, time_steps, c, advection_scheme_key, analytical_solution,
                                                     [ErrorObserver, MomentObserver])
        t2 = time.time()
        print("Time", str(advection_scheme_key), t2 - t1)

        # Plot the solution at last time step, the two error norms, the mass and variance over time step
        axs0.plot(x_grid, u_num_sol, color=plot_color, label=advection_scheme_key)

        # axs1.semilogy(ErrorObserver.linf_array, label="$l_\infty$, " + advection_scheme_key)
        axs1.semilogy(ErrorObserver.l2_array, color=plot_color, label="$l_2$, " + advection_scheme_key)

        axs2.plot((MomentObserver.mass_array - MomentObserver.mass_array[0]) / MomentObserver.mass_array[0],
                  color=plot_color,
                  label=advection_scheme_key)

        axs3.plot((MomentObserver.variance_array - MomentObserver.variance_array[0]) / MomentObserver.variance_array[0],
                  color=plot_color,
                  label=advection_scheme_key)

    axs0.plot(x_grid, u_analytic_sol, "--", color="black", label="analytical")
    axs2.plot(np.arange(time_steps), np.zeros(time_steps), "--", color="black")
    axs3.plot(np.arange(time_steps), np.zeros(time_steps), "--", color="black")

    # Plot Settings
    axs0.set_ylim(-0.1, 1.3)
    axs3.set_ylim(-0.01, 0.01)

    axs0.set_xlabel("x")
    axs0.set_ylabel(r"$\rho$")

    axs1.set_xlabel("# time steps")
    axs1.set_ylabel("Error")

    axs2.set_xlabel("# time steps")
    axs2.set_ylabel("$(M-M_0)/M_0$")

    axs3.set_xlabel("# time steps")
    axs3.set_ylabel("$(V-V_0)/V_0$")

    axs0.legend()
    axs1.legend()
    axs2.legend()
    axs3.legend()

    plt.ticklabel_format(style='sci', axis='both', scilimits=(0, 0), useOffset=False)

    plt.suptitle("c={}, dx={}".format(c, dx), y=0.91)

    if not os.path.exists(savingpath):
        os.makedirs(savingpath)
    # Save Plot
    savingname = savingpath + init_curve + "_c{}dx{}".format(c, dx)
    savingname = savingname.replace(".", "_") + ".pdf"
    plt.savefig(savingname, bbox_inches="tight")
    # plt.show()
