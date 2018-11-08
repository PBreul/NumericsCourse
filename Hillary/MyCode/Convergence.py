import numpy as np
import setup
import matplotlib.pyplot as plt
import advection_schemes
import diagnostics
import os


# This script evolves the linear advection equation in time, with variable initial conditions and numerical schemes
# It will produce plots to test the order of accuracy, that means how does the error scale with the discretisation dx
#  for different schemes.

# Define a function for the analytical solution at different time steps, we can give this function later to
# compare numerics to theory

# Better don't run this with SemiLagrangien, since it is too computationally expensive in the current implementation.

def analytical_solution(x_grid, c, dx, t, init_curve, parameter_initial_cond):
    """Returns the analytical solution to the linear advection problem"""
    analytical = setup.initial_condition((x_grid - c * t * dx) % (x_max - x_min), init_curve, parameter_initial_cond)
    return analytical


if __name__ == "__main__":
    """This function is sets the initial conditions and evolves everything in time. And Plots"""

    savingpath = "Plots/"
    advection_scheme_key_list = ("FTCS", "BTCS", "CTCS", "FTBS", "LaxWendroff", "SemiLagrangien")

    # Setting initial values
    x_min = 0
    x_max = 20

    # Courant Parameter
    # c = np.float(sys.argv[1])
    c = 0.2

    # Key word for the initial value. Possible arguments have to match input of initial conditions function
    init_curve = "gauss"
    # some parameter for the initial condition, e.g. the center of a gaussian curve, if None, center is automatically
    parameter_initial_cond = None

    # Array for number of gridpoints
    gridpoint_array = np.arange(50, 1000, 50)

    # We have to keep the courant number constant, so number of time steps has to scale with number of grid points
    nt_array = gridpoint_array
    # array of discritisations
    dx_array = (x_max - x_min) / gridpoint_array

    # initialise error array
    error_array = np.zeros(len(dx_array))

    # Saving the errors
    error_dict = {"FTCS": None, "BTCS": None, "CTCS": None, "FTBS": None, "LaxWendroff": None, "SemiLagrangien": None}

    # Do this for the following schemes
    for advection_scheme_key in ("FTCS", "BTCS", "CTCS", "FTBS", "LaxWendroff"):

        for i, (dx, time_steps) in enumerate(zip(dx_array, nt_array)):

            # set up discretisation grid
            x_grid = np.arange(x_min, x_max, dx)

            # Calling setup function to get initial condition
            u_init = setup.initial_condition(x_grid, init_curve, parameter_initial_cond)

            # get analytical solution
            u_analytic_sol = analytical_solution(x_grid, c, dx, time_steps, init_curve, parameter_initial_cond)

            # get numerical solution
            u_num_sol = advection_schemes.time_evolution(u_init, time_steps, c, advection_scheme_key,
                                                         analytical_solution)
            # calculate l2 error norm
            error_array[i] = diagnostics.l2_error_norm(u_num_sol, u_analytic_sol)
        error_dict[advection_scheme_key] = error_array.copy()
        # plot the error against the discretisation dx for this scheme
        plt.loglog(dx_array, error_array, ".-", label=advection_scheme_key)

    # Plotting adjustments

    plt.xlabel(r"$\Delta x$")
    plt.ylabel(r"$l_2$")

    # Plot comparing power laws, which are normed to the error of corresponding schemes at smallest dx (-1 because dx
    #  is decreasing)
    plt.loglog(dx_array, dx_array / dx_array[-1] * error_dict["BTCS"][-1], "--", label=r"$\sim\Delta x^1$")

    plt.loglog(dx_array, dx_array / dx_array[-1] * error_dict["FTBS"][-1], "--", label=r"$\sim\Delta x^1$")

    plt.loglog(dx_array, dx_array ** 2 / dx_array[-1] ** 2 * error_dict["LaxWendroff"][-1], "--", label=r"$\sim\Delta x^2$", color="black")

    plt.legend(title="c={}".format(c))

    if not os.path.exists(savingpath):
        os.makedirs(savingpath)
    # Save Plot
    plt.savefig(savingpath + "OrderAccuracy" "c{}.pdf".format(c), bbox_inches="tight")

    plt.show()
