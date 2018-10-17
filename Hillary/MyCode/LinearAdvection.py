import numpy as np
import setup
import matplotlib.pyplot as plt
import advection_schemes
import diagnostics


# This script evolves the linear advection equation in time, with variable initial conditions and numerical schemes

def evolve_and_plot(u_init, c, time_steps, advection_scheme_key, u_analytic_sol):
    """This function is for convenience only. Evolves the initial condition in time with the given scheme,
     takes norm and plots the result"""
    # Calling high level function for the time eveolution. Give advectionscheme, paramater c and time steps as keys.
    # Returns the time evolved array.
    u_num_sol = advection_schemes.time_evolution(u_init, time_steps, c, advection_scheme_key)

    # Calculating the Norms
    l2 = diagnostics.l2_error_norm(u_num_sol, u_analytic_sol)
    linf = diagnostics.l_inf_error_norm(u_num_sol, u_analytic_sol)

    # Plotting
    plt.plot(x_grid, u_num_sol,
             label=advection_scheme_key + ", $l_2$: {}, $l_\infty$: {}".format(np.round(l2, 2), np.round(linf, 2)))


if __name__ == "__main__":
    """This function is sets the initial conditions and evolves everything in time."""

    advection_scheme_key_list = ("FTCS", "BTCS", "CTCS", "FTBS")
    # Setting initial values
    x_min = 0
    x_max = 5
    grid_points = 500
    time_steps = 300

    # Courant Parameter
    c = .5

    # Key word for the initial value. Possible arguments have to match input of initial conditions function
    init_curve = "step"
    # some parameter for the initial condition, e.g. the center of a gaussian curve, if None, center is automatically
    parameter_initial_cond = None

    dx = (x_max - x_min) / grid_points
    x_grid = np.arange(x_min, x_max, dx)

    # Calling setup function to get initial condition
    u_init = setup.initial_condition(x_grid, init_curve, parameter_initial_cond)
    # And a linearly transported version as the analytical solution
    u_analytic_sol = setup.initial_condition((x_grid - c * time_steps * dx)%(x_max-x_min), init_curve,
                                             parameter_initial_cond)

    # Evolving and Plotting. Could be over another list
    # If you want something different, than what this function does, implement it here "by hand"
    ploting_key_list = ("FTCS", "BTCS", "CTCS", "FTBS")
    # ploting_key_list = advection_scheme_key_list

    for advection_scheme_key in ploting_key_list:
        evolve_and_plot(u_init, c, time_steps, advection_scheme_key, u_analytic_sol)

    plt.plot(x_grid, u_analytic_sol, label="analytcical")

    plt.xlabel(r"$x$")
    plt.ylabel(r"$u$")
    plt.xlim(x_min, x_max)
    plt.ylim(-2, 2)
    plt.legend(title=r"$c \cdot n_t \cdot \Delta_x = {}$".format(c * time_steps * dx))
    plt.show()
