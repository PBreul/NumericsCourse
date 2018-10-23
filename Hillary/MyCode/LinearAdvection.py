import numpy as np
import setup
import matplotlib.pyplot as plt
import advection_schemes
import diagnostics
import Observers


# This script evolves the linear advection equation in time, with variable initial conditions and numerical schemes

def evolve_and_plot(u_init, c, time_steps, advection_scheme_key, u_analytic_sol):
    """This function is for convenience only. Evolves the initial condition in time with the given scheme,
     takes norm and plots the result"""
    # Calling high level function for the time evolution. Give advectionscheme, paramater c and time steps as keys.
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
    grid_points = 50
    time_steps = 76

    # Courant Parameter
    c = 1.0

    # Key word for the initial value. Possible arguments have to match input of initial conditions function
    init_curve = "gauss"
    # some parameter for the initial condition, e.g. the center of a gaussian curve, if None, center is automatically
    parameter_initial_cond = None

    ErrorObserver = Observers.ErrorObserver(time_steps)
    MomentObserver = Observers.MomentObserver(time_steps)

    dx = (x_max - x_min) / grid_points
    x_grid = np.arange(x_min, x_max, dx)

    # Calling setup function to get initial condition
    u_init = setup.initial_condition(x_grid, init_curve, parameter_initial_cond)

    # Define a function for the analytical solution at different time steps, we can give this function later to
    # compare numerics to theory
    def analytical_solution(t):
        analytical = setup.initial_condition((x_grid - c * t * dx) % (x_max - x_min), init_curve,
                                             parameter_initial_cond)
        return analytical

    u_analytic_sol = analytical_solution(time_steps)

    # Evolving and Plotting. Could be over another list
    # If you want something different, than what this function does, implement it here "by hand"

    # ploting_key_list = ("FTCS", )
    # # ploting_key_list = advection_scheme_key_list
    #
    # for advection_scheme_key in ploting_key_list:
    #     evolve_and_plot(u_init, c, time_steps, advection_scheme_key, u_analytic_sol)
    advection_scheme_key = "CTCS"
    u_num_sol = advection_schemes.time_evolution(u_init, time_steps, c, advection_scheme_key, analytical_solution,
                                                 [ErrorObserver, MomentObserver])

    # plt.plot(x_grid, u_analytic_sol, label="analytcical")
    # plt.plot(x_grid, u_num_sol)
    #
    # plt.xlabel(r"$x$")
    # plt.ylabel(r"$u$")
    # plt.xlim(x_min, x_max)
    # plt.ylim(-2, 2)
    # plt.legend(title=r"$c \cdot n_t \cdot \Delta_x = {}$".format(c * time_steps * dx))
    # plt.show()
    plt.plot(ErrorObserver.l2_array)
    plt.plot(ErrorObserver.linf_array)
    plt.show()
