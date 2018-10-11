import numpy as np
import setup
import matplotlib.pyplot as plt
import advection_schemes

# This script evolves the linear advection equation in time, with variable initial conditions and numerical schemes

if __name__ == "__main__":
    """This function is sets the initial conditions and evolves everything in time."""

    # Setting initial values
    x_0 = 0
    x_max = 10
    grid_points = 100
    time_steps = 100

    # Courant Parameter
    c = 0.1

    # Key word for the initial value. Possible arguments have to match input of initial conditions function
    init_curve = "gauss"
    arguments_initial_cond = None

    # Key word for advection scheme
    advection_scheme_key = "CTCS"

    dx = (x_max - x_0) / grid_points
    x_grid = np.linspace(x_0, x_max, grid_points)

    # Calling setup function to get initial condition
    u_grid = setup.initial_condition(x_grid, init_curve, arguments_initial_cond)

    # Calling high level function for the time eveolution. Give advectionscheme, paramater c and time steps as keys.
    # Returns the time evolved array.
    u_grid = advection_schemes.time_evolution(u_grid, time_steps, c, advection_scheme_key)

    plt.plot(x_grid, u_grid)
    plt.show()
