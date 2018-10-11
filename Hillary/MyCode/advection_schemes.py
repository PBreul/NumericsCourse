import numpy as np


def ftcs(u_old, c, *args):
    # using np.roll() implies periodic boundary conditions.
    u_new = u_old - 0.5 * c * (np.roll(u_old, -1) - np.roll(u_old, 1))

    return u_new


def ftbs(u_old, c, *args):
    # using np.roll() implies periodic boundary conditions.
    u_new = u_old - c * (u_old - np.roll(u_old, 1))

    return u_new


def ctcs(u_old, c, u_old_old):
    # using np.roll() implies periodic boundary conditions.
    u_new = u_old_old - c * (np.roll(u_old, -1) - np.roll(u_old, 1))

    return u_new


def time_evolution(u_grid, time_steps, c, advection_scheme_key):
    """This function calles the chosen advection routine over and over again until the number of time steps is reached.
     In each step the distribution is evolved in time"""

    # Dictionary of possible advection schemes, contains function plus information if it is centered in tine
    funcdict = {'FTCS': [ftcs, False], 'FTBS': [ftbs, False], "CTCS": [ctcs, True]}

    # choose the function for the desired advection scheme
    advection_scheme, centered_in_time = funcdict[advection_scheme_key]

    # If the sheme is centered in time, we have to save the array of the last time step and give it to the advection
    #  scheme.
    # TODO: Find a nicer solution for this problem

    # Do the time evolution in a loop
    if centered_in_time is True:
        u_old_old = u_grid.copy()

        for t in range(time_steps):
            u_new = advection_scheme(u_grid, c, u_old_old)
            u_old_old = u_grid.copy()
            u_grid = u_new.copy()

    else:
        u_grid = advection_scheme(u_grid, c)

    return u_grid
