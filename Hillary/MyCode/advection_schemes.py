import numpy as np
import Observers


def btcs_m_inv(length, c):
    """Calculates the inverse matrix for the BTCS Scheme"""

    # Set up the matrix as in the notes p.30
    m = np.zeros((length, length))
    for i in range(length):
        m[i, i] = 1
        m[i, i - 1] = -c / 2
        m[i, (i + 1) % length] = c / 2

    # Numerically invert it
    m_inv = np.linalg.inv(m)

    return m_inv


def ftcs(u_old, c, *args):
    """Time Step evolution for FTCS"""
    # using np.roll() implies periodic boundary conditions.
    u_new = u_old - 0.5 * c * (np.roll(u_old, -1) - np.roll(u_old, 1))

    return u_new


def ftbs(u_old, c, *args):
    """Time Step evolution for FTBS"""
    # using np.roll() implies periodic boundary conditions.
    u_new = u_old - c * (u_old - np.roll(u_old, 1))

    return u_new


def ctcs(u_old, c, u_old_old):
    """Time Step evolution for CTCS"""
    # using np.roll() implies periodic boundary conditions.
    u_new = u_old_old - c * (np.roll(u_old, -1) - np.roll(u_old, 1))

    return u_new


def btcs(u_old, c, m_inv, *args):
    """Time Step evolution for CTCS"""
    u_new = np.dot(m_inv, u_old)
    return u_new


def observe(observer, u_grid, analytical_sol_array, t):
    """Checks which type of observer is given and calls the observer function."""
    if type(observer) == Observers.ErrorObserver:
        observer.calculate_errors(u_grid, analytical_sol_array, t)

    elif type(observer) == Observers.MomentObserver:
        observer.calculate_moments(u_grid, t)

    else:
        pass


def time_evolution(u_grid, time_steps, c, advection_scheme_key, analytical_solution_function, observers=None):
    """This function calles the chosen advection routine over and over again until the number of time steps is reached.
     In each step the distribution is evolved in time"""
    # TODO: This function got a bit messy, try to make it cleaner.
    # Dictionary of possible advection schemes, contains function plus information if it is centered in time
    funcdict = {'FTCS': [ftcs, False], 'FTBS': [ftbs, False], "CTCS": [ctcs, True], "BTCS": [btcs, False]}

    # choose the function for the desired advection scheme
    advection_scheme, centered_in_time = funcdict[advection_scheme_key]

    # If the sheme is centered in time, we have to save the array of the last time step and give it to the advection
    #  scheme.
    # TODO: Find a nicer solution for this problem

    # Do the time evolution in a loop
    if centered_in_time is True:

        # We have to do one step with a forward in time scheme before we can use centered in time.
        if time_steps > 0:
            u_old_old = u_grid
            u_grid = ftcs(u_grid, c)

            # If we have observers, execute them
            if observers is not None:
                analyt_sol_array = analytical_solution_function(1)

                # iterate over observer list
                for observer in observers:
                    # Function checks which observe we have and executes it
                    observe(observer,u_grid, analyt_sol_array, 0)

        # Actual time evolution
        for t in range(time_steps - 1):

            u_new = advection_scheme(u_grid, c, u_old_old)
            u_old_old = u_grid.copy()
            u_grid = u_new.copy()

            # If we have observers, execute them
            if observers is not None:
                analyt_sol_array = analytical_solution_function(t + 2)
                # iterate over observer list
                for observer in observers:
                    # Function checks which observe we have and executes it
                    observe(observer,u_grid, analyt_sol_array, t+1)

    else:
        # Check if the Scheme is implicit/BTCS -> Calculate the inverse matrix
        # TODO: This could also be handled in a more elegant way
        if advection_scheme_key == "BTCS":
            m_inv = btcs_m_inv(len(u_grid), c)
        else:
            m_inv = None

        # Actual time evolution
        for t in range(time_steps):
            u_grid = advection_scheme(u_grid, c, m_inv)

            # If we have observers, execute them
            if observers is not None:
                analyt_sol_array = analytical_solution_function(t + 1)

                # iterate over observer list
                for observer in observers:
                    # Function checks which observe we have and executes it
                    observe(observer,u_grid, analyt_sol_array, t)
    return u_grid
