import numpy as np


# This file contains functions for setting initial conditions. They take a linspaced python array


def initial_condition(x, initial_name, *args):
    """High level function that takes the grid and a name of the initial condition. It then calls the appropriate
    function. For convenience only."""

    funcdict = {"Gauss": gauss_curve, "gauss": gauss_curve, "step": step_function}

    assert type(initial_name) == str
    assert initial_name in funcdict

    initial_function = funcdict[initial_name]
    u_in = initial_function(x, *args)

    return u_in


def gauss_curve(x, x_0=None):
    """Gives back a simple Gauss-Curve, if position x_0 of center not specified, take the middle. """
    if x_0 is None:
        x_0 = np.mean(x)

    return np.exp(-(x - x_0) ** 2)


def step_function(x, x_0=None):
    """Gives back a step function, if position x_0 of step not specified, take the middle. """

    if x_0 is None:
        x_0 = np.mean(x)

    idmax = (np.abs(x - x_0)).argmin()
    idmin = (idmax - len(x)//2 + 1)%len(x)

    if idmax > idmin:
        u_in = np.zeros(len(x))
        u_in[idmin:idmax] = 1

    else:
        u_in = np.ones(len(x))
        u_in[idmax:idmin] = 0

    return u_in
